// adapted from https://github.com/vllm-project/vllm/tree/main/csrc/quantization/cutlass_w8a8
// 计划支持一个cutlass ada fp8 rowwise gemm以match pertoken perchannel fp8 quant
// 计划支持一个cutlass ada fp8 tensorwise gemm以match per tensor fp8 quant
#include "fp8_gemm.h"

// clang-format on

using namespace cute;
// kernel guard，保证kernel在sm89上call
template <typename Kernel>
struct enable_sm89_to_sm90 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 890 && __CUDA_ARCH__ < 900
    Kernel::invoke(std::forward<Args>(args)...);
#endif
  }
};
template <typename Arch, template <typename> typename ArchGuard,
          typename ElementAB_, typename ElementD_,
          template <typename, typename> typename Epilogue_, typename TileShape,
          typename WarpShape, typename InstructionShape, int32_t MainLoopStages,
          typename FP8MathOperator = cutlass::arch::OpMultiplyAdd>
// kernel，含其配置过程
struct cutlass_2x_gemm {
  using ElementAB = ElementAB_;
  using ElementD = ElementD_;

  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;

  using Operator =
        typename std::conditional<std::is_same_v<ElementAB, int8_t>,
                                cutlass::arch::OpMultiplyAddSaturate,
                                FP8MathOperator>::type;

  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          TileShape, WarpShape, float, 4, 1 /* epilogue stages */
          >;

  using Epilogue = Epilogue_<ElementD, OutputTileThreadMap>;
  using EVTCompute = typename Epilogue::EVTCompute;

  using D = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementD, cutlass::FloatRoundStyle::round_to_nearest,
      Stride<int64_t, Int<1>, Int<0>>>;

  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute>;

  // These are the minimum alignments needed for the kernels to compile
  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD = 4;

  // clang-format off
  using RowMajor = typename cutlass::layout::RowMajor;
  using ColumnMajor = typename cutlass::layout::ColumnMajor;
  using KernelType =
    ArchGuard<typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementAB, RowMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
      ElementAB, ColumnMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
      float, cutlass::layout::RowMajor, AlignmentCD,
      ElementAcc, float, cutlass::arch::OpClassTensorOp,
      Arch,
      TileShape, WarpShape, InstructionShape,
      EVTD,
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
      MainLoopStages, Operator,
      1 /* epilogue stages */
      >::GemmKernel>;
  // clang-format on

  using Op = cutlass::gemm::device::GemmUniversalAdapter<KernelType>;
};

// kernel caller，第一个模板参数就是cutlass 2x gemm或fp8 callback gemm
template <typename Gemm, typename... EpilogueArgs>
inline void cutlass_gemm_caller(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& b,
                                EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);
  cutlass::gemm::GemmCoord problem_size{m, n, k};

  int64_t lda = a.stride(0);
  int64_t ldb = b.stride(1);
  int64_t ldc = out.stride(0);

  using StrideC = Stride<int64_t, Int<1>, Int<0>>;
  StrideC c_stride{ldc, Int<1>{}, Int<0>{}};

  auto a_ptr = static_cast<ElementAB const*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB const*>(b.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());

  typename Gemm::D::Arguments d_args{c_ptr, c_stride};

  using Epilogue = typename Gemm::Epilogue;
  auto evt_args =
      Epilogue::prepare_args(std::forward<EpilogueArgs>(epilogue_params)...);

  typename Gemm::EVTD::Arguments epilogue_args{
      evt_args,
      d_args,
  };

  typename Gemm::Op::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,  // universal mode
      problem_size,                                           // problem size
      1,                                                      // batch count
      epilogue_args,
      a_ptr,
      b_ptr,
      nullptr,
      nullptr,
      0,
      0,
      0,
      0,
      lda,
      ldb,
      ldc,
      ldc};

  // Launch the CUTLASS GEMM kernel.
  typename Gemm::Op gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  CUTLASS_CHECK(gemm_op.can_implement(args));
  cutlass::Status status = gemm_op(args, workspace.data_ptr(), stream);
  CUTLASS_CHECK(status);
}
//根据smem size是否满足要求来决定call sm89_fp8_fallback_gemm还是cutlass_2x_gemm，这俩的区别在于后者tile shape比较大，对smem要求更大
template <typename Gemm, typename FallbackGemm, typename... EpilogueArgs>
inline void fallback_cutlass_gemm_caller(torch::Tensor& out,
                                         torch::Tensor const& a,
                                         torch::Tensor const& b,
                                         EpilogueArgs&&... args) {
  // In some cases, the GPU isn't able to accommodate the
  // shared memory requirements of the Gemm. In such cases, use
  // the FallbackGemm instead.
  static const int max_shared_mem_per_block_opt_in =
      get_cuda_max_shared_memory_per_block_opt_in(0);

  size_t const gemm_shared_mem_size =
      sizeof(typename Gemm::KernelType::SharedStorage);
  size_t const fallback_gemm_shared_mem_size =
      sizeof(typename FallbackGemm::KernelType::SharedStorage);

  if (gemm_shared_mem_size <= max_shared_mem_per_block_opt_in) {
    return cutlass_gemm_caller<Gemm>(out, a, b,
                                     std::forward<EpilogueArgs>(args)...);
  } else {
    TORCH_CHECK(fallback_gemm_shared_mem_size <=
                max_shared_mem_per_block_opt_in);
    return cutlass_gemm_caller<FallbackGemm>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  }
}

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_fallback_gemm {
  // Shared Memory required by this Gemm - 61440 bytes
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<64, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAdd;
  using Cutlass2xGemm =
      cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90, InType, OutType,
                      Epilogue, TileShape, WarpShape, InstructionShape, 5,
                      FP8MathOperator>;
};

struct sm89_fp8_config_default {
  // M in (128, inf)
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);

    using FallbackGemm =
        typename sm89_fp8_fallback_gemm<InType, OutType,
                                        Epilogue>::Cutlass2xGemm;

    uint32_t const n = out.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 4096) {
      using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;

      return fallback_cutlass_gemm_caller<
          cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else if (np2 <= 8192) {
      using TileShape = typename cutlass::gemm::GemmShape<256, 128, 64>;

      return fallback_cutlass_gemm_caller<
          cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 3, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);

    } else {
      using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;

      return fallback_cutlass_gemm_caller<
          cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

struct sm89_fp8_config_M128 {
  // M in (32, 128]
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);

    using FallbackGemm =
        typename sm89_fp8_fallback_gemm<InType, OutType,
                                        Epilogue>::Cutlass2xGemm;

    uint32_t const n = out.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 8196) {
      using TileShape = typename cutlass::gemm::GemmShape<64, 64, 128>;
      using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 64>;
      using FP8MathOperator = typename cutlass::arch::OpMultiplyAdd;

      return fallback_cutlass_gemm_caller<
          cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else if (np2 <= 16384) {
      using TileShape = typename cutlass::gemm::GemmShape<64, 128, 128>;
      using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
      using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;

      return fallback_cutlass_gemm_caller<
          cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 3, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      using TileShape = typename cutlass::gemm::GemmShape<64, 64, 128>;
      using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 64>;
      using FP8MathOperator = typename cutlass::arch::OpMultiplyAdd;

      return fallback_cutlass_gemm_caller<
          cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

struct sm89_fp8_config_M32 {
  // M in (16, 32]
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);

    using FallbackGemm =
        typename sm89_fp8_fallback_gemm<InType, OutType,
                                        Epilogue>::Cutlass2xGemm;

    uint32_t const n = out.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 8192) {
      using TileShape = typename cutlass::gemm::GemmShape<32, 64, 128>;
      using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;

      return fallback_cutlass_gemm_caller<
          cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else if (np2 <= 16384) {
      using TileShape = typename cutlass::gemm::GemmShape<32, 128, 128>;
      using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 64>;

      return fallback_cutlass_gemm_caller<
          cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 4, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      using TileShape = typename cutlass::gemm::GemmShape<32, 64, 128>;
      using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;

      return fallback_cutlass_gemm_caller<
          cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

struct sm89_fp8_config_M16 {
  // M in [1, 16]
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  static const int32_t MainLoopStages = 5;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);

    using FallbackGemm =
        typename sm89_fp8_fallback_gemm<InType, OutType,
                                        Epilogue>::Cutlass2xGemm;

    uint32_t const n = out.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 8192) {
      using TileShape = typename cutlass::gemm::GemmShape<16, 64, 128>;

      return fallback_cutlass_gemm_caller<
          cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, MainLoopStages,
                                FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else if (np2 <= 24576) {
      using TileShape = typename cutlass::gemm::GemmShape<16, 128, 64>;

      return fallback_cutlass_gemm_caller<
          cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, MainLoopStages,
                                FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
  }
};
};
//根据number tokens分发到不同的tile shape和warp shape和numstages
template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm89_fp8_dispatch(torch::Tensor& out,
                                           torch::Tensor const& a,
                                           torch::Tensor const& b,
                                           EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  uint32_t const m = a.size(0);// number tokens
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(16), next_pow_2(m));  // next power of 2

  if (mp2 <= 16) {
    // M in [1, 16]
    return sm89_fp8_config_M16::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 32) {
    // M in (16, 32]
    return sm89_fp8_config_M32::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
//   } else if (mp2 <= 64) {
//     // M in (32, 64]
//     return sm89_fp8_config_M64::dispatch<InType, OutType, Epilogue>(
//         out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 128) {
    // M in (32, 128]
    return sm89_fp8_config_M128::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
//   } else if (mp2 <= 256) {
//     // M in (128, 256]
//     return sm89_fp8_config_M256::dispatch<InType, OutType, Epilogue>(
//         out, a, b, std::forward<EpilogueArgs>(args)...);
  } else {
    // M in (128, inf)
    return sm89_fp8_config_default::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  }
}
// 检查a b的dtype为fp8，分发到output type为bf16和fp16两种情况
template <template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm89_epilogue(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {

TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

if (out.dtype() == torch::kBFloat16) { // for bf16 model
    return cutlass_gemm_sm89_fp8_dispatch<cutlass::float_e4m3_t,
                                        cutlass::bfloat16_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
} else {// for bf16 model
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_sm89_fp8_dispatch<cutlass::float_e4m3_t,
                                        cutlass::half_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
}
}

// 判断是否存在bias，检查scale的dtype为fp32，把scale打包为EpilogueArgs
// 目前只支持symm quant，不支持zp
// https://github.com/vllm-project/vllm/blob/561b77a0d608a9059318d6cff9f0975439880d77/csrc/quantization/cutlass_w8a8/Epilogues.md
void cutlass_scaled_mm_sm89(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,//[M,1] col vector
                            torch::Tensor const& b_scales,//[1,N] row vector
                            std::optional<torch::Tensor> const& bias/*row vector [1, out feats]*/) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  if (bias) {
    TORCH_CHECK(bias->dtype() == out.dtype(),
                "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogueBias>( // symmetric quant for per tensor/per channel act with bias
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogue>( // symmetric quant for per tensor/per channel act without bias
        out, a, b, a_scales, b_scales);
  }
}
// `ScaledEpilogueAzp`: asymmetric per-tensor quantization for activations, supports bias.
// `ScaledEpilogueAzpPerToken`: asymmetric per-token quantization for activations, supports bias.