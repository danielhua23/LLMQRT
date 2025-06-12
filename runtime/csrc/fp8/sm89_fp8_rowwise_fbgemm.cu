// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include "sm89_fp8_gemm.h"

#include <cute/tensor.hpp>
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>
#include <cutlass/numeric_types.h>
#include <cutlass/trace.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/version.h>

#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>

#include <cute/atom/mma_atom.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <optional>
// #include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
// Two warninngs in Cutlass included header files
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wset-but-not-used")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-but-set-parameter")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wmissing-field-initializers")

// Determine if the architecture supports rowwise scaled mm
// Currently failing on windows with:
// https://github.com/NVIDIA/cutlass/issues/1571
#if !defined(USE_ROCM) && !defined(_WIN32) && defined(CUDA_VERSION) && CUDA_VERSION >= 12000 && defined(CUDA_ARCH) && CUDA_ARCH >= 890

#define BUILD_ROWWISE_FP8_KERNEL
#endif

#if defined(BUILD_ROWWISE_FP8_KERNEL)

template <typename T>
struct get_torch_DtypeOutput {
    static_assert(sizeof(T) == 0, "Unsupported type for torch::Dtype mapping");
};

// 特化 float -> torch::kFloat32
template <>
struct get_torch_DtypeOutput<float> {
    static constexpr torch::Dtype value = torch::kFloat32;
};

// 特化 cutlass::half_t -> torch::kFloat16
template <>
struct get_torch_DtypeOutput<cutlass::half_t> {
    static constexpr torch::Dtype value = torch::kFloat16;
};

// 特化 cutlass::bfloat16_t -> torch::kBFloat16
template <>
struct get_torch_DtypeOutput<cutlass::bfloat16_t> {
    static constexpr torch::Dtype value = torch::kBFloat16;
};

// common utils
using DtypeScale = float;
using DtypeAccum = float;
using DtypeEpilogue = float;
using DtypeOutput = cutlass::bfloat16_t;
// using torch_DtypeOutput = get_torch_DtypeOutput<DtypeOutput>::value;
constexpr torch::Dtype torch_DtypeOutput = get_torch_DtypeOutput<DtypeOutput>::value;
// for SM90
// using Multiply = cutlass::epilogue::fusion::Sm90Compute<
//     cutlass::multiplies,
//     DtypeEpilogue,
//     DtypeEpilogue,
//     cutlass::FloatRoundStyle::round_to_nearest>;

// using Add = cutlass::epilogue::fusion::Sm90Compute<
//     cutlass::plus,
//     DtypeEpilogue,
//     DtypeEpilogue,
//     cutlass::FloatRoundStyle::round_to_nearest>;

// using Cast = cutlass::epilogue::fusion::Sm90Compute<
//     cutlass::epilogue::thread::Identity,
//     DtypeOutput,
//     DtypeEpilogue,
//     cutlass::FloatRoundStyle::round_to_nearest>;

// template <bool LargeTile, bool FastAccum>
// struct Schedule;

// template <>
// struct Schedule</*LargeTile=*/false, /*FastAccum=*/false> {
//   using type = cutlass::gemm::KernelTmaWarpSpecialized;
//   using epilogue_type = cutlass::epilogue::TmaWarpSpecialized;
// };

// template <>
// struct Schedule</*LargeTile=*/true, /*FastAccum=*/false> {
//   // For a 128x128x128 tile with fastAccum = false, using
//   // pingpong schedule will lead to spilling, and WarpSpecialized w/o pingpong
//   // is slow
//   using type = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
//   using epilogue_type = cutlass::epilogue::TmaWarpSpecializedCooperative;
// };

// template <>
// struct Schedule</*LargeTile=*/false, /*FastAccum=*/true> {
//   using type = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
//   using epilogue_type = cutlass::epilogue::TmaWarpSpecialized;
// };

// template <>
// struct Schedule</*LargeTile=*/true, /*FastAccum=*/true> {
//   using type = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
//   using epilogue_type = cutlass::epilogue::TmaWarpSpecialized;
// };

int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

int round_up_to_nearest_multiple(int a, int b) {
  return ceildiv(a, b) * b;
}

// Cutlass rowwise kernel for SM89
template <
    typename ThreadblockShape,
    typename WarpShape,
    int NumStages,
    typename FastAccum,
    typename DtypeA,
    typename DtypeB,
    typename DtypeBias>
torch::Tensor f8f8bf16_rowwise_impl_sm89(
    torch::Tensor XQ, // FP8 row major (M,K)
    torch::Tensor WQ, // FP8 col major (K,N)
    torch::Tensor x_scale, // FP32
    torch::Tensor w_scale, // FP32
    std::optional<torch::Tensor> bias // BF16
){
    // at::Tensor XQ, // FP8 row major (M,K)
    // at::Tensor WQ, // FP8 col major (K,N)
    // at::Tensor x_scale,
    // at::Tensor w_scale,
    // std::optional<at::Tensor> bias,
    // at::Tensor out) {
  int M = XQ.size(0);
  int N = WQ.size(1);
  int K = XQ.size(1);

  using LayoutInputA = cutlass::layout::RowMajor;
  constexpr int AlignmentInputA = 16 / sizeof(DtypeA);

  using LayoutInputB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputB = 16 / sizeof(DtypeB);

  using LayoutOutput = cutlass::layout::RowMajor;
  constexpr int AlignmentOutput = 16 / sizeof(DtypeOutput);

  // Tag indicating the minimum SM that supports the intended feature
  using ArchTag = cutlass::arch::Sm89;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;

  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  using Operator = std::conditional_t<
      FastAccum::value,
      cutlass::arch::OpMultiplyAddFastAccum,
      cutlass::arch::OpMultiplyAdd>;
  constexpr auto NumEVTEpilogueStages = 1;

  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape,
          WarpShape,
          DtypeOutput,
          AlignmentOutput,
          NumEVTEpilogueStages>;

  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputTileThreadMap, DtypeScale,
      cute::Stride<cute::_1, cute::_0, int64_t>>;
  using XScaleArguments = typename XScale::Arguments;

  using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, DtypeScale,
      cute::Stride<cute::_0, cute::_1, int64_t>/*StrideMNL*/>;
  using WScaleArguments = typename WScale::Arguments;

  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, DtypeBias,
      cute::Stride<cute::_0, cute::_1, int64_t>>;
  using BiasArguments = typename Bias::Arguments;

  using ApplyXScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, DtypeEpilogue, DtypeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using EVTApplyXScale = cutlass::epilogue::threadblock::Sm80EVT<
      ApplyXScale,// NodeOp即根节点
      Accum,//childOp
      XScale>;//childOp

  using ApplyWScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, DtypeEpilogue/*elementOut*/, DtypeEpilogue/*elementCompute*/,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using EVTApplyWScale = cutlass::epilogue::threadblock::Sm80EVT<
      ApplyWScale, // NodeOp
      EVTApplyXScale, //childOp
      WScale>;//childOp

  using ApplyBias = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, DtypeEpilogue, DtypeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using EVTApplyBias = cutlass::epilogue::threadblock::Sm80EVT<
      ApplyBias,
      EVTApplyWScale,
      Bias>;

  using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, DtypeOutput,
      cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, int64_t> // StrideMNL
  >;

  using EVTOutput = cutlass::epilogue::threadblock::Sm80EVT<
      Output,
      EVTApplyBias>;

  using EVTKernel = // at::cuda::detail::enable_2x_kernel_for_sm89< ATen里面没找到这个函数，干脆注释掉得了
      typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
          DtypeA, LayoutInputA, cutlass::ComplexTransform::kNone, AlignmentInputA,
          DtypeB, LayoutInputB, cutlass::ComplexTransform::kNone, AlignmentInputB,
          DtypeOutput, LayoutOutput, AlignmentOutput,
          DtypeAccum,
          DtypeEpilogue,
          OperatorClass,
          ArchTag,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          EVTOutput,
          ThreadblockSwizzle,
          NumStages,
          Operator,
          NumEVTEpilogueStages>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<EVTKernel>;

  cutlass::gemm::GemmCoord problem_size(M, N, K);
  constexpr auto SplitKFactor = 1;

  XScaleArguments x_scale_arguments{
      (DtypeScale*)x_scale.data_ptr(),
      DtypeScale(1),
      {cute::_1{}, cute::_0{}, problem_size.m()}
  };
  WScaleArguments w_scale_arguments{
      (DtypeScale*)w_scale.data_ptr(),
      DtypeScale(1),
      {cute::_0{}, cute::_1{}, problem_size.n()}
  };
  BiasArguments bias_arguments{
      bias.has_value() ? reinterpret_cast<DtypeBias*>(bias->data_ptr()) : nullptr,
      DtypeBias(0),
      {cute::_0{}, cute::_1{}, problem_size.n()}
  };

  auto out = torch::empty({problem_size.m(), problem_size.n()},
                          torch::dtype(torch_DtypeOutput).device(XQ.device()) );

  typename Output::Arguments output_arguments{
    (DtypeOutput*)out.data_ptr(),
    {problem_size.n(), cute::_1{}, problem_size.mn().product()}
  };
  typename EVTOutput::Arguments callback_arguments{
    {
      {
        {
          {},                 // Accum
          x_scale_arguments,  // XScale
          {}                  // ApplyXScale
        },                    // EVTApplyXScale
        w_scale_arguments,    // WScale
        {}                    // ApplyWScale
      },                      // EVTApplyWScale
      bias_arguments,         // Bias
      {}                      // ApplyBias
    },                        // EVTApplyBias
    output_arguments          // Output
  };                          // EVTOutput

  typename Gemm::Arguments arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
    SplitKFactor,
    callback_arguments,           // arguments of EVT callbacks
    (DtypeA*)XQ.data_ptr(),
    (DtypeB*)WQ.data_ptr(),
    nullptr,                      // ptr C (unused)
    nullptr,                      // ptr D (unused)
    problem_size.mk().product(),  // batch stride A
    problem_size.nk().product(),  // batch stride B
    0,                            // batch stride C (unused)
    0,                            // batch stride D (unused)
    problem_size.k(),             // stride A
    problem_size.k(),             // stride B
    0,                            // stride C (unused)
    0);                           // stride D (unused)

  Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  auto workspace = XQ.new_empty(
      {static_cast<int64_t>(workspace_size)},
      at::TensorOptions().dtype(at::kByte));

  // Check the problem size is supported or not
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm.initialize(arguments, workspace.data_ptr());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm(at::cuda::getCurrentCUDAStream());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("cutlass cannot run") +
        cutlass::cutlassGetStatusString(status));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <typename... Types>
torch::Tensor dispatch_fp8_rowwise_kernel_sm89(
    torch::Tensor XQ, // FP8
    torch::Tensor WQ, // FP8
    std::optional<torch::Tensor> bias, // BF16
    torch::Tensor x_scale, // FP32
    torch::Tensor w_scale // FP32
){
    // at::Tensor XQ,
    // at::Tensor WQ,
    // at::Tensor x_scale,
    // at::Tensor w_scale,
    // std::optional<at::Tensor> bias,
    // at::Tensor out) {
  int M = XQ.size(0);

  if (M <= 16) {
    return f8f8bf16_rowwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<16, 64, 128>,
        /*WarpShape=*/cutlass::gemm::GemmShape<16, 64, 64>,
        /*NumStages=*/5,
        Types...>(XQ, WQ, x_scale, w_scale, bias);
  } else if (M <= 32) {
    return f8f8bf16_rowwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<32, 64, 128>,
        /*WarpShape=*/cutlass::gemm::GemmShape<16, 64, 64>,
        /*NumStages=*/5,
        Types...>(XQ, WQ, x_scale, w_scale, bias);
  } else if (M <= 64) {
    return f8f8bf16_rowwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<64, 64, 128>,
        /*WarpShape=*/cutlass::gemm::GemmShape<32, 64, 64>,
        /*NumStages=*/5,
        Types...>(XQ, WQ, x_scale, w_scale, bias);
  } else if (M <= 256) {
    return f8f8bf16_rowwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<64, 128, 128>,
        /*WarpShape=*/cutlass::gemm::GemmShape<64, 64, 64>,
        /*NumStages=*/3,
        Types...>(XQ, WQ, x_scale, w_scale, bias);
  } else {
    return f8f8bf16_rowwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<128, 128, 64>,
        /*WarpShape=*/cutlass::gemm::GemmShape<64, 64, 64>,
        /*NumStages=*/5,
        Types...>(XQ, WQ, x_scale, w_scale, bias);
  }
}

template <typename... Types>
torch::Tensor dispatch_fp8_rowwise_kernel_on_sm(
    torch::Tensor XQ, // FP8
    torch::Tensor WQ, // FP8
    std::optional<torch::Tensor> bias, // BF16
    torch::Tensor x_scale, // FP32
    torch::Tensor w_scale){ // FP32
    // float alpha,
    // float beta){
    // at::Tensor XQ,
    // at::Tensor WQ,
    // at::Tensor x_scale,
    // at::Tensor w_scale,
    // std::optional<at::Tensor> bias,
    // at::Tensor out) {
  cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  const bool sm89 = properties != nullptr && properties->major == 8 && properties->minor == 9;
  const bool sm9x = properties != nullptr && properties->major == 9;
  if (!(sm89 || sm9x)) {
    TORCH_CHECK(
        false, "Rowwise scaling only currently supported on SM89 and upper device");
    // placeholder, 无实际意义，避免报错error: return-statement with no value, in function returning ‘at::Tensor’ [-fpermissive]
    auto out = torch::empty({1,1},
                          torch::dtype(torch_DtypeOutput).device(XQ.device()) );
    return out;
//   }

//   if (sm9x) {
//     dispatch_fp8_rowwise_kernel_on_cluster_size_and_transpose<Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  } else {
    return dispatch_fp8_rowwise_kernel_sm89<Types...>(XQ, WQ, bias, x_scale, w_scale);//, out);
  }
}

template <typename... Types>
torch::Tensor dispatch_fp8_rowwise_kernel_on_fast_accum(
    torch::Tensor XQ, // FP8
    torch::Tensor WQ, // FP8
    std::optional<torch::Tensor> bias, // BF16
    torch::Tensor x_scale, // FP32
    torch::Tensor w_scale, // FP32
    // float alpha,
    // float beta,
    bool use_fast_accum){
    // at::Tensor XQ,
    // at::Tensor WQ,
    // at::Tensor x_scale,
    // at::Tensor w_scale,
    // std::optional<at::Tensor> bias,
    // bool use_fast_accum,
    // at::Tensor out) {
  if (use_fast_accum) {
    return dispatch_fp8_rowwise_kernel_on_sm<
        std::true_type,
        Types...>(XQ, WQ, bias, x_scale, w_scale);//, out);
  } else {
    return dispatch_fp8_rowwise_kernel_on_sm<
        std::false_type,
        Types...>(XQ, WQ, bias, x_scale, w_scale);//, out);
  }
}

template <typename... Types>
torch::Tensor dispatch_fp8_rowwise_kernel_on_input_dtypes(
    torch::Tensor XQ, // FP8
    torch::Tensor WQ, // FP8
    std::optional<torch::Tensor> bias, // BF16
    torch::Tensor x_scale, // FP32
    torch::Tensor w_scale, // FP32
    bool use_fast_accum){
    // at::Tensor XQ,
    // at::Tensor WQ,
    // at::Tensor x_scale,
    // at::Tensor w_scale,
    // std::optional<at::Tensor> bias,
    // bool use_fast_accum,
    // at::Tensor out) {
  if (XQ.dtype() == at::kFloat8_e5m2) {
    return dispatch_fp8_rowwise_kernel_on_fast_accum<
        cutlass::float_e5m2_t,
        cutlass::float_e4m3_t,
        Types...>(XQ, WQ, bias, x_scale, w_scale, use_fast_accum);//, out);
  } else {
    return dispatch_fp8_rowwise_kernel_on_fast_accum<
        cutlass::float_e4m3_t,
        cutlass::float_e4m3_t,
        Types...>(XQ, WQ, bias, x_scale, w_scale, use_fast_accum);//, out);
  }
}
// 支持bf16和fp32 bias
torch::Tensor dispatch_fp8_rowwise_kernel_on_bias_dtype(
    torch::Tensor XQ, // FP8
    torch::Tensor WQ, // FP8
    std::optional<torch::Tensor> bias, // BF16
    torch::Tensor x_scale, // FP32
    torch::Tensor w_scale, // FP32
    bool use_fast_accum){
    // at::Tensor XQ,
    // at::Tensor WQ,
    // at::Tensor x_scale,
    // at::Tensor w_scale,
    // std::optional<at::Tensor> bias,
    // bool use_fast_accum,
    // at::Tensor out) {
  if (bias.has_value() && bias->dtype() == at::kBFloat16) {
    return dispatch_fp8_rowwise_kernel_on_input_dtypes<
        cutlass::bfloat16_t>
        (XQ, WQ, bias, x_scale, w_scale, use_fast_accum);//, out);
  } else {
    return dispatch_fp8_rowwise_kernel_on_input_dtypes<
        float>
        //Types...>
        (XQ, WQ, bias, x_scale, w_scale, use_fast_accum);//, out);
  }
}

void check_inputs(
    torch::Tensor input,  // FP8
    torch::Tensor weight, // FP8
    std::optional<torch::Tensor> bias,   // BF16/FP16
    torch::Tensor scale_a, // FP32
    torch::Tensor scale_b // FP32
    ) {
  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(input.device() == weight.device());
  TORCH_CHECK(scale_a.device() == input.device());
  TORCH_CHECK(scale_b.device() == weight.device());

  TORCH_CHECK(input.dtype() == at::kFloat8_e4m3fn || input.dtype() == at::kFloat8_e5m2);
  TORCH_CHECK(weight.dtype() == at::kFloat8_e4m3fn);
  TORCH_CHECK(scale_a.dtype() == at::kFloat);
  TORCH_CHECK(scale_b.dtype() == at::kFloat);

  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(weight.dim() == 2);
  TORCH_CHECK(input.size(1) == weight.size(0)); // a row major b col major
  TORCH_CHECK(scale_a.dim() == 2);
  TORCH_CHECK(scale_b.dim() == 2);
  TORCH_CHECK(scale_a.size(0) == input.size(0));
  TORCH_CHECK(scale_a.size(1) == 1);
  TORCH_CHECK(scale_b.size(0) == 1, "Expected scale_b.size(0) == 1, but got", scale_b.size(0));
  TORCH_CHECK(scale_b.size(1) == weight.size(1));

  TORCH_CHECK(input.stride(1) == 1);
  TORCH_CHECK(input.stride(0) >= input.size(1));
  TORCH_CHECK(weight.stride(0) == 1);
  TORCH_CHECK(weight.stride(1) >= weight.size(0));
  TORCH_CHECK(scale_a.stride(0) == 1);
  TORCH_CHECK(scale_b.stride(1) == 1);

  if (bias.has_value()) {
    TORCH_CHECK(bias->device() == weight.device());
    TORCH_CHECK(bias->dtype() == at::kFloat || bias->dtype() == at::kBFloat16);
    TORCH_CHECK(bias->dim() == 1);
    TORCH_CHECK(bias->size(0) == weight.size(1));
    TORCH_CHECK(bias->stride(0) == 1);
  }

  // TORCH_CHECK(out.device() == a.device());
  // TORCH_CHECK(out.dtype() == at::kBFloat16);
  // TORCH_CHECK(out.dim() == 2);
  // TORCH_CHECK(out.size(0) == a.size(0));
  // TORCH_CHECK(out.size(1) == b.size(1));
  // TORCH_CHECK(out.stride(1) == 1);
  // TORCH_CHECK(out.stride(0) >= out.size(1));
}

// } // namespace

#endif // !defined(USE_ROCM)

// namespace at::cuda::detail {
torch::Tensor f8f8bf16_rowwise(
    torch::Tensor XQ, // FP8
    torch::Tensor WQ, // FP8
    std::optional<torch::Tensor> bias, // BF16
    torch::Tensor x_scale, // FP32
    torch::Tensor w_scale, // FP32
    bool use_fast_accum){
    // at::Tensor XQ, // FP8
    // at::Tensor WQ, // FP8
    // at::Tensor x_scale, // FP32
    // at::Tensor w_scale, // FP32
    // std::optional<at::Tensor> bias, // BF16
    // bool use_fast_accum,
    // at::Tensor& out) {
#if defined(BUILD_ROWWISE_FP8_KERNEL)
  check_inputs(XQ, WQ,  bias, x_scale, w_scale);

  return dispatch_fp8_rowwise_kernel_on_bias_dtype(
      XQ, WQ,  bias, x_scale, w_scale, use_fast_accum);
#else // BUILD_ROWWISE_FP8_KERNEL
  TORCH_CHECK(
      false, "Rowwise scaling is not currently compiled, if you want to use fp8 sm89 kernel, pls add -DBUILD_ROWWISE_FP8_KERNEL");
  return;
#endif
}