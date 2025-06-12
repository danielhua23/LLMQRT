#include "sq_gemm.h"
#include <ATen/ATen.h>
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

#include <iostream>
// should check bias and output type fp32 or int32, w/ or w/o scaling
// sq里面的linear都带alpha和beta两个scale，只有W8A8B32O32LinearWithoutScaling不带
// W8A8B8O8LinearReLU用在mlp.fc1，输出的s8，喂到mlp.fc2继续做s8gemm
// W8A8BFP32OFP32Linear用在mlp.fc2和attn.out_proj
// W8A8B8O8Linear用在qkv，output为int8的原因是这个项目里面把qk_bmm设为了s8s8f32
// PS: 我们只需要用W8A8BBF16OBF16Linear, 我们不涉及mlp里面的act以及int8 bmm
// used by all linear, return BF16/FP16
template <torch::Dtype dtype>
struct ElementOutputType {
    // 默认报错（如果传入非 FP16/BF16 类型）
    static_assert(dtype == torch::kFloat16 || dtype == torch::kBFloat16, "Unsupported dtype");
};

template <>
struct ElementOutputType<torch::kFloat16> {
    using type = cutlass::half_t;
};

template <>
struct ElementOutputType<torch::kBFloat16> {
    using type = cutlass::bfloat16_t;
};

// A[M,K]xB[K,N], A rowmajor B colmajor
torch::Tensor w8a8_int8_linear_bbf16_obf16_per_tensor(torch::Tensor input,  // INT8
                                                    torch::Tensor weight, // INT8
                                                    torch::Tensor bias,   // BF16
                                                    float alpha,          // BF16
                                                    float beta            // BF16
) {
  auto M = input.size(0);
  auto N = weight.size(1); //注意qweight的shape是否转置，这里不要写错了，不然会报illegal memory acess
  auto K = input.size(1);
  // std::cout << "weight.size(0) = " << weight.size(0) << " " << "weight.size(1) = " << weight.size(1) << "\n";
  // 可以根据bias的dtype作为ElementOutput,后面再来尝试跑一下
  // if (bias.dtype() == torch::kFloat16) {
  //     using ElementOutput = ElementOutputType<torch::kFloat16>::type;
  // } 
  // else if (bias.dtype() == torch::kBFloat16) {
  //     using ElementOutput = ElementOutputType<torch::kBFloat16>::type;

  using ElementOutput = cutlass::half_t;//cutlass::bfloat16_t;//
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

#if CUDA_ARCH >= 800
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 8,//128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
#elif CUDA_ARCH >= 750
  using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
      DefaultGemmCfg::InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue>>;
#elif CUDA_ARCH >= 700
  using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
      ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
      DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
      DefaultGemmCfg::InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 1, ElementAccumulator, ElementComputeEpilogue>>;
#else
  #error "Unsupported cuda arch"
#endif

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = input.device();
  // use the broadcasted bias as the output
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});
  cutlass::gemm::GemmCoord problem_size(M, N, K);
  // pytorch tensor不认识cutlass::bfloat16_t，所以这里out(torch bf16)需要先转void再强转cutlass::bfloat16_t
  // 这个问题解决了链接问题ImportError: /usr/local/lib/python3.10/dist-packages/runtime-0.1-py3.10-linux-x86_64.egg/runtime/sq_fp8_kernels.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZNK2at10TensorBase8data_ptrIN7cutlass10bfloat16_tEEEPT_v
  void* out_ptr = out.data_ptr();
  ElementOutput* out_data = static_cast<ElementOutput*>(out_ptr);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input.data_ptr<ElementInputA>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight.data_ptr<ElementInputB>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      out_data, LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha, beta}, 1};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run");
  }

  return out;
}

// torch::Tensor w8a8_int8_linear_bbf16_obf16_per_channel(torch::Tensor input,  // INT8
//                                                         torch::Tensor weight, // INT8
//                                                         torch::Tensor bias,   // BF16
//                                                         torch::Tensor alpha,  // BF16 ,暂不确定是否是torch::Tensor类型
//                                                         torch::Tensor beta    // BF16
// ) {
//   auto M = input.size(0);
//   auto N = weight.size(1);
//   auto K = input.size(1);

//   using ElementOutput = cutlass::bfloat16_t;
//   using ElementAccumulator = int32_t;
//   using ElementComputeEpilogue = float;
//   using ElementInputA = int8_t; // <- data type of elements in input matrix A
//   using ElementInputB = int8_t; // <- data type of elements in input matrix B

//   // The code section below describes matrix layout of input and output
//   // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
//   // for Matrix C
//   using LayoutInputA = cutlass::layout::RowMajor;
//   using LayoutInputB = cutlass::layout::ColumnMajor;
//   using LayoutOutput = cutlass::layout::RowMajor;

// #if CUDA_ARCH >= 800
//   using Gemm = cutlass::gemm::device::Gemm<
//       int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
//       ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
//       cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
//       cutlass::gemm::GemmShape<256, 128, 64>,
//       cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
//       cutlass::epilogue::thread::LinearCombination<
//           ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
//           ElementAccumulator, ElementComputeEpilogue>,
//       cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
// #elif CUDA_ARCH >= 750
//   using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
//       cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
//       ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
//   using Gemm = cutlass::gemm::device::Gemm<
//       int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
//       ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
//       cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
//       DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
//       DefaultGemmCfg::InstructionShape,
//       cutlass::epilogue::thread::LinearCombination<
//           ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
//           ElementAccumulator, ElementComputeEpilogue>>;
// #elif CUDA_ARCH >= 700
//   using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
//       cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
//       ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
//   using Gemm = cutlass::gemm::device::Gemm<
//       int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
//       ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
//       cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
//       DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
//       DefaultGemmCfg::InstructionShape,
//       cutlass::epilogue::thread::LinearCombination<
//           ElementOutput, 1, ElementAccumulator, ElementComputeEpilogue>>;
// #else
//   #error "Unsupported cuda arch"
// #endif

//   auto input_size = cutlass::MatrixCoord(M, K);
//   auto weight_size = cutlass::MatrixCoord(K, N);
//   auto output_size = cutlass::MatrixCoord(M, N);

//   auto device = input.device();
//   // use the broadcasted bias as the output,在epilogue阶段完成D(初始化为bias)=alpha * AB + beta * C(初始化为bias)
//   auto out = bias.to(device).view({1, -1}).repeat({M, 1});

//   cutlass::gemm::GemmCoord problem_size(M, N, K);
  // void* out_ptr = out.data_ptr();
  // ElementOutput* out_data = static_cast<ElementOutput*>(out_ptr);
//   cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
//       input.data_ptr<ElementInputA>(), LayoutInputA::packed(input_size));
//   cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
//       weight.data_ptr<ElementInputB>(), LayoutInputB::packed(weight_size));
//   cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      // out_data, LayoutOutput::packed(output_size));
//       out.data_ptr<ElementOutput>(), LayoutOutput::packed(output_size));

//   typename Gemm::Arguments arguments{
//       problem_size, // <- problem size of matrix multiplication
//       input_ref,    // <- reference to matrix A on device
//       weight_ref,   // <- reference to matrix B on device
//       out_ref,      // <- reference to matrix C on device
//       out_ref,      // <- reference to matrix D on device
//       {alpha.data_ptr<ElementComputeEpilogue>(), beta.data_ptr<ElementComputeEpilogue>()}, 1};
//   Gemm gemm_op;

//   // Using the arguments, query for extra workspace required for matrix
//   // multiplication computation
//   size_t workspace_size = Gemm::get_workspace_size(arguments);

//   // Allocate workspace memory
//   cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

//   // Check the problem size is supported or not
//   cutlass::Status status = gemm_op.can_implement(arguments);
//   if (status != cutlass::Status::kSuccess) {
//     throw std::runtime_error("cutlass cannot implement");
//   }

//   // Initialize CUTLASS kernel with arguments and workspace pointer
//   status = gemm_op.initialize(arguments, workspace.get());
//   if (status != cutlass::Status::kSuccess) {
//     throw std::runtime_error("cutlass cannot initialize");
//   }

//   status = gemm_op();
//   if (status != cutlass::Status::kSuccess) {
//     throw std::runtime_error("cutlass cannot run");
//   }

//   return out;
// }
