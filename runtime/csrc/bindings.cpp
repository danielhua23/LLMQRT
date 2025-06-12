// 把fp8 gemm和sq gemm pybind绑定一下

// #include "fp8/fp8_gemm.h"
// #include "smoothquant/sq_gemm.h"
#include "fp8/sm89_fp8_gemm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("w8a8_int8_linear_bbf16_obf16_per_channel", &w8a8_int8_linear_bbf16_obf16_per_channel,
//         "int8 linear per channel");
  // m.def("w8a8_int8_linear_bbf16_obf16_per_tensor", &w8a8_int8_linear_bbf16_obf16_per_tensor,
  //       "int8 linear per tensor");
  m.def("cutlass_f8f8bf16_tensorwise_sm89", &f8f8bf16_tensorwise, "fp8 tensorwise linear on sm89");
  m.def("cutlass_f8f8bf16_rowwise_sm89", &f8f8bf16_rowwise, "fp8 rowwise linear on sm89");
}
