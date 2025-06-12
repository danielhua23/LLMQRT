from .linear_base import LinearBase
from .linear_awq import WQLinear_GEMM
from .linear_fp8 import FP8StaticLinear, FP8DynamicLinear
from .linear_sq import SqW8A8BBF16OBF16PerTensor
method_to_linear: dict[str, type[LinearBase]] = {
    "awq": WQLinear_GEMM,
    "sq": SqW8A8BBF16OBF16PerTensor, # 待改进，把per channel引入
    "fp8_static_quant": FP8StaticLinear, # per tensor only
    "fp8_dynamic_quant": FP8DynamicLinear, # per tensor default, per token available too
}

def get_concrete_linear_module(quant_method):
    return method_to_linear[quant_method]


    