# 用于定义csrc cuda kernel的torch class，包含init和fwd函数，然后这个class替换到models/qwen3.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
# from vllm.model_executor.parallel_utils.communication_op import (
#     tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce)
# from vllm.model_executor.parallel_utils.parallel_state import (
#     get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
# from vllm.model_executor.parallel_utils.utils import (
#     divide, split_tensor_along_last_dim)
from .quantization.utils.util import set_weight_attrs

logger = init_logger(__name__)
class LinearMethodBase(ABC):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def create_weights(self, 
                       layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype,
                       **extra_weight_attrs) -> Dict[str, Any]:
        """Create weights for a linear layer."""
        raise NotImplementedError

    @abstractmethod
    def apply_weights(self,
                      # weights: Dict[str, torch.Tensor],
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply the weights to the input tensor."""
        raise NotImplementedError
    
class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization.

    Args:
        separate_bias_add: If true, add bias separately after matrix
                           multiplication.
    """

    def __init__(self, separate_bias_add: bool = False):
        self.separate_bias_add = separate_bias_add

    def create_weights(self, 
                       layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype,
                       **extra_weight_attrs) -> Dict[str, Any]:
        weight = Parameter(torch.empty(output_size_per_partition,
                                       input_size_per_partition,
                                       dtype=params_dtype),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        # return {"weight": weight}
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        
    def apply_weights(self,
                      layer: torch.nn.Module,
                      # weights: Dict[str, torch.Tensor],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = layer.weight
        if self.separate_bias_add:
            if bias is not None:
                return F.linear(x, weight) + bias
            return F.linear(x, weight)
        return F.linear(x, weight, bias)
    
class QuantizedLinear(torch.nn.Module):
    """quantized Linear layer with column parallelism.

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        linear_method: (Maybe quantized) linear method.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: : Optional[QuantizationConfig] = None,
        # linear_method: Optional[LinearMethodBase] = None,
        prefix: str = "",
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        # tp_size = get_tensor_model_parallel_world_size()
        # self.output_size_per_partition = divide(output_size, tp_size)
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        if quant_config is None:
            self.linear_method = UnquantizedLinearMethod()
        else: # 经由这个方法dispatch给各个量化方法
            self.linear_method = quant_config.get_quant_method(self,
                                                              prefix=prefix)
        self.linear_weights = self.linear_method.create_weights(
            self.input_size, self.output_size, self.params_dtype)
        for name, weight in self.linear_weights.items():
            if isinstance(weight, torch.Tensor):
                self.register_parameter(name, weight)
                set_weight_attrs(weight, {"weight_loader": self.weight_loader}) # create weights只是创建buffer，这里的写法可能是配合autoWeightLoader调用weight_loader把int8/int4/fp8 load到这块buffer
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition,
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # tp_rank = get_tensor_model_parallel_rank()
        output_dim = getattr(param, "output_dim", None)
        param_data = param.data
        # if output_dim is not None:
        #     shard_size = param_data.shape[output_dim]
        #     start_idx = tp_rank * shard_size
        #     loaded_weight = loaded_weight.narrow(output_dim, start_idx,
        #                                          shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        output = self.linear_method.apply_weights( # apply weights是vllm 0.4.0的api
            self.linear_weights, input_, bias)
        # if self.gather_output:
        #     # All-gather across the partitions.
        #     output = tensor_model_parallel_all_gather(output_parallel)
        # else:
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias