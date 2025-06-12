import transformers
import torch
import re
import gc
from typing import Tuple, Optional
from .linear_base import LinearBase
from runtime.sq_fp8_kernels import cutlass_f8f8bf16_tensorwise_sm89
from runtime.sq_fp8_kernels import cutlass_f8f8bf16_rowwise_sm89

# 目前fp8也可以试mixtral-8x7b,llama3,qwen3
def per_tensor_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize a act tensor using dynamic per-tensor quant.
    Args:
        tensor: The input tensor.
    Return:
        qtensor: quantized act and their scales
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    if tensor.numel() == 0:
        # Deal with empty tensors (triggered by empty MoE experts)
        min_val, max_val = (
            torch.tensor(-16.0, dtype=tensor.dtype),
            torch.tensor(16.0, dtype=tensor.dtype),
        )
    else:
        min_val, max_val = tensor.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = amax.clamp(min=1e-12) / finfo.max
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float()
    return qweight, scale

def static_per_tensor_quantize(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fn)

# DONE, runtime里面这里是per token，针对dyn阶段运行时量化input，quant这里是per channel，针对static quant weight
def per_token_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize a tensor using dynamic per-tensor quant.
    Args:
        tensor: The input tensor.
    Return:
        qtensor: quantized act and their scales
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    # Calculate the scale as dtype max divided by absmax.
    # Since .abs() creates a new tensor, we use aminmax to get
    # the min and max first and then calculate the absmax.
    if tensor.numel() == 0:
        # Deal with empty tensors (triggered by empty MoE experts)
        print("[warning] You are experiencing empty MoE experts, tensor numbers = 0")
        qweight = torch.empty_like(tensor, dtype=torch.float8_e4m3fn)
        scales = torch.ones((*tensor.shape[:-1], 1), dtype=torch.float32)
        return qweight, scales
    amax = tensor.abs().amax(dim=-1, keepdim=True)
    scale = amax.clamp(min=1e-12) / finfo.max
    # scale = finfo.max / amax.clamp(min=1e-12)
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float()#.reciprocal() # 求倒
    return qweight, scale


def fp8_gemm(A, A_scale, B, B_scale, bias, out_dtype):
    if A.numel() == 0:
        # Deal with empty tensors (triggeted by empty MoE experts)
        return torch.empty(size=(0, B.shape[0]), dtype=out_dtype, device=A.device)

    # TODO: Disable native fp8 gemm for now, always just dequantize
    # native_fp8_support = (
    #     torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
    # )
    native_fp8_support = False
    if native_fp8_support:
        need_reshape = A.dim() == 3
        if need_reshape:
            batch_size = A.shape[0]
            A_input = A.reshape(-1, A.shape[-1])
        else:
            batch_size = None
            A_input = A
        output, _ = torch._scaled_mm(
            A_input,
            B.t(),
            out_dtype=out_dtype,
            scale_a=A_scale,
            scale_b=B_scale,
            bias=bias,
        )
        if need_reshape:
            output = output.reshape(
                batch_size, output.shape[0] // batch_size, output.shape[1]
            )
    else:
        output = torch.nn.functional.linear(
            A.to(out_dtype) * A_scale,
            B.to(out_dtype) * B_scale.to(out_dtype),
            bias=bias,
        )
    return output

def replace_module(model, name, new_module: torch.nn.Module):
    if "." in name:
        parent_name = name.rsplit(".", 1)[0] # model.layers.0.self_attn
        child_name = name[len(parent_name) + 1 :] # q_proj
        parent = model.get_submodule(parent_name)
        # Qwen2SdpaAttention(
        # (q_proj): Linear(in_features=5120, out_features=5120, bias=True)
        # (k_proj): Linear(in_features=5120, out_features=1024, bias=True)
        # (v_proj): Linear(in_features=5120, out_features=1024, bias=True)
        # (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
        # (rotary_emb): Qwen2RotaryEmbedding()
        # )
    else:
        parent_name = ""
        parent = model
        child_name = name
    # Qwen2SdpaAttention(
    # (q_proj): FP8DynamicLinear()
    # (k_proj): Linear(in_features=5120, out_features=1024, bias=True)
    # (v_proj): Linear(in_features=5120, out_features=1024, bias=True)
    # (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
    # (rotary_emb): Qwen2RotaryEmbedding()
    # )
    setattr(parent, child_name, new_module) # 把new module替换掉child name，成为parent的新child

dtype_to_str = {
    torch.bfloat16:"bfloat16",
    torch.float16:"float16"
}    
# Class responsible for quantizing weights
class FP8DynamicLinear(LinearBase):
    def __init__(
        self,
        in_features,
        out_features,
        # weight: torch.Tensor,
        # weight_scale: torch.Tensor,
        bias,
        dev="cuda:0",
        dtype=torch.bfloat16,
        qdtype=torch.float8_e4m3fn, # TODO: 需要在quant tool中把e4m3或e5m2加入到quant config
        per_tensor=True,
    ):
        super().__init__()
        self.dtype = dtype
        self.qdtype = qdtype
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn((self.out_features, self.in_features) ,dtype=dtype, device=dev, requires_grad=False))
        self.per_tensor = per_tensor
        if self.per_tensor:
            self.weight_scale = torch.nn.Parameter(torch.randn(
                (1) ,dtype=torch.float32, device=dev, requires_grad=False))
        else:# need to transpose to [1, N]
            self.weight_scale = torch.nn.Parameter(torch.randn(
                (self.out_features,1) ,dtype=torch.float32, device=dev, requires_grad=False))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros( # 对于sq,这里bf16还是fp16需要根据模型类型而定,opt为fp16,qwen2为bf16
                (self.out_features), dtype=dtype, requires_grad=False, device=dev)) 
        else:
            self.bias = None
        
        
    @classmethod
    def from_linear(cls, module: torch.nn.Linear, w_bit=8, group_size=0, init_only=True, dtype=torch.bfloat16, per_tensor=True):
        # fixme: 记得fp8 quant的时候group size改成0
        assert group_size == 128, "not support group wise fp8 quant yet! pls set group_size = 0"
        fp8_dynamic_linear = cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            dev=module.weight.device,
            dtype=dtype,
            per_tensor=per_tensor
        )      
        # import pdb;pdb.set_trace()
        if init_only:
            return fp8_dynamic_linear  
        
    def forward(self, x):
        beta = 1.0
        qweight = self.weight.to(self.qdtype).t()
        x_shape = x.shape
        # scale is computed in runtime, so naming dyn
        if self.per_tensor:  
            qinput, x_scale = per_tensor_quantize(x)
            alpha = self.weight_scale * x_scale # scalar, weight scale和input scale必须都<1，不然肯定出错了
             # 在这里qinput和qweigt都是fp8，qinput.shape=[1,61,5120],
            qinput = qinput.view(-1, x_shape[-1]) # [1,61,5120]=>[61,5120]
            output = cutlass_f8f8bf16_tensorwise_sm89(
                qinput,
                qweight,
                self.bias,#c++层面配合了std::optional就不需要后面的了 if self.bias is not None else torch.zeros((self.out_features), dtype=x.dtype, requires_grad=False, device="cuda:0"),
                alpha,
                beta,
                dtype_to_str.get(self.dtype)
            )
        else:
            qinput, x_scale = per_token_quantize(x)
            qinput = qinput.view(-1, x_shape[-1])# [1,61,5120]=>[61,5120]
            x_scale = x_scale.view(-1, 1) # [1,61,1]=>[61,1]
            # 下面这行会报错：scale_b.size(0) == 1的check不通过，所以最好不要把类的成员变量这样写在等号左右两边
            # self.weight_scale.data = self.weight_scale.to(torch.float32).t() #[N,1]=>[1,N]
            weight_scale = self.weight_scale.to(torch.float32).t()
            output = cutlass_f8f8bf16_rowwise_sm89(
                qinput,
                qweight,
                self.bias,
                x_scale,
                weight_scale, #self.weight_scale.data,
                True # use fast accum
            )
        output = output.view(*x_shape[:-1], -1)
        return output

# static quant only support per tensor
class FP8StaticLinear(LinearBase):
    def __init__(
        self,
        in_features,
        out_features,
        bias,
        dev="cuda:0",
        dtype=torch.bfloat16,
        qdtype=torch.float8_e4m3fn, # TODO: 需要在quant tool中把e4m3或e5m2加入到quant config
        per_tensor=True, # only enable static quant when per tensor
        quantize_output=False
        # weight: torch.nn.Parameter,
        # weight_scale: torch.nn.Parameter,
        # bias: torch.nn.Parameter,
        # input_scale: torch.nn.Parameter,
        # output_scale: Optional[torch.nn.Parameter] = None,
    ):
        super().__init__()
        self.qdtype = qdtype
        self.dtype = dtype
        self.per_tensor = per_tensor
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn((self.out_features, self.in_features) ,dtype=dtype, device=dev, requires_grad=False))
        if True: # fp8_static_linear.per_tensor always true
            self.weight_scale = torch.nn.Parameter(torch.randn(
                (1) ,dtype=torch.float32, device=dev, requires_grad=False))
        else:
            self.weight_scale = torch.nn.Parameter(torch.randn(
                (self.out_features, 1) ,dtype=torch.float32, device=dev, requires_grad=False))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros( # 对于sq,这里bf16还是fp16需要根据模型类型而定,opt为fp16,qwen2为bf16
                (self.out_features), dtype=dtype, requires_grad=False, device=dev)) 
        else:
            self.bias = None
        # static quant only support per tensor act 
        self.input_scale = torch.nn.Parameter(torch.randn((1) ,dtype=torch.float32, device=dev, requires_grad=False))
        self.quantize_output = quantize_output
        self.output_scale = torch.nn.Parameter(torch.randn((1) ,dtype=torch.float32, device=dev, requires_grad=False))
        
    @classmethod
    def from_linear(cls, module: torch.nn.Linear, w_bit=8, group_size=0, init_only=True, dtype=torch.bfloat16, per_tensor=True, quantize_output=False):#, per_tensor=True, input_scale=None, output_scale=None, group_size=0, zeros=None):
        # fixme: 记得fp8 quant的时候group size改成0
        assert group_size == 128, "not support group wise fp8 quant yet! pls set group_size = 0"
        fp8_static_linear = cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            dev=module.weight.device,
            dtype=dtype,
            per_tensor=True,
            quantize_output=quantize_output
        )      
        if init_only:
            return fp8_static_linear  
     
    def forward(self, x):
        beta = 1.0
        qweight = self.weight.to(self.qdtype).t()
        x_shape = x.shape
        # scale is known in advance, so naming static
        if self.per_tensor:
            qinput = static_per_tensor_quantize(x, self.input_scale) # fp8
            alpha = self.weight_scale * self.input_scale # scalar * scalar
            qinput = qinput.view(-1, x_shape[-1]) # [1,61,5120]=>[61,5120]
            output = cutlass_f8f8bf16_tensorwise_sm89(
                qinput,
                qweight,
                self.bias,
                alpha,
                beta,
                dtype_to_str.get(self.dtype)
            )
        # static act只支持per tensor，下面这个不存在
        else:
            print("[error!!] static activation quant only support per tensor!")
        output = output.view(*x_shape[:-1], -1)
        if self.quantize_output:
            qoutput = static_per_tensor_quantize(output, self.output_scale) # fp16/fp32 output / outputscale => fp8
            output = qoutput.to(output.dtype) * self.output_scale # fp8 * outputscale => fp16/bf16

        return output # bf16/fp16
