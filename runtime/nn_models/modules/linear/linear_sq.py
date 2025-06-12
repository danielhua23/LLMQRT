import torch
import warnings
import torch.nn as nn
from torch.autograd import Function
from runtime.utils.common_utils import get_best_device
from runtime.utils.quantization_utils import (
    quantize_per_tensor_absmax,
    quantize_weight_per_channel_absmax,
    quantize_per_token_absmax,
    fake_quantize_activation_per_tensor_absmax,
    fake_quantize_activation_per_token_absmax,
)

# from runtime.sq_fp8_kernels import w8a8_int8_linear_bbf16_obf16_per_tensor#, w8a8_int8_linear_bbf16_obf16_per_channel

# should check bias and output type fp32 or int32, w/ or w/o scaling
# sq里面的linear都带alpha和beta两个scale，只有W8A8B32O32LinearWithoutScaling不带
# W8A8B8O8LinearReLU用在mlp.fc1，输出的s8，喂到mlp.fc2继续做s8gemm
# W8A8BFP32OFP32Linear用在mlp.fc2和attn.out_proj
# W8A8B8O8Linear用在qkv，output为int8的原因是这个项目里面把qk_bmm设为了s8s8f32
# f32的attnscores经过softmax后再次quantize为了s8，pv_bmm设成了s8s8s8，pv的输出s8最后送到out proj输出fp32
# TODO 调研一下除了opt外的其他模型apply sq是如何做精度转换的？是否和opt一样？

# sq官方试验了llama3，mixtral，mistral，那我们这里就拿llama3/qwen3来实验
# https://github.com/mit-han-lab/smoothquant/blob/main/examples/smoothquant_llama_demo.ipynb
# https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/fake_quant.py
# torch.matmul默认只支持s8s8 in s32 out
# 对于s8s8 in f32/s8 out，需要手动将s32 requantize/dequantize为s8/fp32，如下

# from torch.ao.nn.quantized import Quantize, DeQuantize
# # 定义量化/反量化层
# quantize = Quantize(scale=0.1, zero_point=0, dtype=torch.qint8)
# dequantize = DeQuantize()
# # 假设已有 int8 输入和权重
# input_int8 = torch.randint(-128, 127, (2, 3), dtype=torch.int8)
# weight_int8 = torch.randint(-128, 127, (4, 3), dtype=torch.int8)
# # 手动模拟线性层计算
# output_int32 = torch.matmul(input_int8, weight_int8.t())  # 转为 float 避免溢出
# output_fp32 = dequantize(output_int32)  # 反量化（此处仅为示例，需结合真实量化参数）

user_has_been_warned = False
from .linear_base import LinearBase

@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    qt = t.div(scales).round().clamp(-128, 127)
    qt = qt.to(torch.int8)
    return qt, scales

@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)
    q_max = 2 ** (n_bits - 1) - 1
    scales = scales.values
    scales.clamp_(min=1e-5).div_(q_max)
    qt = t.div(scales).round().clamp(-128, 127)
    qt = qt.to(torch.int8)
    return qt, scales #[m,1]

# 输入的处理：需要把x乘以input scale得到int8 x
# 输出的处理：直接s32=>bf16/fp16=s32 * (input scale * weight scale)
class SqW8A8BBF16OBF16PerTensor(LinearBase):
    # For qkv_proj
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev="cuda:0", dtype=torch.float16, weight_scale=1.0, input_scale=1.0, alpha=1.0, beta=1.0):
        super().__init__()
        self.w_bit = w_bit
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('qweight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False,
                                                                device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros( # 对于sq,这里bf16还是fp16需要根据模型类型而定,opt为fp16,qwen2为bf16
                (self.out_features), dtype=dtype, requires_grad=False,device=dev)) 
        else:
            self.bias = None

        # 这里weight scale的shape都要和quant tool选择的PerTensor或者PerChannel对的上才行
        self.register_buffer('weight_scale', torch.tensor(weight_scale, device=dev)) 
        self.register_buffer('input_scale', torch.tensor(input_scale, device=dev))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.qweight = self.qweight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x): # bf16/fp16 in, static quant to s8
        x_shape = x.shape
        x = x.view(-1, x_shape[-1]).to(self.qweight.device)
        # 因为qkv对各自act的scale不一样，所以sq里面不fuse qkv        
        # dyn per tensor quant act
        qx, input_scale = quantize_activation_per_tensor_absmax(x) # per tensor qwen2胡言乱语 搭配torch.matmul时opt正常说话6.3(在取消了quantize函数中的inplace操作，以及增加了quantized val的clamp)
        input_scale = input_scale.to(self.qweight.device)

        # dyn per token quant act
        # qx, input_scale = quantize_activation_per_token_absmax(x) # per channel act也胡言乱语
        
        # static quant act
        # x.div_(self.input_scale.item()).round_().clamp_(-128, 127) 有下划线的函数为原地操作
        # qx = x.to(torch.int8) # static quant, 这不是一个inplace操作需要换行
        
        # naive impl
        # x_bf16 = qx.to(x.dtype).mul(input_scale)  # dynamic quant scale, qwen2 胡言乱语 opt 正常说话6.3(在取消了quantize函数中的inplace操作，以及增加了quantized val的clamp)
        # x_bf16 = qx.to(x.dtype) * self.input_scale.item()  # static quant qwen2 胡言乱语
        
        # weight_bf16 = self.qweight.to(x.dtype) * self.weight_scale.item() # 适用于bf16/fp16 weight的模型，比如opt
        # weight_bf16 = weight_bf16.t()
        # y = torch.matmul(x_bf16, weight_bf16)  # pure BF16/pure FP16 计算
        # if self.bias is not None:
        #     y += self.bias.unsqueeze(0) #[xx, out feats] + [1, out feats]
        # alpha = self.input_scale.item() * self.weight_scale.item() # static act quant
        alpha = input_scale * self.weight_scale.item() # dyn act quant
        qweight = self.qweight.t() #[20480,5120]:[5120,1] => [5120,20480]:[1,5120]
        if self.bias is not None:
            y = w8a8_int8_linear_bbf16_obf16_per_tensor(qx, qweight, # 确保qweight.shape=[K,N],stride=[1,K]
                                                        self.bias,
                                                        alpha, 1.0)
        else:
            a = torch.zeros((self.out_features), dtype=x.dtype, requires_grad=False, device="cuda:0")
            y = w8a8_int8_linear_bbf16_obf16_per_tensor(qx, qweight, 
                a, alpha, 1.0)
        y = y.view(*x_shape[:-1], -1)
        return y

    @classmethod
    def from_linear(cls, module: torch.nn.Linear, w_bit, group_size, init_only=True, dtype=torch.float16):#, input_scale):
        sq_linear = cls(
            w_bit=w_bit,
            group_size=group_size,
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            dev=module.weight.device,
            dtype=dtype,
        )

        if init_only:
            return sq_linear
    
class SqW8A8BBF16OBF16PerChannel(LinearBase):
    # For qkv_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('qweight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        if bias:
            self.register_buffer('bias', torch.zeros( # 对于sq,这里bf16还是fp16需要根据模型类型而定,opt为fp16,qwen2为bf16
                (self.out_features), dtype=dtype, requires_grad=False,device=dev))
        else:
            self.bias = None
            
        self.register_buffer('weight_scale', (self.out_features), dtype=dtype, requires_grad=False,device=dev)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1]).to(self.weight.device)
        qx, input_scale = quantize_activation_per_token_absmax(x) 
        input_scale = input_scale.to(self.qweight.device)
        
        alpha = input_scale * self.weight_scale.item() # [M,N], only support dyn act per token quant
        qweight = self.qweight.t() 
        if self.bias is not None:
            y = w8a8_int8_linear_bbf16_obf16_per_channel(qx, qweight, # 确保qweight.shape=[K,N],stride=[1,K]
                                                        self.bias,
                                                        alpha, 1.0)
        else:
            a = torch.zeros((self.out_features), dtype=x.dtype, requires_grad=False, device="cuda:0")
            y = w8a8_int8_linear_bbf16_obf16_per_channel(qx, qweight, 
                a, alpha, 1.0)
        y = y.view(*x_shape[:-1], -1)
        return y

    @classmethod
    def from_linear(cls, module: torch.nn.Linear, w_bit, group_size, init_only=True, dtype=torch.float16):
        sq_linear = cls(
            w_bit=w_bit,
            group_size=group_size,
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            dev=module.weight.device,
            dtype=dtype,
        )

        if init_only:
            return sq_linear