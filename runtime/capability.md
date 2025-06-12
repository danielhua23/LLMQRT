* fp8:
    * dynamic quant: per tensor/per token
    * static quant: per tensor only
    * kv cache scale
    * scaled_mm支持row wise scale和tensor wise scale gemm肯定是可以的
    * cutlass fp8 ada row wise/ tensorwise gemm with bias
* sq
    * bias也在kernel的epilogue支持
    * SqW8A8BBF16OBF16PerTensor per tensor
    * SqW8A8BBF16OBF16PerChannel ptpc
    * sq跑qwen2总是胡言乱语，问题出在对act的quant上面，这里无论使用static per tensor还是dyn per tensor/per channel都是胡言乱语，后续用opt试一下。quant sq qwen2应该是没有问题的，因为保持act为bf16就是正常输出

* awq
    * weight的每列per group=128 quant
    * 不存在什么static/dyn quant，因为这个是WOQ，act始终不会quant
    * 目前runtime已跑通，说明base.py的入口没问题
    * base.py里面的load checkpoint and dispatch会自动把数据dispatch到它认为当前合理的device上面去，我们还不可控制，所以runtime需要设置环境变量CUDA_VISIBLE_DEVICES=0以使得所有数据都在一个device
    * 待解释的问题：
        * 为什么存int4weight时按照02461357的顺序pack进int32+取时要按照04152637，为什么时这种组合？
        * kernel里面的三次interleave的结果是否是我在kernel那里注释的那样，待验证
