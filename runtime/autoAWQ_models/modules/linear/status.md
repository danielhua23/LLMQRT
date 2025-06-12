* sq跑qwen2总是胡言乱语，问题出在对act的quant上面，这里无论使用static per tensor还是dyn per tensor/per channel都是胡言乱语，后续用opt试一下
* sq_gemm.cu写个python ut，像torch int那样，测试一下
6.3更新
* 发现quant的时候quantize函数中round后没有clamp到[-128,127],故修改.以及喂到quantize函数的t一直是inplace操作,改成非inplace
* x_bf16 = qx.to(torch.float16).mul(input_scale)和weight_bf16 = self.qweight.to(torch.float16) * self.weight_scale.item()用torch.matmul加上(in, out)的shape,可以正常说话,但是cutlass int8下无法正常说话,发现qweightshape为[in, out],但是majorness是row major,这里或许不对,故quant工具生成[out, in] row major的qweight,runtime收到该weight,然后在forward阶段trans成[in, out] col major的qweight
* 基于上面发现改进,尝试在runtime/linear_sq.py来现场trans,使得qweightshape=[out, in] row major=>[in, out] col major后,终于跑通!!

6.11更新fp8
* runtime
    * kernel层面已经没问题，UTpass
    * fuse utils里面，dyn下可以fuse qkv，但是我偷个懒没有fuse，留给他们做作业
    * torch.nn.Parameter不支持fp8类型，所以声明bf16类型，量化weight后先不to为fp8，还是bf16保存，在fwd的时候再来to
    * weight scale如果赋值对象是torch.tensor，那么它的shape为[]，但是我们在init函数里面声明的shape是[1]，所以在quant的时候需要在from linear处对weight scale unsqueeze
    * weight scale和input scale必须都<1，不然肯定出错了

* 跑runtime/example下的代码时都需要加CUDA_VISIBLE_DEVICES=0让所有数据都在一个device，不然load_checkpoint_dispatch会把不同的数据分到两个设备