## runtime支持模型情况
* awq: qwen2,qwen3,llama3, deepseek-v3-1B(随机weight，所以输出不对), qwen3-moe因为量化太久，还没在runtime试过
* sq: only opt precision is OK, both llama3 and qwen2 are bad
* fp8 dyn per tensor: llama3, qwen2, qwen3, qwen3-30b-a3b(无论开启thinking与否都很慢)
* fp8 static per tensor: llama3,qwen2,qwen3, qwen3-30b-a3b(无论开启thinking与否都很慢)
* only awq support fuse qkv now, fp8 dyn can support too but can be a homework