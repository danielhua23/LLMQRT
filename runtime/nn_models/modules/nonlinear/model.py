import torch
import torch.nn as nn
from typing import List
from runtime.utils import fused_utils
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    MoeModelOutputWithPast,
)
from .block import LlamaLikeBlock

class LlamaLikeModel(nn.Module):
    """
    LlamaLikeModel is intended to be reused across models that have
    an architecture that closely resembles Llama, e.g. Mistral and Aquila.
    """

    def __init__(self, vocab_size, blocks, embedding, norm):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.blocks: List[LlamaLikeBlock] = nn.ModuleList(blocks)
        self.norm = norm
        self.last_forward_num_tokens = 0

    @property
    def embed_tokens(self):
        return self.embedding

    @property
    def layers(self):
        return self.blocks

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        *args,
        **kwargs,
    ):
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids, self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape
        # 更新当前fwd的cache start pos
        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.embedding(input_ids) # cuda0
        
        for id, layer in enumerate(self.blocks):
            # 虽然layer分发到了不同的device，但是下面一行把hiddenstates to到了其对应的device
            h = h.to(layer.device) # bug0: 第0个iter可以通过，但是第1个iter，h变成了tensor(..., device='meta', size=(1, 61, 5120), dtype=torch.float16)，报错NotImplementedError: Cannot copy out of meta tensor; no data!
            h = layer(h) # bug1: 第0个iter的输出不是nan，但是第1个iter，全是nan, 造成的地方在于attn
        # import pdb
        # pdb.set_trace()
        h = self.norm(h)

        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=None,
            hidden_states=(),
            attentions=(),
        )

