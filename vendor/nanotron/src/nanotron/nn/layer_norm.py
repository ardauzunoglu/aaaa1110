import torch
from torch import nn

try:
    from flash_attn.ops.triton.layer_norm import layer_norm_fn
except ImportError:
    layer_norm_fn = None


class TritonLayerNorm(nn.LayerNorm):
    def forward(
        self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False
    ):

        if layer_norm_fn is None:
            if residual is not None or dropout_p != 0.0 or prenorm or residual_in_fp32 or return_dropout_mask:
                raise RuntimeError("Fused LayerNorm options require the flash-attn package")
            return torch.nn.functional.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

        return layer_norm_fn(
            input,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=False,
            return_dropout_mask=return_dropout_mask,
        )


# This is equivalent to LLaMA RMSNorm
# https://github.com/huggingface/transformers/blob/28952248b19db29ca25ccf34a5eec413376494a9/src/transformers/models/llama/modeling_llama.py#L112
class TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(
        self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False
    ):

        if layer_norm_fn is None:
            if residual is not None or dropout_p != 0.0 or prenorm or residual_in_fp32 or return_dropout_mask:
                raise RuntimeError("Fused RMSNorm options require the flash-attn package")
            input_dtype = input.dtype
            normalized = input.float()
            normalized = normalized * torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + self.eps)
            return self.weight * normalized.to(input_dtype)

        return layer_norm_fn(
            input,
            self.weight,
            None,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True,
            return_dropout_mask=return_dropout_mask,
        )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, input):
        input_dtype = input.dtype
        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.eps)
        return self.weight * input.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
