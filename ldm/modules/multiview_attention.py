from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any
from .utils import *

from ldm.modules.diffusionmodules.util import checkpoint
import cv2

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
    
# try:
#     from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
#     FLASHATTEN_IS_AVAILBLE = True
# except:
FLASHATTEN_IS_AVAILBLE = False

# CrossAttn precision handling
import os

#Modified: MultiViewBasicTransformerBlock, MultiViewSpatialTransformer

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            with torch.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

def save_attn_vis(attn_score, self_or_cross='self', scale=1):
    # 判断路径和文件
    from glob import glob
    target = "2View"
    exp_name = "test"
    fs = glob(f"{exp_name}_attention_outputs/{target}/*")
    fs.sort()
    if len(fs) == 0:
        os.makedirs(f"{exp_name}_attention_outputs/{target}/000000", exist_ok=True)
        fs = glob(f"{exp_name}_attention_outputs/{target}/*")
        fs.sort()

    last_fs = glob(fs[-1] + '/*')
    if len(last_fs) == 1600:
        os.makedirs(f"{exp_name}_attention_outputs/{target}/{str(int(fs[-1].split('/')[-1]) + 1).zfill(6)}", exist_ok=True)
        fs = glob(f"{exp_name}_attention_outputs/{target}/*")
        fs.sort()

    file_idx = fs[-1].split('/')[-1]
    # load mask
    mask = f"/home/wmlce/ControlNet-main/data/megadepth_0.4_0.7/match_test_image_pairs/000112/mask.png"
    mask = cv2.imread(mask) / 255
    mask = mask[:, :, 0]
    # 一共16层,32个attention,50steps, 1600个总共
    idx = 0

    
    v = 3
    if self_or_cross == 'self':
        # self attn_score [B VHW VHW] B=10
        h = int(math.sqrt(attn_score.shape[1] / v))
        w = int(attn_score.shape[1] / v // h)
    else:
        # cross attn_score [BV HW length] B=10
        v -= 1
        h = int(math.sqrt(attn_score.shape[1] // 2))
        w = int(attn_score.shape[1] // h)

    # 第一个HW是最终赋值的位置，第二个HW是对各个点的权重
    mask = cv2.resize(mask, [w, h], interpolation=cv2.INTER_AREA)
    mask[mask > 0] = 1
    mask = torch.tensor(mask, device=attn_score.device, dtype=attn_score.dtype)
    if self_or_cross == 'self':
        # self attn_score [B VHW VHW] B=10
        attn_score = attn_score.mean(dim=0)
        attn_score = attn_score.reshape(v, h, w, v, h*w)[0, :, :, :, :] * scale * torch.tensor([1, 1, 1.05]).cuda().reshape(1, 1, -1, 1)
        attn_score = attn_score.reshape(h, w, v*h*w)
        attn_score = torch.softmax(attn_score, dim=-1).reshape(h, w, -1)
        attn_score = attn_score * mask[:, :, None]
#         start = int(150 / 512 * w)
#         end = int(250 / 512 * w)
        attn_score = attn_score.reshape(h, w, v, h, w)[:, :, 1:, :, :]
        attn_score =attn_score.reshape(-1, v - 1, h, w)
        attn_score = attn_score.sum(dim=0)
        assert len(attn_score.shape) == 3
        attn_score = attn_score / attn_score.max() * 255
        attn_score = attn_score.permute(1, 0, 2).reshape(h, -1) # [h, w]
        attn_score = torch.clamp(attn_score, min=0, max=255)

#         attn_score = attn_score.reshape(v, h*w, v, h, w)[:, :, 1:, :, :]
#         attn_score = attn_score.reshape(v, h, w, (v-1)*h*w)
#         attn_score = torch.softmax(attn_score, dim=-1)
#         attn_score[0, :, :, :] = attn_score[0, :, :, :] * mask[:, :, None]
# #         start = int(150 / 512 * w)
# #         end = int(250 / 512 * w)
# #         attn_score = attn_score[:, start:end, :]
#         attn_score =attn_score.reshape(-1, v - 1, h, w).mean(dim=0)
#         assert len(attn_score.shape) == 3
# #         attn_score = attn_score / attn_score.max() * 255
#         attn_score = attn_score.permute(1, 0, 2).reshape(h, -1) # [h, w]
# #         attn_score = torch.clamp(attn_score, min=0, max=255)
        
        # attn_score = attn_score.reshape(h, w, h * w)
        # mask = mask.reshape(h, w, 1)
        # attn_score = attn_score * mask
        # attn_score = attn_score.reshape(h * w, h, w)[:, :, :w // 2].mean(dim=0)

    else:
        # cross attn_score [BV HW length] B=10
        attn_score = attn_score.reshape(10, v, h, w, -1).mean(dim=0)
        attn_score = attn_score[0, :, :, :]
        attn_score = attn_score.sum(dim=-1) / attn_score.max() * 255
        attn_score = torch.clamp(attn_score, min=0, max=255)
    while os.path.exists(fs[-1] + f'/{h}x{w}_{self_or_cross}_{str(idx).zfill(3)}.npy'):
        idx += 1
#     print('save to', fs[-1] + f'/{h}x{w}_{self_or_cross}_{str(idx).zfill(3)}.npy')
    np.save(fs[-1] + f'/{h}x{w}_{self_or_cross}_{str(idx).zfill(3)}.npy',
            attn_score.cpu().numpy())

class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None, return_attn=False):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape

        q, k, v = map(
            lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
            (q, k, v),
        )
        
#         with torch.no_grad():
#             att_score = q @ k.transpose(-1, -2)  # [B,L,L]

#         if context.shape == x.shape:
#             self_or_cross = "self"
#             # att_score = torch.softmax(att_score, dim=2)
#         else:
#             self_or_cross = "cross"
#             # att_score = torch.softmax(att_score, dim=1)

#         save_attn_vis(att_score, self_or_cross, scale=self.dim_head**-0.5)

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
        )

        # if return_attn:
        #     return self.to_out(out), att_score
        # else:
        return self.to_out(out)


class FlashAttention(MemoryEfficientCrossAttention):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__(query_dim, context_dim, heads, dim_head, dropout)

    def forward(self, x, context=None, mask=None, return_attn=False):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape

        q, k, v = map(
            lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)

        if exists(mask):
            raise NotImplementedError
        out = out.reshape(b, out.shape[1], self.heads * self.dim_head)

        # if return_attn:
        #     return self.to_out(out), att_score
        # else:
        return self.to_out(out)
    

class MultiViewBasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,
        "softmax-falshatten": FlashAttention
    }

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, **kwargs):
        super().__init__()
        if FLASHATTEN_IS_AVAILBLE:
            print('use flash attention')
            attn_mode = "softmax-falshatten"
        elif XFORMERS_IS_AVAILBLE:
            attn_mode = "softmax-xformers"
        else:
            attn_mode = "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

        self.view_num = kwargs.get("view_num", 4)
        self.concat_target = kwargs.get('concat_target', False)
        self.no_rearrange_selfattn = kwargs.get('no_rearrange_selfattn', False)

    def forward(self, x, context=None, return_attn=False):
        return checkpoint(self._forward, (x, context, return_attn), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, return_attn=False):
        if self.disable_self_attn:
            raise ValueError("The model should not disable self attention as designed.")
        
        # rearrange to let the images affect each other during the self-attn
        if self.concat_target:
            if self.no_rearrange_selfattn:
                x = rearrange(x, '(b v) hw c -> b (v hw) c', v=self.view_num-1).contiguous()
            else:
                img_size = int(math.sqrt(x.shape[1] / 2))
                # x_normal is [[view, target], [view, target], [view, target]]
                x_normal = rearrange(x, '(b v) (h w) c -> b v h w c', v=self.view_num-1, h=img_size)

                # x is [target, *views]
                x = torch.cat((x_normal[:, 0:1, :, img_size:, :], x_normal[:, :, :, 0:img_size, :]), dim=1)
                x = rearrange(x, 'b v h w c -> b (v h w) c').contiguous() 
        else:
            x = rearrange(x, '(b v) hw c -> b (v hw) c', v=self.view_num).contiguous()
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        
        # rearrange back to normally conv and cross-attn
        if self.concat_target:
            if self.no_rearrange_selfattn:
                x = rearrange(x, '(b v) hw c -> b (v hw) c', v=self.view_num-1).contiguous()
            else:
                x = rearrange(x, 'b (v h w) c -> b v h w c', v=self.view_num, h=img_size)
                x_new = torch.zeros_like(x_normal)
                x_new[:, :, :, img_size:, :] = x[:, 0:1, :, :, :]
                x_new[:, :, :, 0:img_size, :] = x[:, 1:, :, :, : ]
                x = rearrange(x_new, 'b v h w c -> (b v) (h w) c').contiguous()
        else:
            x = rearrange(x, 'b (v hw) c -> (b v) hw c', v=self.view_num).contiguous()
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        if return_attn:
            return x, att_score
        else:
            return x


class BasicCrossTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True, num_patches=None):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                             context_dim=context_dim)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        if num_patches is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
            trunc_normal_(self.pos_embed, std=.02)
        else:
            self.pos_embed = None

    def forward(self, x, context=None, return_attn=False):
        if return_attn:
            return self._forward(x, context, return_attn)
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, return_attn=False):
        context = rearrange(context, 'b c h w -> b (h w) c').contiguous()
        if self.pos_embed is not None:
            x = x + self.pos_embed
            context = context + self.pos_embed
        if return_attn:
            x_, att_score = self.attn(self.norm1(x), context=context, return_attn=return_attn)
        else:
            x_ = self.attn(self.norm1(x), context=context, return_attn=return_attn)
        x = x_ + x
        x = self.ff(self.norm2(x)) + x
        if return_attn:
            return x, att_score
        else:
            return x


class MultiViewSpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, one_attn=False, num_patches=None, **kwargs):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.one_attn = one_attn
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if one_attn:
            self.transformer_blocks = nn.ModuleList(
                [BasicCrossTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                            checkpoint=use_checkpoint, num_patches=num_patches)
                 for d in range(depth)]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [MultiViewBasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                       disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, **kwargs)
                 for d in range(depth)]
            )
        if not one_attn:
            if not use_linear:
                self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                      in_channels,
                                                      kernel_size=1,
                                                      stride=1,
                                                      padding=0))
            else:
                self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        else:
            if not use_linear:
                self.proj_out = nn.Conv2d(inner_dim,
                                          in_channels,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0)
            else:
                self.proj_out = nn.Linear(in_channels, inner_dim)
        self.use_linear = use_linear

    def forward(self, x, context=None, **kwargs):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if self.one_attn and kwargs.get('return_attn', False):
                x, att_score = block(x, context=context[i], return_attn=True)
            elif kwargs.get('return_attn', False):
                x, att_score = block(x, context=context[i], return_attn=True)
            else:
                x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)

        if self.one_attn and kwargs.get('return_attn', False):
            return x + x_in, att_score
        else:
            return x + x_in
