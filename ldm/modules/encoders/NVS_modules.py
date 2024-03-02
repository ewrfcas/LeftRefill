from typing import Union, List

import open_clip
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def init_special_embeddings(tokenizer, special_tokens, model, init_text, tokenwise_init):
    # special initialization
    sp_emb_weights = torch.zeros((len(special_tokens), model.token_embedding.embedding_dim), dtype=torch.float32)
    if tokenwise_init:  # init the embedding with splited sentence tokens
        origin_tokens = tokenizer.encode(init_text[0])[:len(special_tokens)]
        for i, tok_idx in enumerate(origin_tokens):
            sp_emb_weights[i] = model.token_embedding.weight[tok_idx, :].detach()
        # 长度不够，平均来凑
        for i in range(len(origin_tokens), len(special_tokens)):
            init_feats = []
            mean_tokens = tokenizer.encode(init_text[i])
            for tok_idx in mean_tokens:
                init_feats.append(model.token_embedding.weight[tok_idx:tok_idx + 1, :].detach())
            init_feats = torch.mean(torch.stack(init_feats, dim=0), dim=0)
            sp_emb_weights[i] = init_feats
    else:
        for i, sp_token in enumerate(special_tokens):
            # we first get token index of the original tokens
            init_feats = []
            if init_text is None:
                origin_tokens = tokenizer.encode(sp_token.strip('<').strip('>').replace('-', ' '))
            else:
                origin_tokens = tokenizer.encode(init_text[i])
            for tok_idx in origin_tokens:
                init_feats.append(model.token_embedding.weight[tok_idx:tok_idx + 1, :].detach())
            init_feats = torch.mean(torch.stack(init_feats, dim=0), dim=0)
            sp_emb_weights[i] = init_feats

    return sp_emb_weights


def tokenize(tokenizer, texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<start_of_text>"]
    eot_token = tokenizer.encoder["<end_of_text>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]  # Truncate
            tokens[-1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class RelPosModel(nn.Module):
    def __init__(self, input_ch=3, out_ch=1024, pos_strengthen=False):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(input_ch, out_ch // 2), nn.SiLU(), nn.Linear(out_ch // 2, out_ch))
        self.pos_strengthen = pos_strengthen
        if pos_strengthen:
            self.mlp2 = nn.Sequential(nn.SiLU(), nn.Linear(out_ch, out_ch))

    def forward(self, x):
        x1 = self.mlp1(x)
        if self.pos_strengthen:
            x2 = self.mlp2(x1)
            return x1, x2
        else:
            return x1


class NVSCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        "last",
        "penultimate"
    ]

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77, freeze=True, layer="last",
                 special_tokens=['<left>', '<right>'], init_text=None, tokenwise_init=False, deep_prompt=False, cross_attn_layers=16,
                 view_prompt=False, view_num=None, view_token_len=1, pos_strengthen=False, cfg_rate=0.0, **kwargs):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.deep_prompt = deep_prompt
        self.cross_attn_layers = cross_attn_layers

        if special_tokens[0].startswith('repeat_'):  # if start with "repeat-{n}", we need to adjust each sp_token with the index
            n = int(special_tokens[0].split('_')[1])
            special_tokens = list(special_tokens)
            init_text = list(init_text)
            special_tokens = special_tokens * n
            init_text = init_text * n
            for i in range(n):
                special_tokens[i] = special_tokens[i].split('_')[-1].replace('>', f'{i}>')
        special_tokens = list(special_tokens)

        if deep_prompt:  # further repeat prompt for different model layers
            deep_special_tokens = []
            for layer_i in range(cross_attn_layers):
                special_tokens_ = [t.replace('>', f'-layer{layer_i}>') for t in special_tokens]
                deep_special_tokens.extend(special_tokens_)
            special_tokens = deep_special_tokens
            init_text = init_text * cross_attn_layers

        if view_prompt:
            for j in range(view_num):
                for l in range(view_token_len):
                    special_tokens.append(f"<view_direct-{j}-{l}>")
                    init_text.append("overhead view, front view, side view, back view")

        self.special_tokens = special_tokens
        self.tokenizer = open_clip.SimpleTokenizer(special_tokens=special_tokens)
        self.vocab_size = model.vocab_size  # vocab_size=49408, <start> is 49406, <end> is 49407
        self.cfg_rate = cfg_rate

        if init_text[0] == "<random>":  # random from \mathcal{N}(0, 1)
            print('!!!!!!!!! random init embedding:', len(special_tokens), model.token_embedding.embedding_dim, '!!!!!!!!!!')
            self.special_embeddings = nn.Embedding(len(special_tokens), embedding_dim=model.token_embedding.embedding_dim)
        else:
            # initialization from model's embedding
            sp_emb_weights = init_special_embeddings(self.tokenizer, special_tokens, model, init_text, tokenwise_init)
            print('!!!!!!!!! new embedding shape:', sp_emb_weights.shape, '!!!!!!!!!!')
            self.special_embeddings = nn.Embedding(len(special_tokens), embedding_dim=model.token_embedding.embedding_dim, _weight=sp_emb_weights)

        self.pos_strengthen = pos_strengthen
        if not view_prompt:
            # rel_pos MLP
            self.rel_pos_model = RelPosModel(input_ch=4, out_ch=1024, pos_strengthen=pos_strengthen)
        else:
            self.rel_pos_model = None

        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        if len(inputs) > 1 and type(inputs[1]) == torch.Tensor:
            [text, rel_pos] = inputs
        else:
            text = inputs
            rel_pos = None

        B, nlayer, L = None, None, None
        if self.deep_prompt:
            tokens = []
            for text_ in text:
                tokens_ = tokenize(self.tokenizer, text_)  # [B,L]
                tokens.append(tokens_)
            tokens = torch.stack(tokens, dim=1)  # [B,nlayer,L]
            B, nlayer, L = tokens.shape
            tokens = tokens.reshape(B * nlayer, L)
        else:
            tokens = tokenize(self.tokenizer, text)  # [B,L]
        # special tokens 赋值>max_emb的数, 对normal_tokens clip, new tokens clip - max_emb, 并且获取mask
        token_mask = (tokens >= self.vocab_size).to(torch.long).unsqueeze(-1).to(self.device)
        tokens_regular = torch.clamp(tokens, 0, self.vocab_size - 1).to(self.device)
        tokens_special = torch.clamp_min(tokens.clone() - self.vocab_size, 0).to(self.device)
        emb_regular = self.model.token_embedding(tokens_regular)
        emb_special = self.special_embeddings(tokens_special)
        text_emb = emb_regular * (1 - token_mask) + emb_special * token_mask

        rel_pos_emb1, rel_pos_emb2 = None, None
        if rel_pos is not None:
            if self.pos_strengthen:
                rel_pos_emb1, rel_pos_emb2 = self.rel_pos_model(rel_pos)  # [B,1,C] replace the last sp token feature
            else:
                rel_pos_emb1 = self.rel_pos_model(rel_pos)
            text_emb[:, len(self.special_tokens) + 1, :] = rel_pos_emb1.to(text_emb.dtype)  # <start_token>使其顺延一个，所以仍是覆盖最后一个token

        if self.cfg_rate > 0.0 and self.training:  # CFG for training
            null_tokens = tokenize(self.tokenizer, [""]).to(self.device)  # [1, L]
            null_emb = self.model.token_embedding(null_tokens)  # [1, L, C]
            rdv = torch.rand(text_emb.shape[0])
            cfg_mask = (rdv < self.cfg_rate).to(dtype=torch.float32, device=text_emb.device).reshape(text_emb.shape[0], 1, 1)  # [B,1,1]
            text_emb = (1 - cfg_mask) * text_emb + cfg_mask * null_emb
        else:
            cfg_mask = None

        z = self.encode_with_transformer(text_emb.to(self.device))  # [B(B*nlayer),L,C]
        if self.deep_prompt:
            z = z.reshape(B, nlayer, L, -1)

        if rel_pos is not None and rel_pos_emb2 is not None:
            if cfg_mask is not None:
                cfg_mask = cfg_mask[:, 0]
                pose_z = rel_pos_emb2.to(text_emb.dtype) * (1 - cfg_mask) + z[:, -1, :] * cfg_mask
            else:
                pose_z = rel_pos_emb2.to(text_emb.dtype)
            z[:, -1, :] = pose_z  # [B,(1),C]

        return z

    def encode_with_transformer(self, x):
        # x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)
