import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel
import torch.nn.functional as F
import open_clip
from typing import Union, List
from ldm.util import default, count_params


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


class PromptCLIPEmbedder(AbstractEncoder):
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
                 view_prompt=True, view_num=4, view_token_len=30, **kwargs):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.deep_prompt = deep_prompt
        self.view_prompt = view_prompt
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
            raise NotImplementedError()
            deep_special_tokens = []
            for layer_i in range(cross_attn_layers):
                special_tokens_ = [t.replace('>', f'-layer{layer_i}>') for t in special_tokens]
                deep_special_tokens.extend(special_tokens_)
            special_tokens = deep_special_tokens
            init_text = init_text * cross_attn_layers
        
        print("max_view_num:", view_num)
        print("view_token_len:", view_token_len)
        if view_prompt:
            for j in range(view_num):
                for l in range(view_token_len):
                    special_tokens.append(f"<view_direct-{j}-{l}")
                    init_text.append("The whole image is splited into two parts with the same size, they share the same scene/landmark captured with different viewpoints and times")
        print("len special tokens: ", len(special_tokens))

        self.special_tokens = special_tokens
        self.tokenizer = open_clip.SimpleTokenizer(special_tokens=special_tokens)
        self.vocab_size = model.vocab_size  # vocab_size=49408, <start> is 49406, <end> is 49407

        if init_text[0] == "<random>":  # random from \mathcal{N}(0, 1)
            print('!!!!!!!!! random init embedding:', len(special_tokens), model.token_embedding.embedding_dim, '!!!!!!!!!!')
            self.special_embeddings = nn.Embedding(len(special_tokens), embedding_dim=model.token_embedding.embedding_dim)
        else:
            # initialization from model's embedding
            sp_emb_weights = init_special_embeddings(self.tokenizer, special_tokens, model, init_text, tokenwise_init)
            print('!!!!!!!!! new embedding shape:', sp_emb_weights.shape, '!!!!!!!!!!')
            self.special_embeddings = nn.Embedding(len(special_tokens), embedding_dim=model.token_embedding.embedding_dim, _weight=sp_emb_weights)
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

    def forward(self, text):
        B, nlayer, L = None, None, None
        if self.deep_prompt:
            raise NotImplementedError()
            tokens = []
            for text_ in text:
                tokens_ = tokenize(self.tokenizer, text_)  # [B,L]
                tokens.append(tokens_)
            tokens = torch.stack(tokens, dim=1)  # [B,nlayer,L]
            B, nlayer, L = tokens.shape
            tokens = tokens.reshape(B * nlayer, L)
        elif self.view_prompt:
            tokens = []
            for text_ in text:
                tokens_ = tokenize(self.tokenizer, text_)
                tokens.append(tokens_)
            tokens = torch.stack(tokens, dim=1)
            B, view, L = tokens.shape
            tokens = tokens.reshape(B * view, L)
        else:
            tokens = tokenize(self.tokenizer, text)  # [B,L]
        # special tokens 赋值>max_emb的数, 对normal_tokens clip, new tokens clip - max_emb, 并且获取mask
        token_mask = (tokens >= self.vocab_size).to(torch.long).unsqueeze(-1).to(self.device)
        tokens_regular = torch.clamp(tokens, 0, self.vocab_size - 1).to(self.device)
        tokens_special = torch.clamp_min(tokens.clone() - self.vocab_size, 0).to(self.device)
        emb_regular = self.model.token_embedding(tokens_regular)
        emb_special = self.special_embeddings(tokens_special)
        text_emb = emb_regular * (1 - token_mask) + emb_special * token_mask
        z = self.encode_with_transformer(text_emb.to(self.device))  # [B(B*nlayer),L,C]
        if self.deep_prompt:
            z = z.reshape(B, nlayer, L, -1)
        
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