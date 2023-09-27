import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from model.pos_encoding import PositionEmbeddingLearned2D, PositionEmbeddingSine2D, build_position_encoding
from model.attention import TransformerLayer

def onehot_argmax(t, dim=1):
    # Compute one-hot version of argmax
    softmax = F.softmax(t, dim=1)
    onehot = torch.zeros_like(t)
    onehot = onehot.scatter_(dim, torch.argmax(t, dim=dim).unsqueeze(1), 1.)
    onehot = onehot + softmax - softmax.detach()
    return onehot

class MlpDecoder(nn.Module):
    def __init__(self, num_patches, slot_dim, feat_dim, normalizer='softmax', self_attn='False', pos_enc='learned') -> None:
        super().__init__()
        self.pos_emb = build_position_encoding(slot_dim, pos_enc, axis=2)
        self.num_patches = num_patches
        self.alpha_holder = nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, feat_dim+1)
        )
        self.normalizer = {
            'softmax': F.softmax,
            'max': onehot_argmax
        }[normalizer]

        self.self_attn = None
        if self_attn:
            self.self_attn = TransformerLayer(
                query_dim=slot_dim,
                ff_dim=slot_dim,
                context_dim=slot_dim,
                heads=4,
                ff_activation='gelu',
                last_norm=True
            )

    def forward(self, slots):
        if self.self_attn is not None:
            slots = self.self_attn(slots, context=slots)
            
        slots = repeat(slots, 'b s d -> b s h w d', h=self.num_patches, w=self.num_patches)
        slots = rearrange(slots, 'b s h w d -> b s (h w) d') + self.pos_emb(slots[:, 0, :, :, :]).unsqueeze(1)
        feat_decode = self.mlp(slots)
        feat, alpha = feat_decode[:, :, :, :-1], feat_decode[:, :, :, -1]
        alpha = self.alpha_holder(self.normalizer(alpha, dim=1))
        recon = einsum(feat, alpha, 'b s hw d, b s hw -> b hw d')
        return recon

class TransformerDecoder(nn.Module):
    def __init__(self, num_patches, slot_dim, feat_dim, normalizer='sofmax', pos_enc='learned') -> None:
        super().__init__()
        self.pos_emb = build_position_encoding(slot_dim, pos_enc, axis=2)
        self.num_patches = num_patches
        self.alpha_holder = nn.Identity()
        get_layer = lambda x=False: TransformerLayer(
            query_dim=slot_dim,
            ff_dim=slot_dim,
            context_dim=slot_dim,
            heads=4,
            ff_activation='gelu',
            last_norm=x
        )

        self.tr1 = get_layer()
        # self.tr2 = get_layer()
        # self.tr3 = get_layer()
        self.last_layer = nn.Linear(slot_dim, feat_dim+1)

        self.normalizer = {
            'softmax': F.softmax,
            'max': onehot_argmax
        }[normalizer]

    def forward(self, slots):
        slots = repeat(slots, 'b s d -> b s h w d', h=self.num_patches, w=self.num_patches)
        slots = rearrange(slots, 'b s h w d -> b s (h w) d') + self.pos_emb(slots[:, 0, :, :, :]).unsqueeze(1)
        
        num_slots = slots.shape[1]
        slots = rearrange(slots, 'b s n d -> (b s) n d')
        slots = self.tr1(slots, context=slots)
        # slots = self.tr2(slots, context=slots)
        # slots = self.tr3(slots, context=slots)
        feat_decode = self.last_layer(slots)
        feat_decode = rearrange(slots, '(b s) n d -> b s n d', s=num_slots)
        
        feat, alpha = feat_decode[:, :, :, :-1], feat_decode[:, :, :, -1]
        alpha = self.alpha_holder(self.normalizer(alpha, dim=1))
        recon = einsum(feat, alpha, 'b s hw d, b s hw -> b hw d')
        return recon