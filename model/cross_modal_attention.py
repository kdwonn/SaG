import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from model.attention import TransformerLayer
from model.encoders import ImageTextEncoders, l2norm, ImageTextEncodersRecon
from model.decoder import MlpDecoder, TransformerDecoder

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.bn(x)
        x = rearrange(x, 'b d n -> b n d')
        x = F.relu(x)
        x = self.linear2(x)
        return x

class CrossModalAttention(nn.Module):
    def __init__(self, encoders: ImageTextEncoders, in_dim, args):
        super().__init__()
        self.encoders = encoders
        self.args = args
        self.amp = args.amp
        self.cma = TransformerLayer(
            query_dim=in_dim,
            ff_dim=in_dim,
            context_dim=in_dim,
            heads=args.cma_heads,
            dim_head=args.cma_head_dim,
            dropout=args.dropout,
            ff_activation='gelu',
            last_norm=True,
            last_fc=args.cma_last_fc,
            qk_norm=args.cma_qk_norm
        )

        self.cma_self = None
        if args.cma_self_attn:
            self.cma_self = TransformerLayer(
                query_dim=in_dim,
                ff_dim=in_dim,
                context_dim=None,
                heads=args.cma_heads,
                dim_head=args.cma_head_dim,
                dropout=args.dropout,
                ff_activation='gelu',
                last_norm=True,
                last_fc=False
            )

        self.last_mlp = None
        if args.cma_last_mlp:
            self.last_mlp = MLP(in_dim, in_dim // 2, in_dim)

        self.txt_l2 = args.info_txt_l2
        
    def forward(self, images, sentences, txt_len):
        with torch.cuda.amp.autocast(enabled=self.amp):
            img_emb, txt_emb, img_attn, txt_attn, img_residual, txt_residual, txt_bert =\
                self.encoders(images, sentences, txt_len)
            
            if self.cma_self is not None:
                img_emb = self.cma_self(img_emb)

            cm_feat = self.cma(
                repeat(txt_emb, 'b n d -> repeat b (n d)', repeat=img_emb.shape[0]), 
                context=img_emb
            )

            if self.last_mlp is not None:
                cm_feat = self.last_mlp(cm_feat)

            cm_feat = l2norm(cm_feat)
            
            if self.txt_l2:
                txt_emb = l2norm(txt_emb)
            
        return cm_feat, img_emb, txt_emb, img_attn, txt_attn, img_residual, txt_residual, txt_bert


class CrossModalAttentionRecon(nn.Module):
    def __init__(self, encoders: ImageTextEncodersRecon, in_dim, args):
        super().__init__()
        self.encoders = encoders
        self.args = args
        self.amp = args.amp
        self.cma = TransformerLayer(
            query_dim=in_dim,
            ff_dim=in_dim,
            context_dim=in_dim,
            heads=args.cma_heads,
            dim_head=args.cma_head_dim,
            dropout=args.dropout,
            ff_activation='gelu',
            last_norm=True,
            last_fc=args.cma_last_fc
        )
        
        self.cma_self = None
        if args.cma_self_attn:
            self.cma_self = TransformerLayer(
                query_dim=in_dim,
                ff_dim=in_dim,
                context_dim=None,
                heads=args.cma_heads,
                dim_head=args.cma_head_dim,
                dropout=args.dropout,
                ff_activation='gelu',
                last_norm=False,
                last_fc=False
            )

        self.last_mlp = None
        if args.cma_last_mlp:
            self.last_mlp = MLP(in_dim, in_dim // 2, in_dim)
        
        Decoder = {
            'mlp': MlpDecoder,
            'transformer': TransformerDecoder
        }[args.recon_decoder]

        self.decoder = Decoder(
            num_patches=int(args.crop_size/16),
            slot_dim=args.embed_dim,
            feat_dim=self.encoders.img_enc.local_feat_dim,
            normalizer=args.decoder_normalizer,
            self_attn=args.decoder_self_attn,
            pos_enc=args.decoder_pos_enc
        )

        self.txt_l2 = args.info_txt_l2
        
    def forward(self, images, sentences, txt_len):
        with torch.cuda.amp.autocast(enabled=self.amp):
            img_slot, img_feat, txt_emb, txt_attn, img_residual, txt_residual, txt_bert =\
                self.encoders(images, sentences, txt_len)
            img_feat_recon = self.decoder(img_slot)

            if self.cma_self is not None:
                img_slot = self.cma_self(img_slot)

            cm_feat = self.cma(
                repeat(txt_emb, 'b n d -> repeat b (n d)', repeat=img_slot.shape[0]), 
                context=img_slot
            )

            if self.last_mlp is not None:
                cm_feat = self.last_mlp(cm_feat)
            cm_feat = l2norm(cm_feat)

            if self.txt_l2:
                txt_emb = l2norm(txt_emb)
            
        return cm_feat, img_slot, txt_emb, img_feat_recon, img_feat, txt_bert