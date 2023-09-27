import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torchvision
import yaml

from torch.autograd import Variable
from einops import rearrange, reduce
from model.aggregator import Aggregator
from transformers import BertModel
from pathlib import Path
from timm.models.helpers import load_custom_pretrained
from model.vit import VisionTransformer
from timm.models.vision_transformer import default_cfgs
from model.attention import TransformerLayer


def get_img_backbone(arch, pretrained, num_layers, img_size):
    if arch == 'resnext_wsl':
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        model.avgpool = nn.Identity()
        model.fc = nn.Identity()
        feat_dim =2048
    elif 'vit' in arch or 'deit' in arch:
        cfg = yaml.load(open(Path(__file__).parent / 'img_encoder_cfg.yaml', 'r'), Loader=yaml.FullLoader)
        cfg = cfg["model"][arch]
        # cfg["image_size"] = (img_size, img_size)
        cfg["image_size"] = (480, 480)
        cfg["backbone"] = arch
        cfg["dropout"] = 0
        cfg["drop_path_rate"] = 0.1
        cfg["n_cls"] = 1000
        cfg["d_ff"] = 4 * cfg["d_model"]
        cfg["n_layers"] = num_layers
        
        if arch in default_cfgs:
            default_cfg = default_cfgs[arch]
        else:
            default_cfg = dict(
                pretrained=False,
                num_classes=1000,
                drop_rate=0.0,
                drop_path_rate=0.0,
                drop_block_rate=None,
            )
        default_cfg["input_size"] = (
            3,
            cfg["image_size"][0],
            cfg["image_size"][1],
        )
    
        _, _ = cfg.pop('normalization'), cfg.pop('backbone')
        model = VisionTransformer(**cfg)
        load_custom_pretrained(model, default_cfg)
        feat_dim = cfg['d_model']
    else:
        model = torchvision.models.__dict__[arch](pretrained=pretrained)
        model.avgpool = nn.Identity()
        model.fc = nn.Identity()
        feat_dim = 2048
    return model, feat_dim


def l2norm(x):
    """L2-normalize columns of x"""
    return F.normalize(x, p=2, dim=-1)


def get_pad_mask(max_length, lengths, set_pad_to_one=True):
    ind = torch.arange(0, max_length).unsqueeze(0)
    if torch.cuda.is_available():
        ind = ind.cuda()
    mask = Variable((ind >= lengths.unsqueeze(1))) if set_pad_to_one \
        else Variable((ind < lengths.unsqueeze(1)))
    return mask.cuda() if torch.cuda.is_available() else mask


def variable_len_pooling(data, input_lens, reduction):
    data = rearrange(data, 'b ... d -> b (...) d')
    if input_lens is None:
        if reduction =='avg':
            ret = reduce(data, 'h i k ->  h k', 'mean')
        elif reduction =='max':
            ret = reduce(data, 'h i k ->  h k', 'max')
        else:
            raise NotImplementedError
    else:
        B, N, D = data.shape
        idx = torch.arange(N).unsqueeze(0).expand(B, -1).cuda()
        idx = idx < input_lens.unsqueeze(1)
        idx = idx.unsqueeze(2).expand(-1, -1, D)
        if reduction == 'avg':
            ret = (data * idx.float()).sum(1) / input_lens.unsqueeze(1).float()
        elif reduction == 'max':
            ret = data.masked_fill(~idx, -torch.finfo(data.dtype).max).max(1)[0]
        else:
            raise NotImplementedError
    return ret

class SetPredictionModule(nn.Module):
    def __init__(
        self, 
        num_embeds,
        axis, 
        pos_enc, 
        args
    ):
        super(SetPredictionModule, self).__init__()
        self.num_embeds = num_embeds
        self.residual_norm = nn.LayerNorm(args.embed_dim)
        self.use_residual = args.agg_residual
        self.fc = nn.Linear(args.embed_dim, 1024) if args.agg_last_fc else nn.Identity()
        self.agg = Aggregator(
            depth = args.agg_depth,
            input_channels = args.agg_input_dim,
            input_axis = axis,
            num_latents = num_embeds,
            latent_dim = args.agg_query_dim,
            cross_heads = args.agg_cross_head,
            latent_heads = args.agg_latent_head,
            cross_dim_head = args.agg_cross_dim,
            latent_dim_head = args.agg_latent_dim,
            num_classes = args.embed_dim,
            attn_dropout = args.dropout,
            ff_dropout = args.dropout,
            weight_tie_layers = args.agg_weight_sharing,
            self_per_cross_attn = args.agg_self_per_cross_attn,
            self_before_cross_attn = args.agg_self_before_cross_attn,
            query_self_attn = args.agg_query_self_attns,
            pos_enc_type = pos_enc,
            last_fc = args.agg_last_fc,
            norm_pre = args.agg_pre_norm,
            norm_post = args.agg_post_norm,
            activation = args.agg_activation,
            last_ln = args.agg_last_ln,
            ff_mult = args.agg_ff_mult,
            cross_attn_type=args.agg_cross_attn_type,
            more_dropout = args.agg_more_dropout,
            xavier_init = args.agg_xavier_init,
            thin_ff = args.agg_thin_ff,
            query_type = args.agg_query_type,
            first_order=args.agg_first_order,
            gumbel_attn=args.agg_gumbel_attn,
            last_gumbel=args.agg_gumbel_last,
            gumbel_tau=args.gumbel_tau,
            cascade_factor=args.cascade_factor,
            var_scaling=args.agg_var_scaling,
            gru=args.agg_gru
        )
        
    def forward(self, local_feat, global_feat=None, pad_mask=None, txt_emb=None):
        set_prediction, attn = self.agg(local_feat, mask=pad_mask, txt_emb=txt_emb) 
        if global_feat is not None and self.use_residual:
            global_feat = global_feat.unsqueeze(1).repeat(1, self.num_embeds, 1)
            out = self.residual_norm(global_feat) + set_prediction
        else:
            out = self.residual_norm(set_prediction)
    
        return out, attn, set_prediction


class ImageTextEncoders(nn.Module):
    def __init__(self, args):
        super(ImageTextEncoders, self).__init__()

        self.img_enc = EncoderImage(args)
        self.txt_enc = EncoderTextBERT(args)
        self.amp = args.amp

    def forward(self, images, sentences, txt_len):
        with torch.cuda.amp.autocast(enabled=self.amp):
            txt_emb, txt_attn, txt_residual, bert_emb = self.txt_enc(Variable(sentences), txt_len)
            img_slot, img_feat, img_attn, img_residual = self.img_enc(Variable(images))
            return img_slot, txt_emb, img_attn, txt_attn, img_residual, txt_residual, bert_emb


class ImageTextEncodersRecon(nn.Module):
    def __init__(self, args):
        super(ImageTextEncodersRecon, self).__init__()

        self.img_enc = EncoderImage(args)
        self.txt_enc = EncoderTextBERT(args)
        self.itot_cross_attn = TextIdentity()
        self.amp = args.amp
        self.slot_cond = args.slot_cond

    def forward(self, images, sentences, txt_len):
        with torch.cuda.amp.autocast(enabled=self.amp):
            txt_emb, words_emb, txt_residual, bert_emb = self.txt_enc(Variable(sentences), txt_len)
            img_slot, img_feat, img_attn, img_residual = self.img_enc(Variable(images), txt_emb=txt_emb if self.slot_cond else None)
            return img_slot, img_feat, txt_emb, words_emb, img_residual, txt_residual, bert_emb


class TextIdentity(nn.Module):
    def __init__(self):
        super(TextIdentity, self).__init__()
        
    def forward(self, txt_emb, img_dim):
        return txt_emb


class EncoderImage(nn.Module):
    def __init__(self, args):
        super(EncoderImage, self).__init__()
        self.butd = 'butd' in args.data_name
        self.global_feat_holder = nn.Identity()

        # Backbone CNN
        self.img_backbone, self.local_feat_dim = get_img_backbone(args.img_backbone, True, args.num_layers, args.crop_size)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if args.agg_1x1_mlp:
            self.spm_1x1 = nn.Sequential(
                nn.LayerNorm(self.local_feat_dim),
                nn.Linear(self.local_feat_dim, self.local_feat_dim),
                nn.ReLU(),
                nn.Linear(self.local_feat_dim, args.agg_input_dim)
            )
        else:
            self.spm_1x1 = nn.Linear(self.local_feat_dim, args.agg_input_dim)
        
        self.num_patches = int(args.crop_size / 16)
        self.set_pred_module = SetPredictionModule(
            num_embeds=args.img_num_embeds, 
            axis=2, 
            pos_enc=args.agg_pos_enc,
            args=args
        )
        
        assert args.img_res_pool in ['avg', 'max']
        self.img_res_pool = args.img_res_pool
        self.residual_fc = nn.Linear(self.local_feat_dim, args.embed_dim)
        self.residual_pool = variable_len_pooling

        if args.agg_xavier_init:
            self.set_prediction_module.init_weights()
        for idx, param in enumerate(self.img_backbone.parameters()):
            param.requires_grad = args.img_finetune

    def residual_connection(self, x, l):
        x = self.residual_fc(self.residual_pool(x, l, self.img_res_pool))
        return x

    def init_weights(self):
        def fn(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        self.mlp.apply(fn)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, images, txt_emb=None):
        out_nxn = self.img_backbone(images)[:, 1:, :]
        out_nxn = rearrange(out_nxn, 'b (h w) d -> b h w d', h=self.num_patches, w=self.num_patches)
        out, attn, residual = self.set_pred_module(
            local_feat=self.spm_1x1(out_nxn), 
            global_feat=self.global_feat_holder(self.residual_connection(out_nxn, None)),
            pad_mask=None,
            txt_emb=txt_emb
        )

        return out, rearrange(out_nxn, 'b ... d -> b (...) d'), attn, residual

    
class EncoderTextBERT(nn.Module):

    def __init__(self, args):
        super(EncoderTextBERT, self).__init__()
        embed_dim = args.embed_dim
        self.embed_dim = embed_dim
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(self.bert.get_input_embeddings().weight.shape[1], self.embed_dim)
        self.txt_pooling = args.txt_pooling

        # Sentence embedding
        assert self.txt_pooling in ['cls', 'max', 'avg']        
        self.dropout = nn.Dropout(args.dropout if not args.text_no_dropout else 0)

        self.agg_l2 = args.txt_l2

    
    def residual_connection(self, bert_out, bert_out_cls, lengths):
        if self.txt_pooling == 'cls':
            ret = bert_out_cls
        elif self.txt_pooling == 'max':
            ret = variable_len_pooling(bert_out, lengths, reduction='max')
        elif self.txt_pooling == 'avg':
            ret = variable_len_pooling(bert_out, lengths, reduction='avg')
        return ret

    def forward(self, x, lengths):
        bert_attention_mask = (x != 0).float()
        pie_attention_mask = (x == 0)
        bert_emb = self.bert(x, bert_attention_mask)
        bert_emb = bert_emb[0]
        cap_len = lengths
        bert_emb_out = self.residual_connection(
            bert_emb,
            bert_emb[:,0],
            cap_len
        )
        local_cap_emb = self.linear(bert_emb)
        out = self.residual_connection(
            local_cap_emb, 
            local_cap_emb[:, 0], 
            cap_len
        )
        
        out = rearrange(out, 'b (n d) -> b n d', n=1)
        out = l2norm(out) if self.agg_l2 else out
        return out, local_cap_emb, out, bert_emb_out
