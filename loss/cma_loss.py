import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum, reduce


def l2norm(x):
    """L2-normalize columns of x"""
    return F.normalize(x, p=2, dim=-1)
    
class CMA_Loss(nn.Module):
    def __init__(self, margin, criterion, mining='hard', detach_target=False, 
            detach_img_target=False, i_t_loss=None, i_t_weight=0, temperature=1, 
            cm_i_weight=1, size_p_loss=None, size_p_weight=0):
        super().__init__()
        self.margin = margin
        self.mining = mining
        self.detach_target = detach_target
        self.detach_img_target = detach_img_target
        
        self.i_t_loss = i_t_loss
        self.i_t_weight = i_t_weight
        self.temperature = temperature
        assert not ((self.i_t_loss is not None) and self.i_t_weight == 0)
        self.cm_i_weight = cm_i_weight
        self.size_p_loss = size_p_loss
        self.size_p_weight = size_p_weight
        
        if criterion == 'contrastive':
            self.loss = self.contrastive_loss
        elif criterion == 'triplet':
            self.loss = self.triplet_loss
        elif criterion == 'info_nce':
            self.loss = self.info_nce_loss
        else:
            raise NotImplementedError
        
    def contrastive_loss(self, sim, gt):
        ap = sim[gt].reshape(sim.shape[0], 1)
        an = sim[~gt].reshape(sim.shape[0], sim.shape[1]-1)
        
        pos = (self.margin - ap).clamp(0.0)
        neg = (-self.margin + an).clamp(0.0)
        loss = (pos + neg).mean()
        
        return loss
    
    def triplet_loss(self, sim, gt):
        assert self.mining in ['hard', 'semi_hard', 'mean']
        ap = sim[gt].reshape(sim.shape[0], 1)
        an = sim[~gt].reshape(sim.shape[0], sim.shape[1]-1)
        
        if self.mining == 'mean':
            loss = (self.margin + an - ap).clamp(min=0.0)
            loss = loss.mean()
        elif self.mining == 'semi_hard':
            loss = (self.margin + an - ap).clamp(min=0.0, max=self.margin)
            num_tri = torch.nonzero(loss).shape[0]
            loss = loss.mean() if num_tri == 0 else loss.sum() / num_tri
        elif self.mining == 'hard':
            loss = (self.margin + an - ap).clamp(min=0.0)
            loss = loss.max(dim=-1)[0].mean()
        else:
            raise NotImplementedError
        return loss
    
    def info_nce_loss(self, sim, pos_gt, neg_gt):
        ap = sim[pos_gt].reshape(sim.shape[0], 1) / self.temperature
        an = sim[neg_gt].reshape(
            sim.shape[0], sim.shape[1]-1
        ) / self.temperature
        logit = torch.cat([ap, an], dim=-1)
        label = torch.zeros(logit.shape[0], dtype=torch.long).cuda()
        loss = F.cross_entropy(logit, label)
        return loss
        
    def forward(self, cm_emb, txt_emb, img_emb, img_attn=None, txt_bert=None):
        
        assert cm_emb.shape[0] == cm_emb.shape[1]
        assert cm_emb.shape[0] == txt_emb.shape[0]
        
        # (optional) size panalty loss
        size_p_loss, size_p_losses = self.size_p_loss(img_attn) \
            if (self.size_p_loss is not None) else (0, {})

        # (optional) i_t loss by image slots and txt embeddings
        i_t_loss, i_t_losses = self.i_t_loss(img_emb, txt_emb, img_emb, txt_emb) \
            if (self.i_t_loss is not None) else (0, {})

        #  finding marching pair using text
        txt_emb = rearrange(txt_emb, 'b n d -> b (n d)')
        if self.detach_target:
            txt_emb = txt_emb.detach()
        sim = torch.einsum('b d, b n d -> b n', txt_emb, rearrange(cm_emb, 'i t d -> t i d'))
        pos_gt = torch.eye(cm_emb.shape[0], dtype=torch.bool).cuda()
        neg_gt = ~pos_gt
        
        cm_t_loss = self.loss(sim, pos_gt, neg_gt)
        
        # finding matching pair using avg pooled slot
        cm_i_loss = .0

        loss = cm_t_loss + self.cm_i_weight * cm_i_loss + self.i_t_weight * i_t_loss + self.size_p_weight * size_p_loss
        losses = {'cm_loss': cm_t_loss, **i_t_losses, **size_p_losses}
        
        return loss, losses

class CMA_Loss_Fast(CMA_Loss):
    def __init__(
        self, margin, criterion, mining='hard', detach_target=False,
        detach_img_target=False, i_t_loss=None, i_t_weight=0, temperature=1, 
        cm_i_weight=1, size_p_loss=None, size_p_weight=0):
        super().__init__(
            margin,
            criterion,
            mining,
            detach_target,
            detach_img_target,
            i_t_loss,
            i_t_weight,
            temperature,
            cm_i_weight,
            size_p_loss,
            size_p_weight,
        )

    def pad_txts(self, txt_emb, txts_per_img):
        txt_chunks = torch.split(txt_emb, txts_per_img.tolist(), dim=0)
        padded_chunks = nn.utils.rnn.pad_sequence(sequences=txt_chunks, batch_first=True, padding_value=0)

        pad_mask = repeat(torch.arange(padded_chunks.shape[1]).cuda(), 'l -> repeat l', repeat=len(txts_per_img))
        pad_mask = pad_mask < txts_per_img.unsqueeze(-1)
        
        return padded_chunks, pad_mask

    def forward(self, cm_emb, txt_emb, txt_num, txt_bert=None):
        assert len(cm_emb) == len(txt_num)

        #  finding marching pair using text
        # TODO check whether it fits with multiple txts
        txt_emb = rearrange(txt_emb, 'b n d -> b (n d)')
        if self.detach_target:
            txt_emb = txt_emb.detach()
        sim = torch.einsum('b d, b n d -> b n', txt_emb, rearrange(cm_emb, 'i t d -> t i d'))

        # TODO redefine positive gt
        # pos_gt = torch.eye(cm_emb.shape[0], dtype=torch.bool).cuda()
        pos_gt = torch.repeat_interleave(
            torch.eye(cm_emb.shape[0], dtype=torch.bool).cuda(),
            repeats=txt_num, dim=0)
        neg_gt = ~pos_gt
        
        cm_t_loss = self.loss(sim, pos_gt, neg_gt)
        
        loss = cm_t_loss
        losses = {'cm_loss': cm_t_loss.item()}
        
        return loss, losses