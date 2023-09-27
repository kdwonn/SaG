import os
import numpy as np
import torch
import cv2
import math
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm

from data import get_test_loader, get_train_loader
from option import parser, verify_input_args
from model.encoders import ImageTextEncodersRecon
from model.cross_modal_attention import CrossModalAttentionRecon
from utils import f_out_hook
from einops import rearrange, reduce, einsum


def encode_data(model, data_loader, crop_size, img_num_embeds, embed_dim, args, encode_head=False):
    """Encode all images and sentences loadable by data_loader"""
    # switch to evaluate mode
    model.eval()
    
    # numpy array to keep all the embeddings
    cm_emb_sz = [len(data_loader.dataset), embed_dim]
    txt_emb_sz = [len(data_loader.dataset), embed_dim]
    img_emb_sz = [len(data_loader.dataset), img_num_embeds, embed_dim]
    cm_embs = torch.zeros(cm_emb_sz, requires_grad=False).cuda()
    txt_embs = torch.zeros(txt_emb_sz, requires_grad=False).cuda()
    img_embs = torch.zeros(img_emb_sz, requires_grad=False).cuda()
    
    agg_depth = len(model.encoders.img_enc.set_pred_module.agg.agg_blocks)
    num_slot = model.encoders.img_enc.set_pred_module.agg.num_latents
    head = model.encoders.img_enc.set_pred_module.agg.agg_blocks[0][0].fn.heads
    head_cm = model.cma.attn.fn.heads
    
    slot_a_maps = torch.zeros(
        [len(data_loader.dataset), agg_depth, num_slot, int(crop_size/16)**2], requires_grad=False).cuda()
    cm_a_maps = torch.zeros(
        [len(data_loader.dataset), num_slot], requires_grad=False).cuda()
    if encode_head:
        slot_head_a_maps = torch.zeros(
            [len(data_loader.dataset), head, num_slot, int(crop_size/16)**2], requires_grad=False).cuda()
    
    slot_a_map = []
    cm_a_map = []
        
    for i, data in tqdm(enumerate(data_loader)):
        img_len = None
        img, txt, txt_len, ids = data
        img, txt, txt_len = img.cuda(), txt.cuda(), txt_len.cuda()
        
        hdlr1 = model.encoders.img_enc.set_pred_module.agg.agg_blocks[0][0].fn.attn_holder.\
            register_forward_hook(f_out_hook(slot_a_map))
        hdlr2 = model.cma.attn.fn.attn_holder.\
            register_forward_hook(f_out_hook(cm_a_map))
        
        # compute the embeddings
        cm_emb, img_emb, txt_emb, _, _, _ = model.forward(img, txt, txt_len)
        del img, txt, img_len, txt_len
        slot_a_map = rearrange(
            torch.cat(slot_a_map, dim=0), 
            '(depth bs h) n d -> bs depth h n d', 
            depth=args.agg_depth, h=head)
        if encode_head:
            slot_head_a_maps[ids] = slot_a_map[:, -1, :, :, :]
        slot_a_map = reduce(slot_a_map, 'bs depth h n d -> bs depth n d', 'mean')
        slot_a_maps[ids] = slot_a_map
        slot_a_map = []

        cm_a_map = rearrange(
            torch.cat(cm_a_map, dim=0), 
            '(bs h) n d -> bs h n d', h=head_cm)
        cm_a_map = reduce(cm_a_map, 'bs h n d -> bs n d', 'mean')
        cm_a_maps[ids] = einsum(cm_a_map, 'b b d -> b d')
        cm_a_map = []
        
        hdlr1.remove()
        hdlr2.remove()

        # preserve the embeddings by copying from gpu and converting to numpy
        cm_embs[ids] = einsum(cm_emb, 'b b d -> b d')
        img_embs[ids] = img_emb.float()
        txt_embs[ids] = rearrange(txt_emb.float(), 'b n d -> (b n) d')
        
    return cm_embs, img_embs, txt_embs, slot_a_maps, cm_a_maps, (slot_head_a_maps if encode_head else None)


def compute_mask_IU(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    I = np.sum(np.logical_and(masks, target))
    U = np.sum(np.logical_or(masks, target))
    return I, U


def compute_mask_IU_torch(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    I = torch.sum(torch.logical_and(masks, target))
    U = torch.sum(torch.logical_or(masks, target))
    return I, U


def resize_and_crop(images, input_h, input_w):
    # Resize and crop images to input_h x input_w size
    B, C, H, W = images.size()
    scale = max(input_h / H, input_w / W)
    resized_h = int(round(H * scale))
    resized_w = int(round(W * scale))
    crop_h = int(math.floor(resized_h - input_h) / 2)
    crop_w = int(math.floor(resized_w - input_w) / 2)
    resize_image = transforms.functional.resize(images, [resized_h, resized_w])
    new_im = torch.zeros((B, C, input_h, input_w)).float() if images.is_cuda is False \
            else torch.zeros((B, C, input_h, input_w)).cuda().float()
    new_im[...] = resize_image[..., crop_h:crop_h+input_h, crop_w:crop_w+input_w]
    
    return new_im


def attn_matrix_sparsity(attn):
    attn = torch.cat(attn[:10], dim=0)
    under_treshold = (attn < 0.09).sum()
    entire_element = (attn >= 0).sum()
    return under_treshold / entire_element

def get_IoU_map(img_a_map_for_slot, feat_map_size, raw_label, raw_img_id, top_cm_idx
                , seg_path, cum_I, cum_U, mIoU, prec, recall, is_save=False):
    a_map = img_a_map_for_slot.reshape(1, feat_map_size, feat_map_size)
    a_map = transforms.functional.resize(a_map, list(raw_label.shape)).squeeze().cpu().numpy()
    a_map = ((a_map - a_map.min()) / (a_map.max() - a_map.min() + 1e-9))

    hard_pred = (a_map >=args.pseudo_threshold)
    I, U = compute_mask_IU(hard_pred, raw_label)
    if is_save:
        a_map_fname = 'threshold{:.1f}/{}_{}.png'.format(args.pseudo_threshold, raw_img_id, top_cm_idx)
        if not os.path.exists(os.path.join(seg_path, os.path.dirname(a_map_fname))):
            os.makedirs(os.path.join(seg_path, os.path.dirname(a_map_fname)))
        Image.fromarray(np.uint8(hard_pred * 255), 'L').save(os.path.join(seg_path, a_map_fname))
    
    cum_I += I
    cum_U += U
    prec += (I / hard_pred.sum() if I != 0 else 0)
    recall += I / raw_label.sum()
    mIoU += I / U
    return cum_I, cum_U, mIoU, prec, recall

def save_slot_attn_map(image, img_a_map_for_slot, feat_map_size, raw_label, raw_img_id, a, seg_path, cm_weight, raw_sentence):
    a_map = img_a_map_for_slot.reshape(1, feat_map_size, feat_map_size)
    a_map = transforms.functional.resize(a_map, list(raw_label.shape)).squeeze()
    a_map = ((a_map - a_map.min()) / (a_map.max() - a_map.min()+1e-9)).cpu().numpy()
    a_map = cv2.applyColorMap(np.uint8(a_map * 255), cv2.COLORMAP_JET)[:, :, ::-1]
    image = transforms.functional.resize(image, list(raw_label.shape))
    img_and_amap = cv2.addWeighted(np.uint8(image.permute(1, 2, 0) * 255), 0.3, a_map, 0.7, 0)
    a_map_fname = 'amap/{}_slot{}_{:.3f}_{}.png'.format(raw_img_id, a, cm_weight, raw_sentence)
    Image.fromarray(img_and_amap).save(os.path.join(seg_path, a_map_fname))

def save_head_attn_map(image, head_a_map, feat_map_size, raw_label, raw_img_id, a, seg_path, head):
    a_map = head_a_map.reshape(1, feat_map_size, feat_map_size)
    a_map = transforms.functional.resize(a_map, list(raw_label.shape)).squeeze()
    a_map = ((a_map - a_map.min()) / (a_map.max() - a_map.min()+1e-9)).cpu().numpy()
    a_map = cv2.applyColorMap(np.uint8(a_map * 255), cv2.COLORMAP_JET)[:, :, ::-1]
    image = transforms.functional.resize(image, list(raw_label.shape))
    img_and_amap = cv2.addWeighted(np.uint8(image.permute(1, 2, 0) * 255), 0.3, a_map, 0.7, 0)
    a_map_fname = 'head_amap/{}_slot{}_head{}.png'.format(raw_img_id, a, head)
    Image.fromarray(img_and_amap).save(os.path.join(seg_path, a_map_fname))

def eval_seg(model, args, split='val'):
    print('Loading dataset')
    data_loader = get_test_loader(args) if split == 'val' else get_train_loader(args)
    dataset = data_loader.dataset
    data_path = dataset.root

    print('Computing results... (eval_on_gpu={})'.format(args.eval_on_gpu))
    cm_embs, img_embs, txt_embs, img_a_map, cm_a_map, head_a_map = encode_data(
        model, 
        data_loader,
        args=args,
        crop_size=args.crop_size,
        img_num_embeds=args.img_num_embeds, 
        embed_dim=args.embed_dim,
        encode_head=args.save_head_map
    )
    
    seg_path = os.path.join(os.path.dirname(args.ckpt), 'seg'.format(os.path.basename(args.ckpt)))
    if not os.path.exists(seg_path):
        os.makedirs(seg_path)
    if not os.path.exists(os.path.join(seg_path, 'threshold{:.1f}'.format(args.pseudo_threshold))):
        os.makedirs(os.path.join(seg_path, 'threshold{:.1f}'.format(args.pseudo_threshold)))
    if not os.path.exists(os.path.join(seg_path, 'amap')):
        os.makedirs(os.path.join(seg_path, 'amap'))
    if not os.path.exists(os.path.join(seg_path, 'head_amap')):
        os.makedirs(os.path.join(seg_path, 'head_amap'))
    if not os.path.exists(os.path.join(seg_path, '{}_image'.format(split))):
        os.makedirs(os.path.join(seg_path, '{}_image'.format(split)))
    if not os.path.exists(os.path.join(seg_path, '{}_label'.format(split))):
        os.makedirs(os.path.join(seg_path, '{}_label'.format(split)))
        
    if 'res' in args.img_backbone:
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    elif 'vit' in args.img_backbone:
        inv_normalize = transforms.Normalize(
            mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
            std=[1/0.5, 1/0.5, 1/0.5]
        )
    else:
        raise NotImplementedError
    
    cm_max_I, cm_max_U, cm_avg_I, cm_avg_U, max_cm_mIoU, avg_cm_mIoU = 0., 0., 0., 0., 0., 0.
    max_cm_prec, avg_cm_prec, max_cm_recall, avg_cm_recall = 0., 0., 0., 0.

    no_cm_I, no_cm_U, no_cm_mIoU = 0., 0., 0.
    no_cm_prec, no_cm_recall = 0., 0.

    cm_min_I, cm_min_U, min_cm_mIoU = 0., 0., 0.
    min_cm_prec, min_cm_recall = 0., 0.

    c_s_max_I, c_s_max_U, c_s_avg_I, c_s_avg_U, max_c_s_mIoU, avg_c_s_mIoU = 0., 0., 0., 0., 0., 0.
    max_c_s_prec, avg_c_s_prec, max_c_s_recall, avg_c_s_recall = 0., 0., 0., 0.

    t_s_max_I, t_s_max_U, t_s_avg_I, t_s_avg_U, max_t_s_mIoU, avg_t_s_mIoU = 0., 0., 0., 0., 0., 0.
    max_t_s_prec, avg_t_s_prec, max_t_s_recall, avg_t_s_recall = 0., 0., 0., 0.
    
    cm_slot_sim = einsum(cm_embs, img_embs, 'b d, b s d -> b s')
    txt_slot_sim = einsum(txt_embs, img_embs, 'b d, b s d -> b s')

    t= tqdm(range(len(dataset)), desc='Eval', leave=True)
    for i in t:
        image = inv_normalize(dataset[i][0])
        raw_sentence, raw_img_id, _, raw_label, _ =  dataset.get_raw_item(i)
        feat_map_size = int(args.crop_size / 16)

        cm_a = cm_a_map[i]
        cm_slot_a = cm_slot_sim[i]
        txt_slot_a = txt_slot_sim[i]

        # Max pooling: Attention map of the closest slot
        #   - cross-attetnion positive similarity score
        top_cm_idx= torch.argmax(cm_a)
        img_a_map_for_slot = img_a_map[i, -1, top_cm_idx]
        cm_max_I, cm_max_U, max_cm_mIoU, max_cm_prec, max_cm_recall = get_IoU_map(img_a_map_for_slot
                        , feat_map_size, raw_label, raw_img_id, top_cm_idx, os.path.join(seg_path,'max_cm')
                        , cm_max_I, cm_max_U, max_cm_mIoU, max_cm_prec, max_cm_recall, is_save=False)


        # Average pooling: Weighted sum of the attention maps with similarity scores between slots
        #   - cross-attetnion positive similarity score
        avg_a_map = img_a_map[i, -1, :]
        avg_a_map = (cm_a.unsqueeze(1) * avg_a_map).sum(dim=0)
        cm_avg_I, cm_avg_U, avg_cm_mIoU, avg_cm_prec, avg_cm_recall = get_IoU_map(avg_a_map
                        , feat_map_size, raw_label, raw_img_id, '', os.path.join(seg_path, 'avg_cm')
                        , cm_avg_I, cm_avg_U, avg_cm_mIoU, avg_cm_prec, avg_cm_recall, is_save=True)

        
        no_cm_a_map = img_a_map[i, -1, :].mean(dim=0)
        no_cm_I, no_cm_U, no_cm_mIoU, no_cm_prec, no_cm_recall = get_IoU_map(no_cm_a_map
                        , feat_map_size, raw_label, raw_img_id, '', os.path.join(seg_path, 'no_cm')
                        , no_cm_I, no_cm_U, no_cm_mIoU, no_cm_prec, no_cm_recall, is_save=True)
        

        min_cm_idx= torch.argmin(cm_a)
        img_a_map_for_slot = img_a_map[i, -1, min_cm_idx]
        cm_min_I, cm_min_U, min_cm_mIoU, min_cm_prec, min_cm_recall = get_IoU_map(img_a_map_for_slot
                        , feat_map_size, raw_label, raw_img_id, top_cm_idx, os.path.join(seg_path,'min_cm')
                        , cm_min_I, cm_min_U, min_cm_mIoU, min_cm_prec, min_cm_recall, is_save=False)

        if i < 1:
            Image.fromarray(np.uint8(image.permute(1, 2, 0) * 255)).save(os.path.join(seg_path, '{}_image'.format(split), raw_img_id+'.png'))
            for a in range(img_a_map.shape[2]):
                img_a_map_for_slot = img_a_map[i, -1, a]
                cm_weight = cm_a[a]
                save_slot_attn_map(image, img_a_map_for_slot, feat_map_size, raw_label, raw_img_id, a, seg_path, cm_weight, raw_sentence)

        if i < 10 and args.save_head_map:
            for a, h in [(a, h) for a in range(img_a_map.shape[2]) for h in range(head_a_map.shape[1])]:
                img_a_map_for_head = head_a_map[i, h, a]
                save_head_attn_map(image, img_a_map_for_head, feat_map_size, raw_label, raw_img_id, a, seg_path,h)

        t.set_postfix({
            "Average":" [%0.1f thr] [cm:%.3f%%] | [max:%.3f%%] | [no_cm:%.3f%%] | [min:%.3f%%]" \
                % (args.pseudo_threshold, avg_cm_mIoU*(100/(i+1)), max_cm_mIoU*(100/(i+1)), no_cm_mIoU*(100/(i+1)), min_cm_mIoU*(100/(i+1)))
            })

    print(
        'cIoU: cm %.3f max %.3f avg %.3f min %.3f' % (
            100*(cm_avg_I/cm_avg_U),
            100*(cm_max_I/cm_max_U),
            100*(no_cm_I/no_cm_U),
            100*(cm_min_I/cm_min_U)
        )
    )
    
    val_cm_dict = {
            'max_cIoU': 100 * (cm_max_I/cm_max_U),
            'max_mIoU': max_cm_mIoU * (100/len(dataset)),
            'max_prec': max_cm_prec * (100/len(dataset)),
            'max_recall': max_cm_recall * (100/len(dataset)),
            'avg_cIoU': 100 * (cm_avg_I/cm_avg_U),
            'avg_mIoU': avg_cm_mIoU * (100/len(dataset)),
            'avg_prec': avg_cm_prec * (100/len(dataset)),
            'avg_recall': avg_cm_recall * (100/len(dataset))
        }
    print(val_cm_dict)
    print('[avg_mIoU: %.3f] on [%s set] of [%s dataset]!!' % (avg_cm_mIoU * (100/len(dataset)), split, os.path.basename(args.data_path)))
    return 


if __name__ == '__main__':
    args = verify_input_args(parser.parse_args())
    opt = verify_input_args(parser.parse_args())
    
    # load model and options
    if args.ckpt == '':
        # args.ckpt = os.path.join('./run_checkpoint', args.remark, 'ckpt.pth.tar')
        args.ckpt = os.path.join('./run_checkpoint', args.remark, 'model_best.pth.tar')
        if not os.path.isfile(args.ckpt):
            args.ckpt = os.path.join('./logs', args.remark, 'model_best.pth.tar')
        print(args.ckpt)
    assert os.path.isfile(args.ckpt)
    
    model = CrossModalAttentionRecon(ImageTextEncodersRecon(args), args.embed_dim, args)
            
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda() if args.multi_gpu else model.cuda()
        torch.backends.cudnn.benchmark = True
    
    # Reproduced weight
    state_dict = torch.load(args.ckpt)['model']
    print('IoU: %.3f'% (torch.load(args.ckpt)['iou']))
    model.load_state_dict(state_dict)
    
    with torch.no_grad():
        eval_seg(model, args, split='val')