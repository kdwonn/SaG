import os 
import os.path as osp
import math
import torch
import wandb
import shutil
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.cuda.amp
import torchvision.transforms as transforms
from einops import rearrange
from tqdm import tqdm

import data
from model.encoders import ImageTextEncodersRecon
from model.cross_modal_attention import CrossModalAttentionRecon
from loss.cma_loss import CMA_Loss, CMA_Loss_Fast
from utils import AverageMeter, set_seed

from option import parser, verify_input_args
from sync_batchnorm import convert_model, SynchronizedBatchNorm2d
from eval_cma_recon import encode_data, compute_mask_IU


def train(epoch, total_iter, data_loader, model, criterion, recon_criterion, recon_weight, 
        optimizer, scaler, recon_warm, args, scheduler = None, bertemb_dict = None):
    # switch to train mode
    model.train()
    if args.bn_eval:
        modules = model.module.modules() if args.multi_gpu else model.modules()
        for m in modules: 
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
    
    # average meters to record the training statistics
    losses = AverageMeter()
    stat_dict = dict()
    losses_dict = dict()
    losses_dict['cm_loss'] = AverageMeter()
    losses_dict['recon'] = AverageMeter()
    
    for itr, data in enumerate(data_loader):
        total_iter += 1

        if args.fast_batch:
            img, txt, txt_len, recovery, num_txts_per_img, ids = data
            img, txt, txt_len, recovery, num_txts_per_img = \
                img.cuda(), txt.cuda(), txt_len.cuda(), recovery.cuda(), num_txts_per_img.cuda()
        else:
            img, txt, txt_len, ids = data
            img, txt, txt_len = img.cuda(), txt.cuda(), txt_len.cuda()

        with torch.cuda.amp.autocast(enabled=args.amp):
            cm_feat, img_emb, txt_emb, img_feat_recon, img_feat, txt_bert = model.forward(img, txt, txt_len) 
            # Use pre-extracted text embedding from external LM for sampling
            if args.pre_bertemb:
                bertemb_list = []
                for i, idx in enumerate(ids):
                    sentence, _, _, _, _ = data_loader.dataset.get_raw_item(idx)
                    bertemb_list.append(bertemb_dict[sentence])
                    txt_bert = torch.stack(bertemb_list)

            if recon_warm:
                loss, loss_dict = 0, {}
            else:
                if args.fast_batch:
                    txt_emb, cm_feat = txt_emb[recovery], cm_feat[:, recovery, :]
                    loss, loss_dict = criterion(cm_feat, txt_emb, num_txts_per_img, txt_bert=txt_bert)
                else:
                    loss, loss_dict = criterion(cm_feat, txt_emb, img_emb, txt_bert=txt_bert)
            
            recon_loss = recon_criterion(img_feat_recon, img_feat.detach())
            loss_dict['recon'] = recon_loss
            loss = loss + recon_weight * recon_loss

            if total_iter < args.lr_warmup_iter:
                loss *= float(total_iter) / args.lr_warmup_iter
            
        if torch.isnan(loss).any():
            print("!! NaN loss detected !!")
            import ipdb; ipdb.set_trace()
            
        losses.update(loss)
        for key, val in loss_dict.items():
            losses_dict[key].update(val)

        # Backprop
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        wandb.log({'iter':total_iter}) if not args.no_wandb else None
        if scheduler is not None and total_iter >= args.lr_warmup_iter:
            scheduler.step()
        
        # Print log info
        if itr > 0 and (itr % args.log_step == 0 or itr + 1 == len(data_loader)):
            log_msg = 'loss: %.4f (%.4f)' %(losses.val, losses.avg)
            for key, val in losses_dict.items():
                log_msg += ', %s: %.4f, (%.4f)' %(key.replace('_loss',''), val.val, val.avg)
            n = int(math.ceil(math.log(len(data_loader) + 1, 10)))
            print('[%d][%*d/%d] %s' %(epoch, n, itr, len(data_loader), log_msg))

            
        

    del cm_feat, txt_emb, loss
    return losses.avg, losses_dict, stat_dict, total_iter

def validation(epoch, data_loader, model, criterion, recon_criterion, recon_weight, args):
    with torch.no_grad():
        #NOTE Compute loss on the validation split
        losses = AverageMeter()
        losses_dict = dict()
        losses_dict['cm_loss'] = AverageMeter()
        losses_dict['recon'] = AverageMeter()
            
        for _, data in tqdm(enumerate(data_loader)):
            img, txt, txt_len, _ = data
            img, txt, txt_len = img.cuda(), txt.cuda(), txt_len.cuda()
            
            with torch.cuda.amp.autocast(enabled=args.amp):
                cm_feat, img_emb, txt_emb, img_feat_recon, img_feat, txt_bert = model.forward(img, txt, txt_len) 

                loss, loss_dict = criterion(cm_feat, txt_emb, img_emb, txt_bert=txt_bert)
                recon_loss = recon_criterion(img_feat_recon, img_feat.clone().detach())
                loss_dict['recon'] = recon_loss

                loss = loss + recon_weight * recon_loss

            if torch.isnan(loss).any():
                print("!! NaN loss detected !!")
                import ipdb; ipdb.set_trace()
                
            losses.update(loss)
            for key, val in loss_dict.items():
                losses_dict[key].update(val)
        
        del img, txt, txt_len, cm_feat, img_emb, txt_emb, img_feat_recon, img_feat
        
        log_msg = 'loss: %.4f (%.4f)' %(losses.val, losses.avg)
        for key, val in losses_dict.items():
            log_msg += ', %s: %.4f, (%.4f)' %(key.replace('_loss',''), val.val, val.avg)
        print('Epoch [%d] validation: %s' %(epoch, log_msg))
        
        # Compute cumulative IoU on the validation split. Considering two different pseudo labelling policy
        cum_max_I, cum_max_U = 0., 0.
        cum_avg_I, cum_avg_U = 0., 0.
        avg_mIoU, max_mIoU = 0., 0.
        max_precision, avg_precision = 0., 0.
        max_recall, avg_recall = 0., 0.
        dataset = data_loader.dataset
        _, _, _, img_a_map, cm_a_map, _ = encode_data(
            model, 
            data_loader,
            crop_size=args.crop_size,
            img_num_embeds=args.img_num_embeds, 
            embed_dim=args.embed_dim,
            args=args
        )
        
        t= tqdm(range(len(dataset)), desc='Evaluating', leave=True)
        for i in t:
            _, raw_img_id, _, raw_label, _ =  dataset.get_raw_item(i)
            feat_map_size = int(args.crop_size / 16)
            
            cm_a = cm_a_map[i]
            top_slot_idx = torch.argmax(cm_a)

            # Pseudo label: Attention map of the closest slot
            img_a_map_for_slot = img_a_map[i, -1, top_slot_idx]
            a_map = img_a_map_for_slot.reshape(1, feat_map_size, feat_map_size)
            a_map = transforms.functional.resize(a_map, list(raw_label.shape)).squeeze().cpu().numpy()
            a_map = ((a_map - a_map.min()) / (a_map.max() - a_map.min() + 1e-9))

            hard_pred = (a_map >= args.pseudo_threshold)
            I, U = compute_mask_IU(hard_pred, raw_label)
            cum_max_I += I
            cum_max_U += U
            
            max_precision += (I / hard_pred.sum() if I != 0 else 0)
            max_recall += I / raw_label.sum()
            max_mIoU += I / U
            
            # Pseudo label: Weighted sum of the attention maps with similarity scores between slots
            avg_a_map = img_a_map[i, -1, :]
            avg_a_map = cm_a.unsqueeze(1) * avg_a_map
            avg_a_map = avg_a_map.sum(dim=0)
            avg_a_map = avg_a_map.reshape(1, feat_map_size, feat_map_size)
            avg_a_map = transforms.functional.resize(avg_a_map, list(raw_label.shape)).squeeze().cpu().numpy()
            avg_a_map = ((avg_a_map - avg_a_map.min()) / (avg_a_map.max() - avg_a_map.min() + 1e-9))

            hard_avg_pred = (avg_a_map >= args.pseudo_threshold)
            I, U = compute_mask_IU(hard_avg_pred, raw_label)
            cum_avg_I += I
            cum_avg_U += U
            
            avg_precision += (I / hard_avg_pred.sum() if I != 0 else 0)
            avg_recall += I / raw_label.sum()
            avg_mIoU += I / U
            
            t.set_postfix({
                "max | avg":" %.3f%%  |  %.3f%% " \
                    % (100*(cum_max_I/cum_max_U), 100*(cum_avg_I/cum_avg_U))
            })

        val_dict = {
            'max_cIoU': 100 * (cum_max_I/cum_max_U),
            'max_mIoU': max_mIoU * (100/len(dataset)),
            'max_precision': max_precision * (100/len(dataset)),
            'max_recall': max_recall * (100/len(dataset)),
            'avg_cIoU': 100 * (cum_avg_I/cum_avg_U),
            'avg_mIoU': avg_mIoU * (100/len(dataset)),
            'avg_precision': avg_precision * (100/len(dataset)),
            'avg_recall': avg_recall * (100/len(dataset))
        }
        
        del img_a_map, avg_a_map, a_map
        del hard_pred, img_a_map_for_slot, hard_avg_pred
        
        return losses.avg, losses_dict, val_dict

def update_best_score(new_score, old_score, is_higher_better):
    if not old_score:
        score, updated = new_score, True
    else:
        if is_higher_better:
            score = max(new_score, old_score)
            updated = new_score > old_score
        else:
            score = min(new_score, old_score)
            updated = new_score < old_score
    return score, updated
    

def warmup(model, epoch, args, multi_gpu):
    if args.img_finetune and args.txt_finetune:
        warm = epoch >= args.warm_epoch
        if args.warm_img:
            for idx, param in enumerate((model.module if multi_gpu else model).encoders.img_enc.img_backbone.parameters()):
                param.requires_grad = warm
        if args.warm_txt:
            for idx, param in enumerate((model.module if multi_gpu else model).encoders.txt_enc.bert.parameters()):
                param.requires_grad = warm

def save_ckpt(state, is_best, filename='ckpt.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        print('Updating the best model checkpoint: {}'.format(prefix + 'model_best.pth.tar'))

def main():
    args = verify_input_args(parser.parse_args())
    set_seed(args.seed)
    if not args.no_wandb:
        wandb.init(project='weak_ref_seg', name = args.remark, group=args.wandb_group, entity='ise', config=args) 
        wandb.config.update(args) 
    log_dir = osp.join('./logs', args.remark)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)

    # Dataloaders
    trn_loader = data.get_train_loader(args)
    test_loader = data.get_test_loader(args)

    # Construct the model
    # NOTE CMA model 
    model = CrossModalAttentionRecon(ImageTextEncodersRecon(args), args.embed_dim, args)
            
    if torch.cuda.is_available():
        if args.multi_gpu:
            model = nn.DataParallel(model, output_device=1)
        if args.sync_bn:
            model = convert_model(model)
        model = model.cuda()
        cudnn.benchmark = True
        
    wandb.watch(models=model, log_freq=1000, log='gradients') if not args.no_wandb else None
            
    # Loss and optimizer
    recon_criterion = nn.MSELoss()
    recon_weight = args.recon_weight
    
    criterion = (CMA_Loss_Fast if args.fast_batch else CMA_Loss)(
        margin=args.margin, 
        criterion=args.cma_criterion, 
        mining=args.cma_mining, 
        detach_target=args.cma_detach_target,
        detach_img_target=args.cma_detach_img_target,
        i_t_loss=None,
        i_t_weight=args.i_t_weight,
        temperature=args.info_temperature,
        cm_i_weight=args.cm_i_weight,
        size_p_loss=None,
        size_p_weight=args.size_p_weight,
        )

    val_criterion = CMA_Loss(
        margin=args.margin, 
        criterion=args.cma_criterion, 
        mining=args.cma_mining, 
        detach_target=args.cma_detach_target,
        detach_img_target=args.cma_detach_img_target,
        i_t_weight=args.i_t_weight,
        temperature=args.info_temperature,
        cm_i_weight=args.cm_i_weight,
        size_p_weight=args.size_p_weight,
        )
        
    module = model.module if args.multi_gpu else model
    param_groups = [
        {'params': module.cma.parameters(), 'lr': args.lr},
        {'params': module.decoder.parameters(), 'lr': args.lr},
        {'params': list(set(module.encoders.img_enc.parameters()).difference(set(module.encoders.img_enc.img_backbone.parameters()))), 
         'lr': args.lr * args.img_spm_lr_scale},
        {'params': module.encoders.img_enc.img_backbone.parameters(), 'lr': args.lr * args.img_lr_scale},
        {'params': list(set(module.encoders.txt_enc.parameters())), 'lr': args.lr * args.txt_lr_scale}
    ]
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamp':
        from adamp import AdamP
        optimizer = AdamP(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    
    if args.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trn_loader)*args.num_epochs)
    elif args.lr_scheduler == 'multi_step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma = args.lr_step_gamma)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma = args.lr_step_gamma)
        
    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    total_iter = 0
    best_loss = 10000
    best_iou = 0

    # Pre-compute bertemb
    bertemb_dict = None
    if args.pre_bertemb:
        bertemb_dict = torch.load(osp.join(osp.dirname(args.data_path), osp.basename(args.data_path)+'_bertemb.pt'))
    
    for epoch in range(args.num_epochs):
        #warm up training data
        warmup(model, epoch, args, args.multi_gpu)
        
        recon_weight = args.recon_weight if epoch >= args.wo_recon_epoch else .0
        # train for one epoch
        loss, losses_dict, stat_dict, total_iter = train(
            epoch, total_iter, trn_loader, model, criterion, recon_criterion, recon_weight, optimizer, scaler, 
            epoch < args.recon_warm_epoch, 
            args, 
            scheduler=lr_scheduler if args.lr_scheduler == 'cosine' else None, 
            bertemb_dict = bertemb_dict
        )
        
        # Compute validation loss
        val_loss, val_losses_dict, val_dict = \
            validation(epoch, test_loader, model, val_criterion, recon_criterion, recon_weight, args)
        print(val_dict)
        
        if not args.no_wandb:
            wandb.log({"epoch": epoch}, step=total_iter)
            
            wandb.log({"loss": loss}, step=total_iter)
            for key, val in losses_dict.items():
                wandb.log({key: val.avg}, step=total_iter)
            for key, val in stat_dict.items():
                wandb.log({key: val.avg}, step=total_iter)
            wandb.log({"LR" : optimizer.param_groups[0]['lr']}, step=total_iter)
            
            wandb.log({"val loss": val_loss}, step=total_iter)
            for key, val in val_losses_dict.items():
                wandb.log({"val "+key: val.avg}, step=total_iter)
            for key, val in val_dict.items():
                wandb.log({key: val}, step=total_iter)
                
        # evaluate on validation set
        with torch.no_grad():
            # remember best rsum and save ckpt
            # best_loss, updated = update_best_score(loss, best_loss, is_higher_better=False)
            best_iou, best_updated = update_best_score(val_dict['avg_mIoU'], best_iou, is_higher_better=True)
            save_ckpt({
                'args': args,
                'epoch': epoch,
                'iou': val_dict['avg_mIoU'],
                'loss': loss,
                'model': model.state_dict(),
            }, best_updated, prefix=log_dir + '/')
        
        # adjust learning rate if rsum stagnates
        if args.lr_scheduler != 'cosine':
            lr_scheduler.step()

if __name__ == '__main__':
    main()