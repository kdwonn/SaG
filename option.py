import os, sys, pdb
import argparse

parser = argparse.ArgumentParser(description='Parameters for training PVSE')

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# Names, paths, logging, etc
parser.add_argument('--data_name', default='coco', choices=('coco','phrasecut'), help='Dataset name (coco|cub)')
parser.add_argument('--data_path', default='./data/coco/Gref_480_batch', help='path to datasets')
parser.add_argument('--data_split', default='train', help='path to datasets')
parser.add_argument('--vocab_path', default=CUR_DIR+'/vocab/', help='Path to saved vocabulary pickle files')
parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log')
parser.add_argument('--log_dir', default=CUR_DIR+'/logs/', help='Path to save result logs') 

# Data parameters
parser.add_argument('--word_dim', default=300, type=int, help='Dimensionality of the word embedding')
parser.add_argument('--workers', default=16, type=int, help='Number of data loader workers')
parser.add_argument('--crop_size', default=224, type=int, help='Size of an image crop as the CNN input')

# Model parameters
parser.add_argument('--img_backbone', default='vit_small_patch16_384', help='The img encoder backbone')
parser.add_argument('--embed_dim', default=1024, type=int, help='Dimensionality of the joint embedding')
parser.add_argument('--margin', default=0.1, type=float, help='Rank loss margin')
parser.add_argument('--dropout', default=0.0, type=float, help='Dropout rate')

# Attention parameters
parser.add_argument('--img_num_embeds', default=1, type=int, help='Number of img embeddings')
parser.add_argument('--txt_num_embeds', default=1, type=int, help='Number of txt embeddings')

# Training / optimizer setting
parser.add_argument('--img_finetune', action='store_true', help='Fine-tune img backbone')
parser.add_argument('--txt_finetune', action='store_true', help='Fine-tune the word embedding')
parser.add_argument('--num_epochs', default=30, type=int, help='Number of training epochs')
parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch')
parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay (l2 norm) for optimizer')
parser.add_argument('--lr', default=.0002, type=float, help='Initial learning rate')
parser.add_argument('--ckpt', default='', type=str, metavar='PATH', help='path to latest ckpt (default: none)')
parser.add_argument('--eval_on_gpu', action='store_true', help='Evaluate on GPU (default: CPU)')

# customized settings
parser.add_argument('--warm_epoch', default=30, type=int, help='warm up epochs')
parser.add_argument('--remark', type=str)
parser.add_argument('--wandb_group', type=str, default='pseudo_labelling')
parser.add_argument('--no_wandb', action='store_true')
parser.add_argument('--lr_scheduler', type=str, default='cosine')
parser.add_argument('--lr_milestones', nargs='+', type=int ,help='step value used in step scheduler')
parser.add_argument('--lr_step_gamma', type=float, default=0.5, help='step value used in step scheduler')
parser.add_argument('--lr_step_size', type=int, help='step value used in step scheduler')
parser.add_argument('--warm_txt', action='store_true')
parser.add_argument('--warm_img', action='store_true')
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument('--sync_bn', action='store_true')
parser.add_argument('--fast_batch', action='store_true')
parser.add_argument('--num_texts', default=0, type=int)
parser.add_argument('--semi_hard_triplet', action='store_true')
parser.add_argument('--img_spm_lr_scale', type=float, default=1.0)
parser.add_argument('--txt_lr_scale', type=float, default=1.0)
parser.add_argument('--img_lr_scale', type=float, default=1.0)
parser.add_argument('--optimizer', type=str)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--lr_warmup_iter', type=int, default=0)
parser.add_argument('--bn_eval', action='store_true')

# Aggregator options
parser.add_argument('--agg_query_self_attns', type=int, default=0)
parser.add_argument('--agg_self_per_cross_attn', type=int, default=0)
parser.add_argument('--agg_self_before_cross_attn', type=int, default=0)
parser.add_argument('--agg_depth', type=int, default=1)
parser.add_argument('--agg_cross_head', type=int, default=4)
parser.add_argument('--agg_cross_dim', type=int, default=64)
parser.add_argument('--agg_residual', action='store_true')

parser.add_argument('--agg_latent_head', type=int, default=8)
parser.add_argument('--agg_latent_dim', type=int, default=32)

parser.add_argument('--agg_last_fc', action='store_true')
parser.add_argument('--agg_input_dim', type=int, default=1024)
parser.add_argument('--agg_query_dim', type=int, default=1024)
parser.add_argument('--agg_pre_norm', action='store_true')
parser.add_argument('--agg_post_norm', action='store_true')
parser.add_argument('--agg_activation', type=str, default='geglu')
parser.add_argument('--agg_last_ln', action='store_true')
parser.add_argument('--agg_weight_sharing', action='store_true')
parser.add_argument('--agg_ff_mult', type=float, default=2)
parser.add_argument('--agg_xavier_init', action='store_true')
parser.add_argument('--agg_more_dropout', action='store_true')
parser.add_argument('--agg_thin_ff', action='store_true')
parser.add_argument('--agg_first_order', action='store_true')
parser.add_argument('--agg_pos_enc', type=str)
parser.add_argument('--agg_gru', action='store_true')

# Options for attention mechanisms
parser.add_argument('--agg_cross_attn_type', default='transformer', choices=('slot', 'transformer'))
parser.add_argument('--agg_gumbel_attn', action='store_true')
parser.add_argument('--agg_gumbel_last', action='store_true')

# Options for query 
parser.add_argument('--agg_query_slot', action='store_true') # Legacy option
parser.add_argument('--agg_query_type', default='query', choices=('query', 'random', 'entity'))
parser.add_argument('--cascade_factor', type=int)
parser.add_argument("--agg_var_scaling", type=float, default=0)
parser.add_argument('--decoder_normalizer', type=str, default='softmax')
parser.add_argument("--recon_decoder", type=str, default='mlp')
parser.add_argument("--decoder_self_attn", action='store_true')
parser.add_argument("--decoder_pos_enc", default='learned', choices=('learned', 'sine'))
parser.add_argument("--slot_cond", action='store_true')

# Options for encoder
parser.add_argument('--txt_pooling', default='max', choices=('cls','max', 'avg'))
parser.add_argument('--txt_l2', action='store_true')
parser.add_argument('--img_res_pool', default='avg', choices=('avg','max'))
parser.add_argument('--text_no_dropout', action='store_true')

# Options for pseudo labelling
parser.add_argument('--pseudo_threshold', default=0.5, type=float)

# Options for cross-modal attention
parser.add_argument('--cma_heads', default=1, type=int)
parser.add_argument('--cma_head_dim', default=512, type=int)
parser.add_argument('--cma_criterion', type=str)
parser.add_argument('--cma_mining', type=str)
parser.add_argument('--cma_detach_target', action='store_true')
parser.add_argument('--cma_detach_img_target', action='store_true')
parser.add_argument('--cma_self_attn', action='store_true')
parser.add_argument('--use_aug', action='store_true')
parser.add_argument('--cma_last_fc', action='store_true')
parser.add_argument('--cma_qk_norm', action='store_true')
parser.add_argument('--cma_last_mlp', action='store_true')

# Options for loss configuration
parser.add_argument('--i_t_weight', default=0.0, type=float)
parser.add_argument('--info_temperature', default=1.0, type=float)
parser.add_argument('--gumbel_tau', default=1.0, type=float)
parser.add_argument('--cm_i_weight', default=1.0, type=float)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--recon_weight", type=float, default=1.0)
parser.add_argument("--num_layers", type=int, default=12)
parser.add_argument("--recon_warm_epoch", type=int, default=0)
parser.add_argument("--wo_recon_epoch", type=int, default=0)
parser.add_argument("--amap_save", action="store_true")
parser.add_argument("--agg_1x1_mlp", action="store_true")
parser.add_argument("--info_txt_l2", action="store_true")
parser.add_argument("--pre_bertemb", action="store_true")
parser.add_argument("--eval_epoch", default=1, type=int)

# Options for size-penalty attention
parser.add_argument('--size_p_weight', default=0, type=float)
parser.add_argument('--size_gamma', default=0.01, type=float)
parser.add_argument('--size_penalty', default=5, type=float)

parser.add_argument("--save_head_map", action='store_true')

def verify_input_args(args):
    # Process input arguments
    assert not(args.agg_gumbel_attn and args.agg_gumbel_last)
    if args.agg_query_slot:
        args.agg_query_type = 'sampling'
    if args.agg_query_type == 'entity':
        assert args.cascade_factor is not None
    return args
