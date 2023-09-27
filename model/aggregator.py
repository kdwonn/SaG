import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from model.pos_encoding import build_position_encoding
from model.attention import *

class Aggregator(nn.Module):
    def __init__(
        self,
        *,
        depth,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        self_per_cross_attn = 1,
        self_before_cross_attn = 0,
        query_self_attn = 1,
        pos_enc_type = 'none',
        last_fc = True,
        norm_pre = True,
        norm_post = True, 
        activation = 'geglu',
        last_ln = False,
        ff_mult = 4,
        cross_attn_type = 'transformer',
        more_dropout = False,
        xavier_init = False,
        thin_ff = False,
        query_type = 'query',
        first_order=False,
        gumbel_attn=False,
        last_gumbel=False,
        gumbel_tau = 1,
        cascade_factor = None,
        var_scaling=False,
        gru=False
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
            depth: Depth of net.
            input_channels: Number of channels for each token of the input.
            input_axis: Number of axes for input data (2 for images, 3 for video)
            num_latents: Number of latents, or induced set points, or centroids.
                Different papers giving it different names.
            latent_dim: Latent dimension.
            cross_heads: Number of heads for cross attention. Paper said 1.
            latent_heads: Number of heads for latent self attention, 8.
            cross_dim_head: Number of dimensions per cross attention head.
            latent_dim_head: Number of dimensions per latent self attention head.
            num_classes: Output number of classes.
            attn_dropout: Attention dropout
            ff_dropout: Feedforward dropout
            weight_tie_layers: Whether to weight tie layers (optional).
            self_per_cross_attn: Number of self attention blocks per cross attn.
        """
        super().__init__()
        self.input_axis = input_axis
        self.num_classes = num_classes

        input_dim = input_channels
        self.input_dim = input_channels
        self.pos_enc = build_position_encoding(input_dim, pos_enc_type, self.input_axis)

        self.num_latents = num_latents
        self.query_type = query_type
        self.latent_dim = latent_dim
        self.first_order = first_order
        self.last_gumbel = last_gumbel
        self.gumbel_tau = gumbel_tau
        self.depth = depth
        self.cross_heads = cross_heads
        
        self.queries = Queries(self.query_type, self.num_latents, self.latent_dim, cascade_factor=cascade_factor, var_scale=var_scaling)
        
        assert (norm_pre or norm_post)
        prenorm = PreNorm if norm_pre else lambda dim, fn, context_dim=None: fn
        postnorm = PostNorm if norm_post else nn.Identity
        ff = ThinFeedForward if thin_ff else FeedForward
        
        cross_attn_config = {'heads': cross_heads, 'dim_head': cross_dim_head, 'dropout': attn_dropout, 'attn_type': cross_attn_type, 'more_dropout': more_dropout, 'xavier_init': xavier_init, 'gumbel': gumbel_attn}
        cross_ff_config = {'dropout': ff_dropout, 'activation': activation, 'mult': ff_mult, 'more_dropout': more_dropout, 'xavier_init': xavier_init}
        latent_attn_config = {'heads': latent_heads, 'dim_head': latent_dim_head, 'dropout': attn_dropout, 'more_dropout': more_dropout, 'xavier_init': xavier_init}
        latent_ff_config = {'dropout': ff_dropout, 'activation': activation, 'mult': ff_mult, 'more_dropout': more_dropout, 'xavier_init': xavier_init}
        
        # * decoder cross attention layers
        get_cross_attn = lambda: prenorm(latent_dim, Attention(latent_dim, input_dim, **cross_attn_config), context_dim = input_dim)
        get_cross_ff = lambda: prenorm(latent_dim, ff(latent_dim, **cross_ff_config))
        get_cross_postnorm = lambda: postnorm(latent_dim)
        
        # * self attention of queries (first self attention layer of decoder)
        get_latent_attn = lambda: prenorm(latent_dim, Attention(latent_dim, **latent_attn_config))
        get_latent_ff = lambda: prenorm(latent_dim, ff(latent_dim, **latent_ff_config))
        get_latent_postnorm = lambda: postnorm(latent_dim)
        
        # * encoder layers
        get_pre_self_attn = lambda: prenorm(input_dim, Attention(input_dim, **latent_attn_config))
        get_pre_self_ff = lambda: prenorm(input_dim, ff(input_dim, **latent_ff_config))
        get_pre_self_postnorm = lambda: postnorm(input_dim)
        
        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff, get_pre_self_attn, get_pre_self_ff = map(cache_fn, \
            (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff, get_pre_self_attn, get_pre_self_ff)
        )
                
        """ Overall architecture
            prediction = layers(query_self_attns(queries), pre_self_attns(data))
            Each layer consists of single self attention and single cross attention.
        """
        # self attention before going into decoder, coresponding to the DETR encoder
        self.pre_self_attns = nn.ModuleList([])
        for _ in range(self_before_cross_attn):
            self.pre_self_attns.append(nn.ModuleList([
                get_pre_self_attn(**{'_cache': False}),
                get_pre_self_postnorm(),
                get_pre_self_ff(**{'_cache': False}),
                get_pre_self_postnorm()
            ]))
        
        # self attention for decoder query (not necessary but following DETR's choice)
        self.query_pre_self_attns = nn.ModuleList([])
        for _ in range(query_self_attn):
            self.query_pre_self_attns.append(nn.ModuleList([
                get_latent_attn(**{'_cache': False}),
                get_latent_postnorm(),
                get_latent_ff(**{'_cache': False}),
                get_latent_postnorm()
            ]))
            
        self.agg_blocks = nn.ModuleList([])
        for i in range(depth):
            should_cache = i >= 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}
            
            # self attention after cross attention.
            self_attns = nn.ModuleList([])
            for _ in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_postnorm(),
                    get_latent_ff(**cache_args),
                    get_latent_postnorm()
                ]))
            
            self.agg_blocks.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_postnorm(),
                get_cross_ff(**cache_args),
                get_cross_postnorm(),
                self_attns
            ]))

        # Function used to update slot in each iteration: GRU(original slot attention) or residual sum
        if gru:
            self.gru = torch.nn.GRUCell(self.latent_dim, self.latent_dim)
            self.slot_update_fn = lambda update, prev: \
                self.gru(update.reshape(-1, self.latent_dim), prev.reshape(-1, self.latent_dim)).reshape(-1, self.num_latents, self.latent_dim)
        else:
            self.slot_update_fn = lambda update, prev: update + prev

        # Last FC layer
        if not last_fc:
            assert latent_dim == self.num_classes
        self.last_layer = nn.Sequential(
            nn.LayerNorm(latent_dim) if last_ln and not postnorm else nn.Identity(),
            nn.Linear(latent_dim, num_classes) if last_fc else nn.Identity()
        )
        
        self.encoder_output_holder = nn.Identity()
        self.decoder_output_holder = nn.Identity()

    def forward(self, data, mask = None, txt_emb = None):
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # concat to channels of data and flatten axis
        pos = self.pos_enc(data)
        data = rearrange(data, 'b ... d -> b (...) d')
        x = self.queries(b, txt_emb).type_as(data)
        
        for enc_block in self.pre_self_attns:
            pre_self_attn, pn1, self_ff, pn2 = enc_block
            data = pre_self_attn(data, mask = mask, q_pos = pos, k_pos = pos) + data
            data = pn1(data)
            data = self_ff(data) + data
            data = pn2(data)  
        data = self.encoder_output_holder(data)
        
        for query_self_attn, pn1, self_ff, pn2 in self.query_pre_self_attns:
            x = query_self_attn(x) + x
            x = pn1(x)
            x = self_ff(x) + x
            x = pn2(x)

        data = data if pos is None else data + pos
        for i, (cross_attn, pn1, cross_ff, pn2, self_attns) in enumerate(self.agg_blocks):
            is_gumbel = ((i == len(self.agg_blocks)-1) and (self.last_gumbel))
            if i == len(self.agg_blocks) - 1 and self.first_order:
                x = x.detach()
            out, attn = cross_attn(x, context = data, mask = mask, is_gumbel=is_gumbel, gumbel_tau=self.gumbel_tau)
            x = self.slot_update_fn(out, x)
            x = pn1(x)
            x = cross_ff(x) + x
            x = pn2(x)

            for self_attn, pn1, self_ff, pn2 in self_attns:
                x = rearrange(x, 'b (c k) d -> (b c) k d', k=self.queries.cascade_factor)
                out, attn = self_attn(x, mask = mask)
                x = out + x 
                x = pn1(x)
                x = self_ff(x) + x
                x = pn2(x)
                x = rearrange(x, '(b c) k d -> b (c k) d', c=self.num_latents//self.queries.cascade_factor)
                
        attn = rearrange(attn, '(bs h) n d -> bs h n d', h=self.cross_heads)
        attn = reduce(attn, 'bs h n d -> bs n d', 'mean')
        x = self.decoder_output_holder(x)
        
        return self.last_layer(x), attn
