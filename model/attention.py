from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def cache_fn2(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    index = y_soft.topk(1, dim)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret

def gumbel_softmax(logits, dim, tau= 1):
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = logits / tau + gumbels  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    index = y_soft.topk(1, dim)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)
    
class SandwichNorm(nn.Module):
    def __init__(self, indim, outdim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(indim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None
        self.outnorm = nn.LayerNorm(outdim)

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.outnorm(self.fn(x, **kwargs))
    
class PostNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = 'geglu', more_dropout = False, xavier_init = False):
        super().__init__()
        act_in_dim = int(dim * mult)
        act_out_dim = act_in_dim
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'geglu':
            act_in_dim *= 2
            self.activation = GEGLU()
        else:
            raise NotImplementedError("Invalid activation function")
            
        self.net = nn.Sequential(
            nn.Linear(dim, act_in_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(act_out_dim, dim),
            nn.Dropout(dropout) if more_dropout else nn.Identity()
        )
        
        if xavier_init:
            self._reset_parameter()
    
    def _reset_parameter(self):
        def fn(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        self.net.apply(fn)

    def forward(self, x):
        return self.net(x)

class ThinFeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = 'geglu', more_dropout = False, xavier_init = False):
        super().__init__()
        act_in_dim = dim * mult
        act_out_dim = act_in_dim
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError("Invalid activation function")
            
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            self.activation,
            nn.Dropout(dropout)
        )
        
        if xavier_init:
            self._reset_parameter()
    
    def _reset_parameter(self):
        def fn(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        self.net.apply(fn)

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self, 
        query_dim, 
        context_dim = None, 
        heads = 8, dim_head = 64, 
        dropout = 0., 
        attn_type = 'transformer', 
        more_dropout = False, 
        xavier_init = False,
        qk_norm = False,
        gumbel = False,
        gumbel_tau = 1
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        
        self.qk_norm = qk_norm
        self.scale = nn.Parameter(torch.zeros(1, heads, 1, 1)) if self.qk_norm else dim_head ** -0.5
        self.max_scale = 20
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.attn_holder = nn.Identity()
        self.gumbel = gumbel
        
        self.attn_type = attn_type
        self.attn_matrix_dropout = nn.Dropout(dropout) if more_dropout else nn.Identity()
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        if xavier_init:
            self._reset_parameter()
        
    def _reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        
    def norm_attn(self, sim, is_gumbel = False, gumbel_tau=1):
        if self.gumbel or is_gumbel:
            norm_fn = (lambda logits, dim : gumbel_softmax(logits, dim=dim, tau=gumbel_tau)) if self.training else hard_softmax
        else:
            norm_fn = F.softmax
            
        if self.attn_type == 'transformer':
            attn = norm_fn(sim, dim=-1)
            attn = self.attn_holder(attn)
        elif self.attn_type == 'slot':
            attn = norm_fn(sim, dim=1)
            attn = self.attn_holder(attn)
            # attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-7)
        else:
            raise NotImplementedError
        
        return attn

    def forward(self, x, context = None, mask = None, k_pos = None, q_pos = None, is_gumbel = False, gumbel_tau=1):
        h = self.heads

        q = self.to_q(x if q_pos is None else x + q_pos)
        context = default(context, x)
        k = self.to_k(context if k_pos is None else context + k_pos)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        
        if self.qk_norm:
            v = F.normalize(v, dim=-1, p=2)
            sim = einsum('b i d, b j d -> b i j', F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2))
            sim = rearrange(sim, '(b h) i j -> b h i j', h = h) * (self.scale.sigmoid() * self.max_scale)
            sim = rearrange(sim, 'b h i j -> (b h) i j')
        else:
            sim = einsum('b i d, b j d -> b i j', q, k)
            sim = sim * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(mask, max_neg_value)
        # attention, what we cannot get enough of
        attn = self.norm_attn(sim, is_gumbel = is_gumbel, gumbel_tau=gumbel_tau)
        out_attn = attn
        if self.attn_type == 'slot':
            if exists(mask):
                attn = attn.masked_fill(mask, 0)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-7)
        
        if torch.isnan(attn).any():
            import pdb; pdb.set_trace()
            
        attn = self.attn_matrix_dropout(attn)
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out), out_attn


class TransformerLayer(nn.Module):
    def __init__(
        self, 
        query_dim,
        ff_dim, 
        context_dim = None, 
        heads = 8, 
        dim_head = 64, 
        dropout = 0., 
        attn_type = 'transformer', 
        more_dropout = False, 
        xavier_init = False,
        ff_mult = 4, 
        ff_dropout = 0., 
        ff_activation = 'geglu', 
        ff_more_dropout = False, 
        ff_xavier_init = False,
        pre_norm=True,
        thin_ff=False,
        last_norm=False,
        last_fc=False,
        qk_norm=False
    ):
        super().__init__()
        
        prenorm = PreNorm if pre_norm else lambda dim, fn, context_dim=None: fn
        postnorm = PostNorm if not pre_norm else nn.Identity
        feedforward = ThinFeedForward if thin_ff else FeedForward
        
        attn = Attention(
            query_dim, 
            context_dim=context_dim,
            heads=heads, 
            dim_head=dim_head, 
            dropout=dropout, 
            attn_type=attn_type, 
            more_dropout=more_dropout, 
            xavier_init=xavier_init,
            qk_norm=qk_norm
        )
        self.attn = prenorm(query_dim, attn, context_dim=context_dim)
        self.attn_postnorm = postnorm(query_dim)
        
        ff = feedforward(
            ff_dim, 
            dropout = ff_dropout, 
            activation = ff_activation, 
            mult=ff_mult, 
            more_dropout = ff_more_dropout, 
            xavier_init = ff_xavier_init
        )
        self.ff = prenorm(ff_dim, ff)
        self.ff_postnorm = postnorm(ff_dim)
        
        self.lastnorm = postnorm(query_dim) if last_norm else nn.Identity() 
        self.lastfc = nn.Linear(query_dim, query_dim) if last_fc else nn.Identity()
    
    def forward(self, x, context=None, mask=None, k_pos=None, q_pos=None):
        out, attn = self.attn(x, context=context, mask=mask, k_pos=k_pos, q_pos = q_pos)
        x = x + out
        x = self.attn_postnorm(x)
        x = self.ff(x) + x
        x = self.ff_postnorm(x)
        x = self.lastfc(self.lastnorm(x))
        return x
    
class Queries(nn.Module):
    def __init__(self, query_type, num, dim, cascade_factor=None, var_scale=0):
        super().__init__()
        self.query_type = query_type
        self.num = num
        self.dim = dim
        self.cascade_factor = cascade_factor
        self.scale = 1/(self.dim*var_scale) if var_scale != 0 else 1
        if self.query_type == 'query':
            self.params = nn.Parameter(torch.randn(self.num, self.dim))
        elif self.query_type == 'random':
            self.slots_mu = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 1, dim)), gain=nn.init.calculate_gain("linear"))
            self.slots_log_sigma = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 1, dim)), gain=nn.init.calculate_gain("linear"))
        elif self.query_type == 'entity':
            num_dist = self.num // self.cascade_factor
            self.slots_mu = nn.Parameter(nn.init.xavier_uniform_(torch.randn(1, num_dist, dim), gain=nn.init.calculate_gain("linear")))
            self.slots_log_sigma = nn.init.xavier_uniform_(torch.randn(1, num_dist, dim), gain=nn.init.calculate_gain("linear"))
            # making l2-norm of sigma to be around 1 (Confirmed in debugging)
            self.slots_log_sigma = nn.Parameter(self.slots_log_sigma + torch.log(torch.tensor(self.scale)))
        else:
            raise NotImplementedError
        
    def forward(self, b, condition=None):
        if self.query_type == 'query':
            ret = repeat(self.params, 'n d -> b n d', b = b)
        elif self.query_type == 'random':
            slots_init = torch.randn((b, self.num, self.dim)).cuda()
            ret = self.slots_mu + self.slots_log_sigma.exp() * slots_init
        elif self.query_type == 'entity':
            slots_init = torch.randn((b, self.num, self.dim)).cuda()
            ret = repeat(self.slots_mu, 'b n d -> b (n r) d', r=self.cascade_factor) + \
                repeat(self.slots_log_sigma, 'b n d -> b (n r) d', r=self.cascade_factor).mul(0.5).exp() * slots_init
        else:
            raise NotImplementedError

        if condition is not None:
            ret = ret + condition

        return ret 