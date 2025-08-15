import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, orthogonal_
#from .gnn_module import SmoothConvEncoder, SmoothConvDecoder, index_points
#from torch_scatter import scatter
# helpers



class SpatialTemporalEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            # Rearrange('b c n -> b n c'),
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        if depth > 4:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=[32, 16, 8, 8] +
                                                                             [1] * (depth - 4),
                                                     attention_init='orthogonal')
        else:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=[32] + [16]*(depth-2) + [1],
                                                     attention_init='orthogonal')

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, n, t(*c)]
                input_pos,  # [b, n, 2]
                ):
        x = torch.cat((x, input_pos), dim=-1)
        x = self.to_embedding(x)
        x = self.s_transformer.forward(x, input_pos)
        x = self.project_to_latent(x)

        return x




class TransformerCatNoCls(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 use_ln=False,
                 scale=16,     # can be list, or an int
                 dropout=0.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 attention_init='orthogonal',
                 init_gain=None,
                 use_relu=False,
                 cat_pos=False,
                 ):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']

        if isinstance(scale, int):
            scale = [scale] * depth
        assert len(scale) == depth

        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        self.use_ln = use_ln

        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList([
                    PreNorm(dim, StandardAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim,  FeedForward(dim, mlp_dim, dropout=dropout)
                                  if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout))]),
                )
        else:
            for d in range(depth):
                if scale[d] != -1 or not cat_pos:
                    attn_module = LinearAttention(dim, attn_type,
                                                   heads=heads, dim_head=dim_head, dropout=dropout,
                                                   relative_emb=True, scale=scale[d],
                                                   relative_emb_dim=relative_emb_dim,
                                                   min_freq=min_freq,
                                                   init_method=attention_init,
                                                   init_gain=init_gain
                                                   )
                else:
                    attn_module = LinearAttention(dim, attn_type,
                                                  heads=heads, dim_head=dim_head, dropout=dropout,
                                                  cat_pos=True,
                                                  pos_dim=relative_emb_dim,
                                                  relative_emb=False,
                                                  init_method=attention_init,
                                                  init_gain=init_gain
                                                  )
                if not use_ln:
                    self.layers.append(
                        nn.ModuleList([
                                        attn_module,
                                        FeedForward(dim, mlp_dim, dropout=dropout)
                                        if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout)
                        ]),
                        )
                else:
                    self.layers.append(
                        nn.ModuleList([
                            nn.LayerNorm(dim),
                            attn_module,
                            nn.LayerNorm(dim),
                            FeedForward(dim, mlp_dim, dropout=dropout)
                            if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout),
                        ]),
                    )

    def forward(self, x, pos_embedding):
        # x in [b n c], pos_embedding in [b n 2]
        b, n, c = x.shape

        for layer_no, attn_layer in enumerate(self.layers):
            if not self.use_ln:
                [attn, ffn] = attn_layer

                x = attn(x, pos_embedding) + x
                x = ffn(x) + x
            else:
                [ln1, attn, ln2, ffn] = attn_layer
                x = ln1(x)
                x = attn(x, pos_embedding) + x
                x = ln2(x)
                x = ffn(x) + x
        return x





class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim*2),
            GeGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ReLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def masked_instance_norm(x, mask, eps = 1e-5):
    """
    x of shape: [batch_size (N), num_objects (L), features(C)]
    mask of shape: [batch_size (N), num_objects (L), 1]
    """
    mask = mask.float()  # (N,L,1)
    mean = (torch.sum(x * mask, 1) / torch.sum(mask, 1))   # (N,C)
    mean = mean.detach()
    var_term = ((x - mean.unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,C)
    var = (torch.sum(var_term, 1) / torch.sum(mask, 1))  #(N,C)
    var = var.detach()
    mean_reshaped = mean.unsqueeze(1).expand_as(x)  # (N, L, C)
    var_reshaped = var.unsqueeze(1).expand_as(x)    # (N, L, C)
    ins_norm = (x - mean_reshaped) / torch.sqrt(var_reshaped + eps)   # (N, L, C)
    return ins_norm


class GeGELU(nn.Module):
    """https: // paperswithcode.com / method / geglu"""
    def __init__(self):
        super().__init__()
        self.fn = nn.GELU()

    def forward(self, x):
        c = x.shape[-1]  # channel last arrangement
        return self.fn(x[..., :int(c//2)]) * x[..., int(c//2):]



class PointWiseDecoder2D(nn.Module):
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 out_steps,  # 10
                 propagator_depth,
                 scale=8,
                 dropout=0.,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels//2, 'galerkin', 4,
                                                self.latent_channels//2, self.latent_channels//2,
                                                relative_emb=True,
                                                scale=16.,
                                                relative_emb_dim=2,
                                                min_freq=1/64)

        self.expand_feat = nn.Linear(self.latent_channels//2, self.latent_channels)

        self.propagator = nn.ModuleList([
               nn.ModuleList([nn.LayerNorm(self.latent_channels),
               nn.Sequential(
                    nn.Linear(self.latent_channels + 2, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False))])
            for _ in range(propagator_depth)])

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels * self.out_steps, bias=True))

    def propagate(self, z, pos):
        for layer in self.propagator:
            norm_fn, ffn = layer
            z = ffn(torch.cat((norm_fn(z), pos), dim=-1)) + z
        return z

    def decode(self, z):
        z = self.to_out(z)
        return z

    def get_embedding(self,
                      z,  # [b, n c]
                      propagate_pos,  # [b, n, 2]
                      input_pos
                      ):
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.expand_feat(z)
        return z

    def forward(self,
                z,              # [b, n, c]
                propagate_pos   # [b, n, 2]
                ):
        z = self.propagate(z, propagate_pos)
        u = self.decode(z)
        u = rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps)
        return u, z                # [b c_out t n], [b c_latent t n]

    def rollout(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 2]
                forward_steps,
                input_pos):
        history = []
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.expand_feat(z)

        # forward the dynamics in the latent space
        for step in range(forward_steps//self.out_steps):
            z = self.propagate(z, propagate_pos)
            u = self.decode(z)
            history.append(rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps))
        history = torch.cat(history, dim=-2)  # concatenate in temporal dimension,  # [b, length_of_history*c, n]

        #reshape
        history = rearrange(history, 'b (t c) n -> b n t c', c=self.out_channels, t=self.out_steps)
        return history 


class CrossFormer(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,
                 heads,
                 dim_head,
                 mlp_dim,
                 residual=True,
                 use_ffn=True,
                 use_ln=False,
                 relative_emb=False,
                 scale=1.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 dropout=0.,
                 cat_pos=False,
                 ):
        super().__init__()

        self.cross_attn_module = CrossLinearAttention(dim, attn_type,
                                                       heads=heads, dim_head=dim_head, dropout=dropout,
                                                       relative_emb=relative_emb,
                                                       scale=scale,

                                                       relative_emb_dim=relative_emb_dim,
                                                       min_freq=min_freq,
                                                       init_method='orthogonal',
                                                       cat_pos=cat_pos,
                                                       pos_dim=relative_emb_dim,
                                                  )
        self.use_ln = use_ln
        self.residual = residual
        self.use_ffn = use_ffn

        if self.use_ln:
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)

        if self.use_ffn:
            self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, z, x_pos=None, z_pos=None):
        # x in [b n1 c]
        # b, n1, c = x.shape   # coordinate encoding
        # b, n2, c = z.shape   # system encoding
        if self.use_ln:
            z = self.ln1(z)
            if self.residual:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos)) + x
            else:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos))
        else:
            if self.residual:
                x = self.cross_attn_module(x, z, x_pos, z_pos) + x
            else:
                x = self.cross_attn_module(x, z, x_pos, z_pos)

        if self.use_ffn:
            x = self.ffn(x) + x

        return x
    


# code copied from: https://github.com/ndahlquist/pytorch-fourier-feature-networks
# author: Nic Dahlquist
class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):

        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


# New position encoding module
# modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d//2], t[..., d//2:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)


class LinearAttention(nn.Module):
    """
    Contains following two types of attention, as discussed in "Choose a Transformer: Fourier or Galerkin"

    Galerkin type attention, with instance normalization on Key and Value
    Fourier type attention, with instance normalization on Query and Key
    """
    def __init__(self,
                 dim,
                 attn_type,                 # ['fourier', 'galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 init_params=True,
                 relative_emb=False,
                 scale=1.,
                 init_method='orthogonal',    # ['xavier', 'orthogonal']
                 init_gain=None,
                 relative_emb_dim=2,
                 min_freq=1/64,             # 1/64 is for 64 x 64 ns2d,
                 cat_pos=False,
                 pos_dim=2,
                 use_ln=False
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.attn_type = attn_type
        self.use_ln = use_ln

        self.heads = heads
        self.dim_head = dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        if attn_type == 'galerkin':
            if not self.use_ln:
                self.k_norm = nn.InstanceNorm1d(dim_head)
                self.v_norm = nn.InstanceNorm1d(dim_head)
            else:
                self.k_norm = nn.LayerNorm(dim_head)
                self.v_norm = nn.LayerNorm(dim_head)

        elif attn_type == 'fourier':
            if not self.use_ln:
                self.q_norm = nn.InstanceNorm1d(dim_head)
                self.k_norm = nn.InstanceNorm1d(dim_head)
            else:
                self.q_norm = nn.LayerNorm(dim_head)
                self.k_norm = nn.LayerNorm(dim_head)

        else:
            raise Exception(f'Unknown attention type {attn_type}')

        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim + pos_dim*heads, dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / dim_head
            self.diagonal_weight = 1. / dim_head
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain

        self.init_method = init_method
        if init_params:
            self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            assert not cat_pos
            self.emb_module = RotaryEmbedding(dim_head // self.relative_emb_dim, min_freq=min_freq, scale=scale)

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        for param in self.to_qkv.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # for q
                    init_fn(param[h * self.dim_head:(h + 1) * self.dim_head, :], gain=self.init_gain)

                    param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                torch.diag(torch.ones(
                                                                                    param.size(-1),
                                                                                    dtype=torch.float32))

                    # for k
                    init_fn(param[2 * h * self.dim_head:(2 * h + 1) * self.dim_head, :], gain=self.init_gain)
                    param.data[2 * h * self.dim_head:(2 * h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                        torch.diag(torch.ones(
                                                                                            param.size(-1),
                                                                                            dtype=torch.float32))

                    # for v
                    init_fn(param[3 * h * self.dim_head:(3 * h + 1) * self.dim_head, :], gain=self.init_gain)
                    param.data[3 * h * self.dim_head:(3 * h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                        torch.diag(torch.ones(
                                                                                            param.size(-1),
                                                                                            dtype=torch.float32))

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, x, pos=None, not_assoc=False, padding_mask=None):
        # padding mask will be in shape [b, n, 1], it will indicates which point are padded and should be ignored
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if pos is None and self.relative_emb:
            raise Exception('Must pass in coordinates when under relative position embedding mode')

        if padding_mask is None:
            if self.attn_type == 'galerkin':
                k = self.norm_wrt_domain(k, self.k_norm)
                v = self.norm_wrt_domain(v, self.v_norm)
            else:  # fourier
                q = self.norm_wrt_domain(q, self.q_norm)
                k = self.norm_wrt_domain(k, self.k_norm)
        else:
            grid_size = torch.sum(padding_mask, dim=[-1, -2]).view(-1, 1, 1, 1)  # [b, 1, 1]
            padding_mask = repeat(padding_mask, 'b n d -> (b h) n d', h=self.heads)  # [b, n, 1]

            if self.use_ln:
                if self.attn_type == 'galerkin':
                    k = self.k_norm(k)
                    v = self.v_norm(v)
                else:  # fourier
                    q = self.q_norm(q)
                    k = self.k_norm(k)
            else:

                if self.attn_type == 'galerkin':
                    k = rearrange(k, 'b h n d -> (b h) n d')
                    v = rearrange(v, 'b h n d -> (b h) n d')

                    k = masked_instance_norm(k, padding_mask)
                    v = masked_instance_norm(v, padding_mask)

                    k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)
                    v = rearrange(v, '(b h) n d -> b h n d', h=self.heads)
                else:  # fourier
                    q = rearrange(q, 'b h n d -> (b h) n d')
                    k = rearrange(k, 'b h n d -> (b h) n d')

                    q = masked_instance_norm(q, padding_mask)
                    k = masked_instance_norm(k, padding_mask)

                    q = rearrange(q, '(b h) n d -> b h n d', h=self.heads)
                    k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)

            padding_mask = rearrange(padding_mask, '(b h) n d -> b h n d', h=self.heads)  # [b, h, n, 1]


        if self.relative_emb:
            if self.relative_emb_dim == 2:
                freqs_x = self.emb_module.forward(pos[..., 0], x.device)
                freqs_y = self.emb_module.forward(pos[..., 1], x.device)
                freqs_x = repeat(freqs_x, 'b n d -> b h n d', h=q.shape[1])
                freqs_y = repeat(freqs_y, 'b n d -> b h n d', h=q.shape[1])

                q = apply_2d_rotary_pos_emb(q, freqs_x, freqs_y)
                k = apply_2d_rotary_pos_emb(k, freqs_x, freqs_y)
            elif self.relative_emb_dim == 1:
                assert pos.shape[-1] == 1
                freqs = self.emb_module.forward(pos[..., 0], x.device)
                freqs = repeat(freqs, 'b n d -> b h n d', h=q.shape[1])
                q = apply_rotary_pos_emb(q, freqs)
                k = apply_rotary_pos_emb(k, freqs)
            else:
                raise Exception('Currently doesnt support relative embedding > 2 dimensions')

        elif self.cat_pos:
            assert pos.size(-1) == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.heads, 1, 1])
            q, k, v = [torch.cat([pos, x], dim=-1) for x in (q, k, v)]

        if not_assoc:
            # this is more efficient when n<<c
            score = torch.matmul(q, k.transpose(-1, -2))
            if padding_mask is not None:
                padding_mask = ~padding_mask
                padding_mask_arr = torch.matmul(padding_mask, padding_mask.transpose(-1, -2))  # [b, h, n, n]
                mask_value = 0.
                score = score.masked_fill(padding_mask_arr, mask_value)
                out = torch.matmul(score, v) * (1./grid_size)
            else:
                out = torch.matmul(score, v) * (1./q.shape[2])
        else:
            if padding_mask is not None:
                q = q.masked_fill(~padding_mask, 0)
                k = k.masked_fill(~padding_mask, 0)
                v = v.masked_fill(~padding_mask, 0)
                dots = torch.matmul(k.transpose(-1, -2), v)
                out = torch.matmul(q, dots) * (1. / grid_size)
            else:
                dots = torch.matmul(k.transpose(-1, -2), v)
                out = torch.matmul(q, dots) * (1./q.shape[2])
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class StandardAttention(nn.Module):
    """Standard scaled dot product attention"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., causal=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.causal = causal  # simple autogressive attention with upper triangular part being masked zero

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            if self.causal:
                raise Exception('Passing in mask while attention is not causal')
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)     # similarity score

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class CrossLinearAttention(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,  # ['fourier', 'galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 init_params=True,
                 relative_emb=False,
                 scale=1.,
                 init_method='orthogonal',  # ['xavier', 'orthogonal']
                 init_gain=None,
                 relative_emb_dim=2,
                 min_freq=1 / 64,  # 1/64 is for 64 x 64 ns2d,
                 cat_pos=False,
                 pos_dim=2,
                 use_ln=False,
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.attn_type = attn_type
        self.use_ln = use_ln

        self.heads = heads
        self.dim_head = dim_head

        # query is the classification token
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        if attn_type == 'galerkin':
            if not self.use_ln:
                self.k_norm = nn.InstanceNorm1d(dim_head)
                self.v_norm = nn.InstanceNorm1d(dim_head)
            else:
                self.k_norm = nn.LayerNorm(dim_head)
                self.v_norm = nn.LayerNorm(dim_head)

        elif attn_type == 'fourier':
            if not self.use_ln:
                self.q_norm = nn.InstanceNorm1d(dim_head)
                self.k_norm = nn.InstanceNorm1d(dim_head)
            else:
                self.q_norm = nn.LayerNorm(dim_head)
                self.k_norm = nn.LayerNorm(dim_head)

        else:
            raise Exception(f'Unknown attention type {attn_type}')

        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim + pos_dim*heads, dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / dim_head
            self.diagonal_weight = 1. / dim_head
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain
        self.init_method = init_method
        if init_params:
            self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            self.emb_module = RotaryEmbedding(dim_head // self.relative_emb_dim, min_freq=min_freq, scale=scale)

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        for param in self.to_kv.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # for k
                    init_fn(param[h*self.dim_head:(h+1)*self.dim_head, :], gain=self.init_gain)
                    param.data[h*self.dim_head:(h+1)*self.dim_head, :] += self.diagonal_weight * \
                                                                          torch.diag(torch.ones(
                                                                              param.size(-1), dtype=torch.float32))

                    # for v
                    init_fn(param[2 * h * self.dim_head:(2*h+1) * self.dim_head, :], gain=self.init_gain)
                    param.data[2 * h * self.dim_head:(2*h+1) * self.dim_head, :] += self.diagonal_weight * \
                                                                           torch.diag(torch.ones(
                                                                               param.size(-1), dtype=torch.float32))
        for param in self.to_q.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # for q
                    init_fn(param[h * self.dim_head:(h + 1) * self.dim_head, :], gain=self.init_gain)
                    param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                torch.diag(torch.ones(
                                                                                    param.size(-1), dtype=torch.float32))

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, x, z, x_pos=None, z_pos=None, padding_mask=None):
        # x (z^T z)
        # x [b, n1, d]
        # z [b, n2, d]
        n1 = x.shape[1]   # x [b, n1, d]
        n2 = z.shape[1]   # z [b, n2, d]
        if padding_mask is not None:
            grid_size = torch.sum(padding_mask, dim=1).view(-1, 1, 1, 1)

        q = self.to_q(x)

        kv = self.to_kv(z).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        if (x_pos is None or z_pos is None) and self.relative_emb:
            raise Exception('Must pass in coordinates when under relative position embedding mode')
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        if padding_mask is None:
            if self.attn_type == 'galerkin':
                k = self.norm_wrt_domain(k, self.k_norm)
                v = self.norm_wrt_domain(v, self.v_norm)
            else:  # fourier
                q = self.norm_wrt_domain(q, self.q_norm)
                k = self.norm_wrt_domain(k, self.k_norm)
        else:
            padding_mask = repeat(padding_mask, 'b n d -> (b h) n d', h=self.heads)  # [b, n, 1]
            if self.use_ln:
                if self.attn_type == 'galerkin':
                    k = self.k_norm(k)
                    v = self.v_norm(v)
                else:  # fourier
                    q = self.q_norm(q)
                    k = self.k_norm(k)
            else:


                if self.attn_type == 'galerkin':
                    k = rearrange(k, 'b h n d -> (b h) n d')
                    v = rearrange(v, 'b h n d -> (b h) n d')

                    k = masked_instance_norm(k, padding_mask)
                    v = masked_instance_norm(v, padding_mask)

                    k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)
                    v = rearrange(v, '(b h) n d -> b h n d', h=self.heads)
                else:  # fourier
                    q = rearrange(q, 'b h n d -> (b h) n d')
                    k = rearrange(k, 'b h n d -> (b h) n d')

                    q = masked_instance_norm(q, padding_mask)
                    k = masked_instance_norm(k, padding_mask)

                    q = rearrange(q, '(b h) n d -> b h n d', h=self.heads)
                    k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)

            padding_mask = rearrange(padding_mask, '(b h) n d -> b h n d', h=self.heads)  # [b, h, n, 1]

        if self.relative_emb:
            if self.relative_emb_dim == 2:

                x_freqs_x = self.emb_module.forward(x_pos[..., 0], x.device)
                x_freqs_y = self.emb_module.forward(x_pos[..., 1], x.device)
                x_freqs_x = repeat(x_freqs_x, 'b n d -> b h n d', h=q.shape[1])
                x_freqs_y = repeat(x_freqs_y, 'b n d -> b h n d', h=q.shape[1])

                z_freqs_x = self.emb_module.forward(z_pos[..., 0], z.device)
                z_freqs_y = self.emb_module.forward(z_pos[..., 1], z.device)
                z_freqs_x = repeat(z_freqs_x, 'b n d -> b h n d', h=q.shape[1])
                z_freqs_y = repeat(z_freqs_y, 'b n d -> b h n d', h=q.shape[1])

                q = apply_2d_rotary_pos_emb(q, x_freqs_x, x_freqs_y)
                k = apply_2d_rotary_pos_emb(k, z_freqs_x, z_freqs_y)

            elif self.relative_emb_dim == 1:
                assert x_pos.shape[-1] == 1 and z_pos.shape[-1] == 1
                x_freqs = self.emb_module.forward(x_pos[..., 0], x.device)
                x_freqs = repeat(x_freqs, 'b n d -> b h n d', h=q.shape[1])

                z_freqs = self.emb_module.forward(z_pos[..., 0], x.device)
                z_freqs = repeat(z_freqs, 'b n d -> b h n d', h=q.shape[1])

                q = apply_rotary_pos_emb(q, x_freqs)  # query from x domain
                k = apply_rotary_pos_emb(k, z_freqs)  # key from z domain
            else:
                raise Exception('Currently doesnt support relative embedding > 2 dimensions')
        elif self.cat_pos:
            assert x_pos.size(-1) == self.pos_dim and z_pos.size(-1) == self.pos_dim
            x_pos = x_pos.unsqueeze(1)
            x_pos = x_pos.repeat([1, self.heads, 1, 1])
            q = torch.cat([x_pos, q], dim=-1)

            z_pos = z_pos.unsqueeze(1)
            z_pos = z_pos.repeat([1, self.heads, 1, 1])
            k = torch.cat([z_pos, k], dim=-1)
            v = torch.cat([z_pos, v], dim=-1)

        if padding_mask is not None:
            q = q.masked_fill(~padding_mask, 0)
            k = k.masked_fill(~padding_mask, 0)
            v = v.masked_fill(~padding_mask, 0)
            dots = torch.matmul(k.transpose(-1, -2), v)
            out = torch.matmul(q, dots) * (1. / grid_size)
        else:
            dots = torch.matmul(k.transpose(-1, -2), v)
            out = torch.matmul(q, dots) * (1./n2)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


# helpers

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = idx.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    # print("point", points.device, "batch indice", batch_indices.device, "id", idx.device)
    new_points = points[batch_indices, idx, :]
    return new_points


# helper
def knn(x1, x2, k):
    """
    knn 
    """
    # x1 [b, n1, c], x2 [b, n2, c]
    inner = -2 * torch.matmul(x1, rearrange(x2, 'b n c -> b c n'))  # [b n1 n2]
    xx = torch.sum(x1 ** 2, dim=-1, keepdim=True)  # [b, n1, 1]
    yy = torch.sum(x2 ** 2, dim=-1, keepdim=True)  # [b, n2, 1]
    pairwise_distance = -xx - inner - yy.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, n1, k)
    return idx



class SimpleAttentionEncoder(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 seq_len,                  # this should be the input sequence length
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 n_patch=16,
                 out_grid=64,
                 emb_dropout=0.1,           # dropout of embedding
                 attention_init='xavier',
                 ):
        super().__init__()
        self.n_patch = n_patch
        self.out_grid = out_grid

        t = seq_len
        self.dropout = nn.Dropout(emb_dropout)

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
        )

        self.temp_embedding = nn.Parameter(
            torch.cat((torch.tensor([-1.]), torch.linspace(0, 1, t)), dim=0).view(1, t+1, 1), requires_grad=False)   # [b, t, 1]
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_emb_dim), requires_grad=True)
        self.cls_emb = nn.Parameter(torch.randn(1, 1, 2), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

        self.st_transformer = STTransformerCat(in_emb_dim, 1, 4, 64, in_emb_dim, 'galerkin', attention_init=attention_init)

        self.s_transformer = TransformerCat(in_emb_dim*2, depth-1, 4, 64, in_emb_dim*2, 'galerkin', attention_init=attention_init)

        self.to_cls = nn.Sequential(
            nn.LayerNorm(2*in_emb_dim),
            nn.Linear(2*in_emb_dim, out_seq_emb_dim, bias=True)
            )

        self.shrink_temporal = nn.Sequential(
            Rearrange('b n t c -> b n (t c)'),
            nn.Linear(t*in_emb_dim, 2*in_emb_dim, bias=False),
        )
        self.expand_feat = nn.Linear(in_emb_dim, 2*in_emb_dim)

        self.project_to_latent = nn.Sequential(
            nn.Linear(2*in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, c, t, n]
                input_pos,  # [b, n, 2]
                ):
        x = rearrange(x, 'b c t n-> b t n c')
        x = self.to_embedding(x)
        x = self.dropout(x)

        x, x_cls = self.st_transformer.forward(x,
                                             self.cls_token,
                                             self.temp_embedding,
                                             input_pos)
        x = self.shrink_temporal(x)
        x_cls = self.expand_feat(x_cls)
        x, x_cls = self.s_transformer.forward(x, x_cls, input_pos, self.cls_emb)
        x, x_cls = self.project_to_latent(x), self.to_cls(x_cls) * self.gamma

        return x, x_cls


class NoSTAttentionEncoder(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 seq_len,                  # this should be the input sequence length
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 emb_dropout=0.1,           # dropout of embedding
                 ):
        super().__init__()

        t = seq_len
        self.dropout = nn.Dropout(emb_dropout)

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 2*in_emb_dim), requires_grad=True)
        self.cls_emb = nn.Parameter(torch.randn(1, 1, 2), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

        self.shrink_temporal = nn.Sequential(
            Rearrange('b t n c -> b n (t c)'),
            nn.Linear(t * in_emb_dim, 2 * in_emb_dim, bias=False),
        )
        self.s_transformer = TransformerCat(in_emb_dim*2, depth, 4, 64, in_emb_dim*2, 'galerkin', init_scale=32)

        self.to_cls = nn.Sequential(
            nn.Linear(2*in_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.LayerNorm(out_seq_emb_dim))

        self.project_to_latent = nn.Sequential(
            nn.Linear(2*in_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.InstanceNorm1d(in_emb_dim))

    def forward(self,
                x,  # [b, c, t, n]
                input_pos,  # [b, n, 2]
                ):
        x = rearrange(x, 'b c t n-> b t n c')
        x = self.to_embedding(x)
        x = self.dropout(x)
        x = self.shrink_temporal(x)

        x, x_cls = self.s_transformer.forward(x, self.cls_token, input_pos, self.cls_emb)
        x, x_cls = self.project_to_latent(x), self.to_cls(x_cls) * self.gamma

        return x, x_cls


class SpatialEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 res,
                 use_ln=True,
                 emb_dropout=0.05,           # dropout of embedding
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                 'galerkin',
                                                 use_relu=False,
                                                 use_ln=use_ln,
                                                 scale=[res, res//4] + [1]*(depth-2),
                                                 relative_emb_dim=2,
                                                 min_freq=1 / res,
                                                 dropout=0.03,
                                                 attention_init='orthogonal')

        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, n, c]
                input_pos,  # [b, n, 2]
                ):

        x = self.to_embedding(x)
        x = self.dropout(x)

        x = self.s_transformer.forward(x, input_pos)
        x = self.to_out(x)

        return x


class Encoder1D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 emb_dropout=0.05,           # dropout of embedding
                 res=2048,
                 ):
        super().__init__()

        self.dropout = nn.Dropout(emb_dropout)

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        self.transformer = TransformerCatNoCls(in_emb_dim, depth, 1, in_emb_dim, in_emb_dim, 'fourier',
                                               scale=[8.] + [4.]*2 + [1.]*(depth-3),
                                               relative_emb_dim=1,
                                               min_freq=1/res,
                                               use_ln=True,
                                               dropout=emb_dropout,
                                               attention_init='orthogonal')

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, n, c]
                input_pos,  # [b, n, 1]
                ):
        x = torch.cat((x, input_pos/16.), dim=-1)
        x = self.to_embedding(x)
        # x = self.dropout(x)
        # x = torch.cat((x, input_pos), dim=-1)
        x = self.transformer.forward(x, input_pos)
        x = self.project_to_latent(x)

        return x


# for ablation
class NoRelSpatialTemporalEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            # Rearrange('b c n -> b n c'),
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        if depth > 4:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=-1,
                                                     attention_init='orthogonal')
        else:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=-1,
                                                     attention_init='orthogonal')

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, t(*c)+2, n]
                input_pos,  # [b, n, 2]
                ):

        x = self.to_embedding(x)
        x = self.s_transformer.forward(x, input_pos)
        x = self.project_to_latent(x)

        return x


class SpatialOperator2D(nn.Module):
    # this directly takes in the input and output solution
    # input and output are supposed to be on the same grid
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_chanels,
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 res,
                 use_ln=True,
                 emb_dropout=0.05,           # dropout of embedding
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.s_transformer = TransformerWithPad(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                 'galerkin',
                                                 use_relu=False,
                                                 use_ln=use_ln,
                                                 scale=[res, res//4] + [1]*(depth-2),
                                                 relative_emb_dim=2,
                                                 min_freq=1 / res,
                                                 dropout=0.,
                                                 attention_init='orthogonal')

        self.ln = nn.LayerNorm(in_emb_dim)

        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim+1, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, out_chanels, bias=False))

    def forward(self,
                x,  # [b, n, c]
                input_pos,  # [b, n, 2]
                param,  # [b,]
                pad_mask,    # [b, n, 1]
                ):
        param = param.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], 1)  # [b, n, 1]
        x = torch.cat((x, param, input_pos), dim=-1)
        x = self.to_embedding(x)
        x = x.masked_fill(~pad_mask, 0.)

        x_skip = x
        x = self.dropout(x)

        x = self.s_transformer.forward(x, input_pos, pad_mask)

        x = x.masked_fill(~pad_mask, 0.)

        x = self.ln(x+x_skip)
        x = torch.cat((x, param), dim=-1)
        x = self.to_out(x)
        x = x.masked_fill(~pad_mask, 0.)


        return x

    def denormalize(self,
                    x,   # [b, n, c]
                    dataset,
                    bound_mask,   # [b, n, 4]  left right bottom top
                    ):

        # impose Dirichlet boundary condition

        # velocity
        x[:, :, :2] = x[:, :, :2] * dataset.statistics['vel_std'] + dataset.statistics['vel_mean']
        all_bound = torch.any(bound_mask, dim=-1, keepdim=True)
        x[:, :, :2] = x[:, :, :2].masked_fill(all_bound, 0.)

        # pressure
        x[:, :, 2] = x[:, :, 2] * dataset.statistics['prs_std'] + dataset.statistics['prs_mean']

        # temperature
        x[:, :, 3] = x[:, :, 3] * dataset.statistics['temp_std'] + dataset.statistics['temp_mean']
        x[:, :, 3] = x[:, :, 3].masked_fill(bound_mask[..., 0], 1.)
        x[:, :, 3] = x[:, :, 3].masked_fill(bound_mask[..., 1], 0.)

        return x


class Oformer(nn.Module):
    def __init__(self,
                 n_channels=1,
                 in_timesteps =10,
                 encoder_emb_dim=128,
                 out_seq_emb_dim=128,
                 encoder_heads=4,
                 encoder_depth=2,
                 decoder_emb_dim=256,
                 out_channels=3,
                 out_step=1,
                 propagator_depth=2, #  decoder depth
                 fourier_frequency=8
                 ):
        super().__init__()
        in_channels = n_channels * in_timesteps + 2
        self.encoder = SpatialTemporalEncoder2D(
                in_channels,
                encoder_emb_dim,
                out_seq_emb_dim,
                encoder_heads,
                encoder_depth,
            )
        self.decoder = PointWiseDecoder2D(
                decoder_emb_dim,
                out_channels,
                out_step,
                propagator_depth,
                scale=fourier_frequency,
                dropout=0.0,
            )
        self.grid = None
        
    def get_grid(self,shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        if self.grid is None:
            x0, y0 = torch.meshgrid(torch.linspace(0, 1, size_x), torch.linspace(0, 1, size_y))
            xs = torch.cat((x0[None, ...], y0[None, ...]), dim=0)  # [2, 64, 64]
            # print(xs.shape)
            grid = rearrange(xs, 'c h w -> (h w) c').unsqueeze(0)  # [64*64, 2]
            self.grid = grid.to(device)
        else:
            grid = self.grid.to(device)
        return grid

    def forward(self, x): 
        B, H, W, T, C = x.shape
        device = x.device
        grid = self.get_grid(x.shape, device) # (1, H*W, C)
        
        input_pos = prop_pos = repeat(grid, '() n c -> b n c', b=B)  # (B, H*W, C)
        input_pos = input_pos.to(device)
        prop_pos= prop_pos.to(device)
        in_seq = rearrange(x, 'b h w t c -> b (h w) (t c)')
        
        if self.training:
            sampling_ratio = np.random.uniform(0.45, 0.95)
            input_idx = torch.as_tensor(
                np.concatenate(
                    [np.random.choice(input_pos.shape[1], int(sampling_ratio*input_pos.shape[1]), replace=False).reshape(1,-1)
                        for _ in range(in_seq.shape[0])], axis=0)
                ).view(in_seq.shape[0], -1).to(x.device)
            in_seq = index_points(in_seq, input_idx)
            input_pos = index_points(input_pos, input_idx)

        mask = input_pos    
        z = self.encoder.forward(in_seq, mask)

        x_out = self.decoder.rollout(z, prop_pos, 1, mask)  # [b, seq_len, n]
        
        # reshape x_out
        x_out = rearrange(x_out, 'b (h w) t c->b h w t c', h=H, w=W)
        return x_out
    





if __name__ == '__main__':
    x = torch.rand(2, 128, 128, 10, 3)
    # x = torch.rand(2, 96, 196, 10, 3)
    model = Oformer(n_channels=x.shape[-1], in_timesteps =x.shape[-2],
                    encoder_emb_dim=96, out_seq_emb_dim=192, encoder_depth=3,
                    decoder_emb_dim=384,propagator_depth=1, encoder_heads=1, fourier_frequency=8)
    model.train()
    y = model(x)
    print('y shape', y.shape)