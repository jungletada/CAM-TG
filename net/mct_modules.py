import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from timm.models.registry import register_model

from net.gcn_lib import Grapher


def nlc2nchw(x, d_size):
    _, N, C = x.shape
    assert d_size[0] * d_size[1] == N, f"{d_size} not equal to {N}"
    x = x.permute(0, 2, 1).reshape(-1, C, d_size[0], d_size[1]).contiguous()
    return x


def nchw2nlc(x):
    B, C = x.shape[:2]
    x = x.permute(0, 2, 3, 1).reshape(B, -1, C).contiguous()
    return x


class LayerNormGeneral(nn.Module):
    r""" General LayerNorm for different situations.
    https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py#L313
    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_dim by default. 
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance. 
        scale (bool): Flag indicates whether to use scale or not.
        bias (bool): Flag indicates whether to use scale or not.

        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.

        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True.

        For the several metaformer baslines,
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).
    """
    def __init__(self, affine_shape=None, normalized_dim=(1, 2, 3), scale=True, 
        bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x
    
    
class Attention(nn.Module):
    def __init__(self, dim, num_classes=None, num_heads=4, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.Cls = 0 if num_classes is None else num_classes
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # Here N = #patches + #class-tokens
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        # d for each head, Nd heads in total. --> B x Nd x N x d for {q, k, v}.
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x Nd x N x N
        #======================================================================#   
        if self.Cls != 0:            
            attn_cls, attn_pat = torch.split(attn, [self.Cls, N-self.Cls], dim=-1)
            attn_pat = attn_pat.softmax(dim=-1)
            attn_cls = attn_cls.softmax(dim=-1)
            attn = torch.cat((attn_cls, attn_pat), dim=-1) 
        else:
            attn = attn.softmax(dim=-1)
        #======================================================================#     
        weights = attn # # B x Nd x N x N
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.Cls != 0:
            return x, weights
        else:
            return x
        

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_classes, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ):
        super().__init__()
        self.Cls = 0 if num_classes is None else num_classes
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_classes=num_classes, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop,)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.Cls != 0:
            o, weights = self.attn(self.norm1(x))
            x = x + self.drop_path(o)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, weights
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class Semantic2FeatureBlock(nn.Module):
    def __init__(self, channels, num_classes, num_heads=4, kernel_size=9, dilation=1, 
                 drop_path=0., proj_drop=0.):
        super().__init__()
        self.n_heads = num_heads
        self.proj_q = nn.Linear(channels, channels)
        self.proj_kv = nn.Conv2d(channels, 2 * channels, 1)
        
        self.norm_cls = nn.LayerNorm(channels)
        self.norm_x = LayerNormGeneral(affine_shape=(channels, 1, 1))
        
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scale = (channels//self.n_heads) ** -0.5
        self.act = nn.GELU()
        self.grapher = Grapher(
            in_channels=num_heads * num_classes, 
            kernel_size=kernel_size,
            dilation=dilation,
            conv='mr', 
            act='gelu', 
            groups=num_heads,
            norm='batch',
            bias=True, 
            stochastic=False, 
            drop_path=drop_path,
            epsilon=0.2, 
            r=1)
        
        self.proj_patch = nn.Sequential(
            nn.Conv2d(num_classes, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),)
        
    def forward(self, x_cls, input_patch):
        """
        Input:
            x_cls->[B, Cls, C] 
            x_patch->[B, C, Hp, Wp]
        Return:
            out_cls->[B, Cls, C] 
        """
        B, Cls, C = x_cls.shape
        Hp, Wp = input_patch.shape[2:]
        query = self.proj_q(self.norm_cls(x_cls))
        query = query.reshape(B, -1, self.n_heads, C//self.n_heads).permute(0, 2, 1, 3) # [B, Nd, Cls, d]
        
        kv = self.proj_kv(self.norm_x(input_patch)).reshape(B, 2, C, Hp, Wp).permute(1, 0, 2, 3, 4)
        k, v = kv[0], kv[1]
        
        key = k.reshape(B, self.n_heads, C//self.n_heads, Hp * Wp) # [B, Nd, d, Hp*Wp]
        value = v.reshape(B, self.n_heads, C//self.n_heads, Hp * Wp).permute(0, 1, 3, 2) # [B, Nd, Hp*Wp, d]
        
        attn = query @ key * self.scale # [B, Nd, Cls, Hp*Wp]
        # attn = guide.reshape(B, self.n_heads*Cls, Hp, Wp)
        # attn = self.grapher(attn) # forward attention to Grapher
        # attn = attn.reshape(B, self.n_heads, Cls, Hp*Wp) # [B, Nd, Cls, Hp*Wp]
        attn = attn.softmax(dim=-1)
    
        out_cls = attn @ value # [B, Nd, Cls, d]
        out_cls = out_cls.permute(0, 2, 1, 3).reshape(B, Cls, C) # [B, C, Cls]
        out_cls = self.proj(out_cls)
        out_cls = x_cls + self.proj_drop(out_cls) # [B, Cls, C]
        
        # use updated class tokens to guide the output patch
        out_patch = input_patch.view(B, C, -1)      # [B, C, Hp*Wp]
        # print(f"out cls: {out_cls.transpose(-2, -1).shape}")
        # print(f"out pat: {out_patch.shape}")
        out_patch = out_cls @ out_patch # [B, [Cls, C]@[C, Hp*Wp]]
        out_patch = self.proj_patch(out_patch.view(B, -1, Hp, Wp))
        
        return out_cls, out_patch
        
        
class Feature2SemanticBlock(nn.Module):
    def __init__(self, channels, num_classes, num_heads=4, kernel_size=9, dilation=1, 
                 drop_path=0., proj_drop=0.):
        super().__init__()
        self.n_heads = num_heads
        self.proj_q = nn.Conv2d(channels, channels, 1)
        self.proj_kv = nn.Linear(channels, 2 * channels)
        
        self.norm_cls = nn.LayerNorm(channels)
        self.norm_x = LayerNormGeneral(affine_shape=(channels, 1, 1))
        
        self.proj = nn.Conv2d(channels, channels, 1)
        self.proj_drop = nn.Dropout2d(proj_drop)
        self.scale = (channels//self.n_heads) ** -0.5
        
        self.grapher = Grapher(
            in_channels=num_heads * num_classes, 
            kernel_size=kernel_size,
            dilation=dilation,
            conv='mr', 
            act='gelu', 
            groups=num_heads,
            norm='batch',
            bias=True, 
            stochastic=False, 
            drop_path=drop_path,
            epsilon=0.2, 
            r=1)
        
    def forward(self, x_cls, x_patch):
        """
        Input:
            x_cls->[B, Cls, C] as key and value
            x_patch->[B, C, Hp, Wp] as query
        Return:
            out_patch->[B, C, Hp, Wp] as output
        """
        B, Cls, C = x_cls.shape # C = Nd * d
        Hp, Wp = x_patch.shape[2:]
        
        query = self.proj_q(self.norm_x(x_patch)).reshape(B, C, -1) # [B, C, Hp*Wp]
        query = query.reshape(B, self.n_heads, C//self.n_heads, -1).permute(0, 1, 3, 2) # [B, Nd, Hp*Wp, d]
        
        kv = self.proj_kv(self.norm_cls(x_cls)).reshape(B, Cls, 2, -1).permute(2, 0, 1, 3) # [B, Cls, 2, C]->[2, B, Cls, C]
        k, v = kv[0], kv[1] # [B, Cls, C]
        
        key = k.reshape(B, Cls, self.n_heads, C//self.n_heads).permute(0, 2, 3, 1)   # [B, Nd, d, Cls]
        value = v.reshape(B, Cls, self.n_heads, C//self.n_heads).permute(0, 2, 1, 3) # [B, Nd, Cls, d]
        
        attn = query @ key * self.scale # [B, Nd, Hp*Wp, Cls]->[Hp*Wp, d] @ [d, Cls]
        attn = attn.reshape(B, self.n_heads, Hp, Wp, Cls).permute(
            0, 1, 4, 2, 3).reshape(B, self.n_heads*Cls, Hp, Wp) # [B, Nd, Hp, Wp, Cls]->[B, Nd, Cls, Hp, Wp]->[B, Nd*Cls, Hp, Wp]
        attn = self.grapher(attn) # forward attention to Grapher
        attn = attn.reshape(B, self.n_heads, Cls, Hp*Wp).permute(0, 1, 3, 2) # [B, Nd, Hp*Wp, Cls]
        attn = attn.softmax(dim=-1)
    
        out_x = attn @ value # [B, Nd, Hp*Wp, d] -> [Hp*Wp, Cls] @ [Cls, d]
        out_x = out_x.permute(0, 1, 3, 2).reshape(B, C, Hp, Wp) # [B, C, Hp, Wp]
        out_x = self.proj(out_x)
        out_x = x_patch + self.proj_drop(out_x)
        return out_x
    

class GraphAttentionLayer(nn.Module):
    def __init__(self, channels, num_classes, num_heads=4, kernel_size=9, dilation=1, 
                 drop_path=0., proj_drop=0.):
        super().__init__()
        kargs = {
            'channels': channels, 
            'num_classes': num_classes, 
            'num_heads': num_heads,
            'drop_path': drop_path, 
            'proj_drop': proj_drop, 
            'kernel_size': kernel_size, 
            'dilation': dilation
        }
        self.sem2feat = Semantic2FeatureBlock(**kargs)
        self.feat2sem = Feature2SemanticBlock(**kargs)
        self.ffn = MLP(channels, channels*4, channels)
    
    def forward(self, x_cls, x_patch):
        """
        Input:
            x_cls, x_patch
        Return:
            x_cls, x_patch
        """
        pat_size = x_patch.shape[2:]
        n_cls = x_cls.shape[1]
        x_cls = self.sem2feat(x_cls, x_patch)
        x_patch = self.feat2sem(x_cls, x_patch)
        x_patch = nchw2nlc(x_patch)
        x = torch.cat((x_cls, x_patch), dim=1)
        x = self.ffn(x)
        x_cls, x_patch = x[:, :n_cls], x[:, n_cls:]
        x_patch = nlc2nchw(x_patch, pat_size)
        return x_cls, x_patch
    

class SpatialPriorModule(nn.Module):
    def __init__(self, 
                 inplanes=64, 
                 embed_dims=[96, 192, 384, 768], 
                 norm_layer=nn.BatchNorm2d, 
                 act_layer=nn.GELU):
        super().__init__()
        self.stem = nn.Sequential(*[ # downsample by 4
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(inplanes),
            act_layer(),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(inplanes),
            act_layer(),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(inplanes),
            act_layer(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]) 
        self.conv2 = nn.Sequential(*[# downsample by 2
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(2 * inplanes),
            act_layer(),
        ])
        self.conv3 = nn.Sequential(*[ # downsample by 2
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(4 * inplanes),
            act_layer(),
        ])
        self.conv4 = nn.Sequential(*[ # downsample by 2
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(4 * inplanes),
            act_layer(),
        ])
        self.conv5 = nn.Sequential(*[ # downsample by 2
            nn.Conv2d(4 * inplanes, 8 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(8 * inplanes),
            act_layer(),
        ])
        
        # self.fc1 = nn.Conv2d(inplanes, embed_dims[0], kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dims[0], kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dims[1], kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dims[2], kernel_size=1, stride=1, padding=0, bias=True)
        self.fc5 = nn.Conv2d(8 * inplanes, embed_dims[3], kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        
        # c1 = self.fc1(c1) # 4s
        c2 = self.fc2(c2) # 8s
        c3 = self.fc3(c3) # 16s
        c4 = self.fc4(c4) # 32s
        c5 = self.fc5(c5) # 64s
    
        return [c2, c3, c4, c5]