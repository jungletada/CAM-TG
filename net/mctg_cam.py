import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, to_2tuple

from net.modules import DownConv
from net.modules import SpatialPriorModule, SpatialFuseModule
from net.base_vit import VisionTransformer

    
class MCTG(VisionTransformer):
    def __init__(self, decay_parameter=0.996, input_size=448, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.head.apply(self._init_weights)
        
        img_size = to_2tuple(input_size)
        patch_size = to_2tuple(self.patch_embed.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed_cls, std=.02)
        trunc_normal_(self.pos_embed_pat, std=.02)
        self.avgpool2d = nn.AdaptiveAvgPool2d(1)
         
        #========================================================================================#
        self.stage_indices = (0, 3, 6, 9, 12)
        self.spatial_dims = [384, 384, 384, 384]
        mask_ratios = [0.3, 0.2, 0.1, 0.0]
        self.stages = len(self.stage_indices) - 1
       
        self.decay_parameter = decay_parameter
        
        self.spatial_prior = SpatialPriorModule(
            inplanes=64, embed_dims=self.spatial_dims)
        
        self.spatial_fuse = nn.ModuleList([SpatialFuseModule(
            query_dim=self.spatial_dims[i], key_dim=self.embed_dim, num_heads=6, 
            qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., drop_path=0., 
            num_classes=self.num_classes, norm_layer=nn.LayerNorm, mask_ratio=mask_ratios[i])
            for i in range(self.stages)])
        
        self.spatial_downsamples = nn.ModuleList([
            DownConv(in_dim=self.spatial_dims[i], 
                     out_dim=self.spatial_dims[i+1])
            for i in range(self.stages-1)])
        
        self.spatial_heads = nn.ModuleList(
            [nn.Conv1d(self.spatial_dims[i], self.embed_dim, 1)
             for i in range(self.stages)])
        self.proj_cls_embed = nn.Linear(self.stages, self.num_classes)
         
           
    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1] - self.num_classes
        N = self.num_patches
        if npatch == N and h == w:
            return self.pos_embed_pat
        
        patch_pos_embed = self.pos_embed_pat
        h0 = h // self.patch_embed.patch_size[0]
        w0 = w // self.patch_embed.patch_size[1]
        dim = x.shape[-1]
        Np = int(math.sqrt(N))
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, Np, Np, dim).permute(0, 3, 1, 2),
            size=(h0, w0),
            mode='bicubic',
            align_corners=False)
        assert h0 == patch_pos_embed.shape[-2] and w0 == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def build_class_tokens(self, z):
        """
        Input: feats -> list [B x C x H^ x W^]
        Return: cls-tokens -> B x Cls x C
        """
        B = z[0].shape[0]
        cls_tokens = []
        for i, feat in enumerate(z):
            out = self.avgpool2d(feat).reshape(B, self.spatial_dims[i], -1) # B x Ct x 1
            # out = self.spatial_heads[i](out)
            cls_tokens.append(out)
        cls_tokens = torch.cat(cls_tokens, dim=-1)   # B x C x 4
        cls_tokens = self.proj_cls_embed(cls_tokens) # B x C x Cls
        cls_tokens = cls_tokens.permute(0, 2, 1).contiguous() # B x Cls x C
        return cls_tokens
         
    def forward_features(self, x):
        """ C: embedding dimension
            Np: num of patches"""
        B, _, H, W = x.shape  # B x 3 x H x W
        sp_feats = self.spatial_prior(x) # list [B x C x H^ x W^]
        x = self.patch_embed(x) # e.g., b x Np x C
        
        if not self.training:
            pos_embed_pat = self.interpolate_pos_encoding(x, H, W)
            x = x + pos_embed_pat
        else: x = x + self.pos_embed_pat
        
        nn_cls_tokens = self.cls_token.expand(B, -1, -1) + self.pos_embed_cls
        cls_tokens = self.build_class_tokens(sp_feats) + nn_cls_tokens
        
        x = torch.cat((cls_tokens, x), dim=1) # Concat input with Nc class tokens
        x = self.pos_drop(x)                  # B x (N') x C, where N' = Nc + Np
        
        attn_weights = []
        for i in range(self.stages):
            for j in range(self.stage_indices[i], self.stage_indices[i+1]):# for each layer
                x, weights_j = self.blocks[j](x) # weights_j: the j-th layer attention weights
                attn_weights.append(weights_j) 
            sp_feats[i] = self.spatial_fuse[i](
                x_query=sp_feats[i], x_key=x) # feature fusion
            
            if i != self.stages - 1:
                z = self.spatial_downsamples[i](sp_feats[i])
                sp_feats[i + 1] = sp_feats[i + 1] + z
    
        x_cls, x_pat =  x[:, 0:self.num_classes], x[:, self.num_classes:]
        # [B x Cls x C], [B x Np x C], list[B x Hd x N' x N'], list[B x C x H^ x W^]
        return x_cls, x_pat, attn_weights, sp_feats
    
    def reshape_patch_tokens(self, patch_tokens, H, W):
        B, _, C = patch_tokens.shape
        Hp = H // self.patch_embed.patch_size[0]
        Wp = W // self.patch_embed.patch_size[1]
        patch_tokens = torch.reshape(patch_tokens, [B, Hp, Wp, C])
        patch_tokens = patch_tokens.permute([0, 3, 1, 2]).contiguous() # B x C x Hp x Wp
        return patch_tokens
    
    def foward_patch_tokens(self, patch_tokens):
        """ MCTformer Plus Weighted Patch Tokens """
        B, Cls, Hp, Wp = patch_tokens.shape
        N = Hp * Wp
        flattened_tokens = patch_tokens.view(B, Cls, -1).permute(0, 2, 1) # B x (Hp x Wp) x Cls
        sorted_tokens, _ = torch.sort(flattened_tokens, -2, descending=True)
        weights = torch.logspace(start=0, end=N-1,steps=N, base=self.decay_parameter).cuda()
        patch_logits = torch.sum(sorted_tokens * weights.unsqueeze(0).unsqueeze(-1), dim=-2) / weights.sum()
        return patch_logits
    
    def forward_attention(self, patch_tokens, weights, fuse_layers=3, patch_refine=True):
        Cls = self.num_classes # simplify code
        B, _, Hp, Wp = patch_tokens.shape
        # Get L attention maps and obtain average along heads 
        weights = torch.stack(weights)              # L x B x Nd x (Cls+Np) x (Cls+Np)
        weights = torch.mean(weights, dim=2)        # L x B x (Cls+Np) x (Cls+Np) with L = 12
        attn_maps = weights[-fuse_layers:].mean(0)  # B x (Cls+Np) x (Cls+Np)
        cls2pat = attn_maps[:, :Cls, Cls:].reshape([B, Cls, Hp, Wp]) # B x Cls x Hp x Wp
        # Class activation map
        feature_map = patch_tokens.detach().clone() # B x Cls x Wp x Hp
        feature_map = F.relu(feature_map)           # With ReLU Activation
        cams = cls2pat * feature_map                #  B x Nc x Hp x Wp
        cams = torch.sqrt(cams)
        
        if patch_refine:
            pat2pat = weights[:, :, Cls:, Cls:]  #  L x B x Np x Np
            pat2pat = torch.sum(pat2pat, dim=0)  # B x Np x Np
            Hf, Wf = cams.shape[2:]
            cams = torch.matmul(
                    pat2pat.unsqueeze(1),    # B x 1 x Np x Np
                    cams.view(B, Cls, -1, 1) # B x Cls x Np x 1
            ).reshape(B, Cls, Hf, Wf)
            
        return cams
    
    def forward(self, x, return_att=False, n_layers=12):
        H, W = x.shape[2:]
        class_tokens, patch_tokens, weights, _ = self.forward_features(x)
        cls_logits = class_tokens.mean(-1)      # B x Nc
        
        patch_tokens = self.reshape_patch_tokens(patch_tokens, H, W) # B x C x Hp x Wp
        patch_tokens = self.head(patch_tokens)                       # B x Cls x Hp x Wp
        patch_logits = self.foward_patch_tokens(patch_tokens)
        
        if return_att:
            cams = self.forward_attention(
                patch_tokens, weights, fuse_layers=n_layers)
            return cls_logits, cams
        
        else:
            return cls_logits, patch_logits


class MCTGCAM(MCTG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward_attention(self, patch_tokens, weights, fuse_layers=12):
        Cls = self.num_classes # simplify code
        B, _, Hp, Wp = patch_tokens.shape
        # Get L attention maps and obtain average along heads 
        weights = torch.stack(weights)              # L x B x Nd x (Cls+Np) x (Cls+Np)
        weights = torch.mean(weights, dim=2)        # L x B x (Cls+Np) x (Cls+Np) with L = 12
        attn_maps = weights[-fuse_layers:].mean(0)  # B x (Cls+Np) x (Cls+Np)
        cls2pat = attn_maps[:, :Cls, Cls:].reshape([B, Cls, Hp, Wp]) # B x Cls x Hp x Wp
        # Class activation map
        feature_map = patch_tokens.detach().clone() # B x Cls x Wp x Hp
        feature_map = F.relu(feature_map)           # With ReLU Activation
        
        pat2pat = weights[:, :, Cls:, Cls:]         #  L x B x Np x Np
        cams = cls2pat * feature_map                #  B x Nc x Hp x Wp
        cams = torch.sqrt(cams)
        
        return cams, pat2pat
    
    def forward(self, x):
        B, _, H, W = x.shape   # batch size=2
        Cls = self.num_classes
        _, patch_tokens, weights, _ = self.forward_features(x)
        patch_tokens = self.reshape_patch_tokens(patch_tokens, H, W)  # B x C x Hp x Wp
        patch_tokens = self.head(patch_tokens)  # B x Cls x Hp x Wp
        
        cams, pat2pat = self.forward_attention(
            patch_tokens, weights, fuse_layers=12)
        
        pat2pat = torch.sum(pat2pat, dim=0) # B x Np x Np
        Hf, Wf = cams.shape[2:]
        cams = torch.matmul(
                pat2pat.unsqueeze(1),    # B x 1 x Np x Np
                cams.view(B, Cls, -1, 1) # B x Cls x Np x 1
            ).reshape(B, Cls, Hf, Wf)
        return cams
      
        