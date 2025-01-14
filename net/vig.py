# 2022.10.31-Changed for building ViG model
# Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from net.gcn_lib import Grapher, act_layer # net.

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'gnn_patch16_224': _cfg(
        crop_pct=0.9, input_size=(3, 224, 224),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'gnn_patch16_512': _cfg(
        crop_pct=0.9, input_size=(3, 512, 512),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),}


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Downsample = 16 stride
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=512, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim))

    def forward(self, x):
        x = self.convs(x)
        return x


class DeepGCN(nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        self.opt = opt
        groups = 4
        k = opt.k
        conv = opt.conv
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        drop_path = opt.drop_path
        stochastic = opt.use_stochastic
        self.channels = opt.n_filters
        self.n_blocks = opt.n_blocks
        self.input_size = opt.input_size
        self.num_classes = opt.n_classes
        self.stem = Stem(out_dim=self.channels, act=act)

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        
        Hp, Wp = self.input_size[0]//16, self.input_size[1]//16
        self.pos_embed = nn.Parameter(torch.zeros(1, self.channels, Hp, Wp))
        max_dilation = Hp * Wp // max(num_knn)

        if opt.use_dilation:
            self.backbone = Seq(*[Seq(
                Grapher(self.channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, groups, norm, 
                        bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                FFN(self.channels, self.channels * 4, act=act, drop_path=dpr[i])) 
                    for i in range(self.n_blocks)])
        else:
            self.backbone = Seq(*[Seq(
                Grapher(self.channels, num_knn[i], 1, conv, act, norm,
                    bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                FFN(self.channels, self.channels * 4, act=act, drop_path=dpr[i])) 
                    for i in range(self.n_blocks)])
        
        self.prediction = nn.Conv2d(self.channels, opt.n_classes, 1, bias=False)
        
        self.model_init()
        
    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
    
    def forward(self, inputs):
        x = self.stem(inputs)
        x = x + self.pos_embed
        for i in range(self.n_blocks):
            x = self.backbone[i](x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.prediction(x)
        x = x.squeeze(-1).squeeze(-1)
        return x
            

class OptInit:
    def __init__(self, num_knn, num_classes, drop_path_rate, drop_rate, input_size):
        self.k = num_knn # neighbor num (default:9)
        self.conv = 'mr' # graph conv layer {edge, mr}
        self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
        self.norm = 'batch' # batch or instance normalization {batch, instance}
        self.bias = True # bias of conv layer True or False
        self.n_blocks = 12 # number of basic blocks in the backbone
        self.n_filters = 192 # number of channels of deep features
        self.n_classes = num_classes # Dimension of out_channels
        self.dropout = drop_rate # dropout rate
        self.use_dilation = True # use dilated knn or not
        self.epsilon = 0.2 # stochastic epsilon for gcn
        self.use_stochastic = False # stochastic for gcn, True or False
        self.drop_path = drop_path_rate
        self.input_size = to_2tuple(input_size)
            

def vig_variant_opt_init(model_type, **kwargs):        
    opt = OptInit(**kwargs)    
    model_type = model_type.lower()  
    if model_type == 'ti' or model_type == 't':
        return opt
    elif model_type == 's':
        opt.n_blocks = 16
        opt.n_filters = 320
        return opt
    elif model_type == 'b': 
        opt.n_blocks = 16
        opt.n_filters = 640
        return opt
    else:
        raise ValueError(f'{model_type} is not supported')
     
     
# def vig_ti_224_gelu(pretrained=False, **kwargs):
#     opt = vig_variant_opt_init(model_type='Ti', **kwargs)
#     model = DeepGCN(opt)
#     model.default_cfg = default_cfgs['gnn_patch16_224']
#     return model


# def vig_s_224_gelu(pretrained=False, **kwargs):
#     opt = vig_variant_opt_init(model_type='S', **kwargs)
#     model = DeepGCN(opt)
#     model.default_cfg = default_cfgs['gnn_patch16_224']
#     return model


# def vig_b_224_gelu(pretrained=False, **kwargs):
#     opt = vig_variant_opt_init(model_type='B', **kwargs)
#     model = DeepGCN(opt)
#     model.default_cfg = default_cfgs['gnn_patch16_224']
#     return model


class Net(DeepGCN):
    def __init__(self, **kwargs):
        opt=vig_variant_opt_init(model_type='S', **kwargs)
        super().__init__(opt)
        
    def forward(self, x):
        pass
    
        
class ViG_CAM(DeepGCN):
    def __init__(self, model_type, **kwargs):
        opt = vig_variant_opt_init(model_type, **kwargs)
        super(ViG_CAM, self).__init__(opt)
    
    def resize_image(self, H, W):
        """
            Resize an image such that the shorter side is 224 pixels, 
            and the other side is scaled proportionally.
        """
        target_size = self.input_size[0]
        if H < W:
            scale_factor = target_size / H
            new_H = target_size
            new_W = int(W * scale_factor)
        else:
            scale_factor = target_size / W
            new_H = int(H * scale_factor)
            new_W = target_size

        return new_H, new_W
    
    def forward(self, inputs):
        H, W = inputs.shape[2:]
        # if min(H, W) <= self.input_size[0]:
        #     new_H, new_W = self.resize_image(H, W)
        #     inputs = F.interpolate(
        #         input=inputs,
        #         size=(new_H, new_W),
        #         mode='bilinear',
        #         align_corners=False)
            
        x = self.stem(inputs)
        pos_embed = F.interpolate(
            self.pos_embed,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False)
        
        x = x + pos_embed
        
        for i in range(self.n_blocks):
            x = self.backbone[i](x)
        x = self.prediction(x)
        x = F.relu(x.detach().clone())
        return x


if __name__ == '__main__':
    kwargs = {
        'num_classes': 80, 
        'drop_path_rate': 0.0, 
        'drop_rate': 0.0, 
        'num_knn': 9, 
        'input_size': 224,}
    model = vig_s_224_gelu(**kwargs)
    model.eval()

    x = torch.randn(1, 3, 196, 324)
    # out = model(x)
    # print(out.shape)
    
    cam_model = ViG_CAM(model_type='s', **kwargs)
    cam_model.eval()
    
    c = cam_model(x)
    print(c.shape)