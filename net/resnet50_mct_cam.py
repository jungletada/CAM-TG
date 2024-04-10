import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50
from timm.models.layers import trunc_normal_
from net.mct_modules import Semantic2FeatureBlock, SpatialPriorModule


class MCTResNet50_Cls(nn.Module):
    def __init__(self, stride=16, n_classes=20):
        super(MCTResNet50_Cls, self).__init__()
        if stride == 16:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,self.resnet50.layer1)
        else:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 1, 1), dilations=(1, 1, 2, 2))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,self.resnet50.layer1)
        
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.n_classes = n_classes
        self.n_stages = 4
        self.spm_channel = 320
        self.channels = [256, 512, 1024, 2048]
        self.cls_token = nn.Parameter(torch.zeros(1, self.n_classes, self.spm_channel))
        trunc_normal_(self.cls_token, std=.02)
        
        self.classifier = nn.Conv2d(2048, n_classes, 1, bias=False)
        self.spm = SpatialPriorModule(64, embed_dims=[self.spm_channel] * self.n_stages)
        
        n_knn = [15, 12, 9, 6]
        dilations = [1, 2, 2, 2]
        dpr = [x.item() for x in torch.linspace(0, 0.1, self.n_stages)]
        
        self.sem_gnn_layer = nn.Sequential(*[
            Semantic2FeatureBlock(
                self.spm_channel, 
                num_classes=self.n_classes, 
                kernel_size=n_knn[i], 
                dilation=dilations[i], 
                drop_path=dpr[i], 
                num_heads=4,
                proj_drop=0.)
            for i in range(self.n_stages)])

        self.proj_fusion = nn.Sequential(
            nn.Conv2d(self.spm_channel * self.n_stages, self.channels[-1], 1),
            nn.BatchNorm2d(self.channels[-1]),
            nn.GELU())
        
        self.down_sample = nn.ModuleList([
            nn.Conv2d(self.spm_channel, self.spm_channel, 3, 2, 1)
            for i in range(self.n_stages - 1)])
        
        self.proj_scaled_x = nn.ModuleList([
            nn.Conv2d(self.channels[i], self.spm_channel, 3, 1, 1)
            for i in range(self.n_stages)])
        
        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier, self.spm, self.sem_gnn_layer, 
                                          self.proj_fusion, self.proj_scaled_x, self.down_sample])

    def forward_sem(self, x_spm, x_cls, x, stage):
        scaled_x = F.interpolate(x, size=x_spm[stage].shape[2:], mode='bilinear')
        x_fuse = x_spm[stage] + self.proj_scaled_x[stage](scaled_x)
        x_cls, x_fuse = self.sem_gnn_layer[stage](x_cls, x_fuse)
        
        if stage != self.n_stages - 1:
            x_fuse = self.down_sample[stage](x_fuse)
            x_spm[stage+1] = x_spm[stage+1] + x_fuse
        return x_cls, x_spm
    
    def feature_fusion(self, x_spm, x):
        for i in range(self.n_stages):
            x_spm[i] = F.interpolate(x_spm[i], size=x.shape[2:], mode='bilinear')
        fusion = torch.cat(x_spm, dim=1)
        fusion = x + self.proj_fusion(fusion)
        return fusion
    
    def forward(self, x):
        x_spm = self.spm(x)
        x_cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.stage1(x)
        x_cls, x_spm = self.forward_sem(x_spm, x_cls, x, stage=0)
    
        x = self.stage2(x)
        x_cls, x_spm = self.forward_sem(x_spm, x_cls, x, stage=1)
        
        x = self.stage3(x)
        x_cls, x_spm = self.forward_sem(x_spm, x_cls, x, stage=2)
        
        x = self.stage4(x)
        x_cls, x_spm = self.forward_sem(x_spm, x_cls, x, stage=3)
        
        x = self.feature_fusion(x_spm, x)
        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)
        x_cls = x_cls.mean(dim=-1)
        return x_cls, x

    def train(self, mode=True):
        super(MCTResNet50_Cls, self).train(mode)
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
    

class MCTResNet50_CAM(MCTResNet50_Cls):
    def __init__(self, stride=16, n_classes=20):
        super(MCTResNet50_CAM, self).__init__(stride=stride, n_classes=n_classes)

    def forward(self, x, separate=False):
        x = F.conv2d(x, self.classifier.weight)
        x_spm = self.spm(x)
        x_cls = self.cls_token.expand(x.shape[0], -1, -1)
        
        x = self.stage1(x)
        x_cls, x_spm = self.forward_sem(x_spm, x_cls, x, stage=0)
    
        x = self.stage2(x)
        x_cls, x_spm = self.forward_sem(x_spm, x_cls, x, stage=1)
        
        x = self.stage3(x)
        x_cls, x_spm = self.forward_sem(x_spm, x_cls, x, stage=2)
        
        x = self.stage4(x)
        x_cls, x_spm = self.forward_sem(x_spm, x_cls, x, stage=3)
        
        x = self.feature_fusion(x_spm, x)
        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)
        return x