import clip
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class CLIP_MODEL(nn.Module):
    def __init__(self, nef, sent_type,device_cuda):
        super(CLIP_MODEL, self).__init__()
        """if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 512  # define a uniform ranker"""
        self.nef = nef   
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.device = torch.device('cuda',1)
        self.device =device_cuda
        self.model, self.preprocess = clip.load("RN101", device=self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.float()
        self.model.eval()
        self.sent_type = sent_type
        self.MLP = MLP_CLIP(nef=self.nef, sent_type=self.sent_type, device=self.device)

    def image_encode(self, x):
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        image_code, cnn_feature = self.model.encode_image(x)  # last , layer3
        #print("31imagecode %s cnn_feat %s"%(image_code.shape,cnn_feature.shape)) 
        # RN101: image_code size ([batch_size, 512]); cnn_feature size ([batch_size, 1024, 14, 14])
        #31imagecode torch.Size([1, 512]) cnn_feat torch.Size([1, 1024, 14, 14])
        # 34imagecode torch.Size([1, 512]) cnn_feat torch.Size([1, 512, 14, 14])

        image_code, cnn_feature = self.MLP.image_feature(image_code, cnn_feature) 
        #print("34imagecode %s cnn_feat %s"%(image_code.shape,cnn_feature.shape))
        # cnn_feature = cnn_feature.type(torch.FloatTensor).cuda()
        # cnn_feature = self.emb_features(cnn_feature)
        # # image_code = image_code.type(torch.FloatTensor).cuda()
        # image_code = self.emb_cnn_code(image_code)
        return cnn_feature, image_code

    def sent_encode(self, text):
        text_features, word_embs = self.model.encode_text(text)
        word_embs, text_features = self.MLP.sent_feature(word_embs, text_features)
        # # RN101: text_features size ([batch_size, 512]); word_embs size ([batch_size, 512, 77])
        # text_features = text_features.type(torch.FloatTensor).cuda()
        # word_embs = word_embs.type(torch.FloatTensor).cuda()
        # word_embs = word_embs.permute(0, 2, 1)
        # word_embs = self.dense_word(word_embs)
        # word_embs = self.LayerNorm(word_embs)
        # word_embs = word_embs.permute(0, 2, 1)
        # text_features = self.dense_sent(text_features)
        return word_embs, text_features

    def forward(self, image_code, text_features):
        print("53imagesize%s"%image_code.shape)
        logits_per_image, logits_per_text = self.model.feature_to_logit(image_code, text_features)
    
        return logits_per_image, logits_per_text
        
        
        
class MLP_CLIP(nn.Module):
    def __init__(self, nef, sent_type, device):
        super(MLP_CLIP, self).__init__()
        self.sent_type = sent_type
        self.nef = nef
        self.define_module()
        self.init_trainable_weights()
        #self.device_cuda=torch.device('cuda',1)
        self.device_cuda=device
        
    """def conv1x1(in_planes, out_planes, bias=False):
        "1x1 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)"""
        
    def define_module(self):
        self.emb_features = nn.Conv2d(1024, self.nef, kernel_size=1, stride=1, padding=0,bias = False)
        # conv1x1(1024, self.nef)

        self.dense_word = nn.Linear(512, self.nef)
        if self.sent_type == 'CLIP_FT':
            self.dense_sent = nn.Linear(512, self.nef)
            self.emb_cnn_code = nn.Linear(512, self.nef)
        self.LayerNorm = torch.nn.LayerNorm(512, eps=1e-12)
        # self.dense_word_2 = nn.Linear(512, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        if self.sent_type == 'CLIP_FT':
            self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def image_feature(self, image_code, cnn_feature):
        cnn_feature = cnn_feature.type(torch.FloatTensor).to(self.device_cuda)
        cnn_feature = self.emb_features(cnn_feature)
        # image_code = image_code.type(torch.FloatTensor).cuda()
        if self.sent_type == 'CLIP_FT':
            image_code = self.emb_cnn_code(image_code)
        return image_code, cnn_feature

    def sent_feature(self, word_embs, text_features):
        text_features = text_features.type(torch.FloatTensor).to(self.device_cuda)
        word_embs = word_embs.type(torch.FloatTensor).to(self.device_cuda)
        word_embs = word_embs.permute(0, 2, 1)
        word_embs = self.dense_word(word_embs)
        word_embs = self.LayerNorm(word_embs)
        # word_embs = self.dense_word_2(word_embs)
        word_embs = word_embs.permute(0, 2, 1)
        if self.sent_type == 'CLIP_FT':
            text_features = self.dense_sent(text_features)
        return word_embs, text_features
        
