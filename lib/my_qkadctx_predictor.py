import torch
import torch.nn as nn
from torch.nn.functional import normalize, relu
from torch.nn.utils.rnn import pad_packed_sequence
from torchvision import models
import torch.nn.functional as F


__all__ = ['VanillaPredictor', 'LTRPredictor', 'RNNBCEPredictor', 'AttVanillaPredictor', 'RankPredictor',
           'SquashedRankPredictor']


def make_resnet_layer4():
    downsample = nn.Sequential(
        nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(2048)
    )
    layers = [models.resnet.Bottleneck(1024, 512, 1, downsample)]
    for _ in range(2):
        layers.append(models.resnet.Bottleneck(2048, 512))
    return nn.Sequential(*layers)








# sip qk add  its right
#/home/wj/code/ref_nms/output/my_sipqk_ad_refcoco_0903233435_b.pth
class AttVanillaPredictorV2(nn.Module):

    def __init__(self, att_dropout_p, rank_dropout_p):
        super(AttVanillaPredictorV2, self).__init__()
        # Head network with pretrained weight
        self.head = make_resnet_layer4()
        #self.head1= make_resnet_layer4()
        # Layers of new branch  

       
        #print("266......")
        self.vis_a_fck = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True)
        )
        #print("272......")
        
        """
        self.vis_a_fcv = nn.Sequential(
            nn.Linear(2053, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True)
        )
        """
        
        self.mapping_sent = nn.Sequential(
          nn.Linear(768,512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Dropout(p=0.1),  # follow one stage
          nn.Linear(512, 512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
        )
        
        #print("298......")
        
        #self.rank_fc_gt = nn.Linear(512, 1, bias=True)
        #print("300......")
       
    def forward(self, roi_feat, packed_sent_feat,bert_feats,lfeat):
        """
        Args:
            roi_feat: [N, R, 1024, 7, 7].
            packed_sent_feat: `PackedSequence` object, should be moved to proper device in advance.
            sent_feats: [N,R,768]??

        Returns:
            ref_score: [N, R].

        """
        # Extract visual feature with ResNet Conv-head
        #print("276roishape%s"%(roi_feat.shape,)) # [1, 305, 1024, 7, 7]) N is batch, R is region number   [32, 32, 1024, 7, 7]
  
        N, R, *_ = roi_feat.shape
        head_feat = self.head(roi_feat.reshape(N * R, 1024, 7, 7)).reshape(N, R, 2048, 7, 7)
        head_pool = head_feat.mean(dim=(3, 4))  # [N, R, 2048]
        

        # add loc 
        #head_feat1 = self.head1(roi_feat.reshape(N * R, 1024, 7, 7)).reshape(N, R, 2048, 7, 7)
        #head_pool1 = head_feat1.mean(dim=(3, 4))  # [N, R, 2048]
        #bert_feats =bert_feats.detach()

        #feat_cat_loc = torch.cat([head_pool, lfeat], 2)  # N R 2048+5 = N R 2053
        feat_cat_loc_mapk = self.vis_a_fck(head_pool) # N , R, 512
        sent_feat_map = self.mapping_sent(bert_feats)# 512
        sent_feat_map= torch.unsqueeze(sent_feat_map,1)
        sent_feat_map0 =sent_feat_map.expand(-1, R, -1) # N R 512
        cos_sim = F.cosine_similarity(feat_cat_loc_mapk, sent_feat_map0, dim=-1) # [32,32]
        #q_mul_k = feat_cat_loc_mapk * sent_feat_map0 
        #print("117............cos_sim%s"%(cos_sim.shape,))
        #score_gt_neg0 = torch.sum(q_mul_k, dim=2)# N R 512
        #q_mul_k = torch.softmax(q_mul_k,dim=2) # N R 512
        #feat_cat_loc_mapv = self.vis_a_fcv(feat_cat_loc)
        #sent_img_qkv= q_mul_k*feat_cat_loc_mapv # N R 512
        #sent_img_qkv = relu(q_mul_k, inplace=True)
        #sent_img_qkv = normalize(sent_img_qkv, dim=2)    # [N, R, 512]
        #score_gt_neg = self.rank_fc_gt(sent_img_qkv)          # [N, R, 1]
        
        #score_gt_neg0 = score_gt_neg.squeeze(2)               # [N, R]
        #score_gt_neg = score_gt_neg0/0.06
        
        
        
        # Extract word feature with RNN

        # add loc and gt 

        return cos_sim



class SquashedRankPredictor(nn.Module):

    def __init__(self, dropout_p=0.5):
        super(SquashedRankPredictor, self).__init__()
        self.head = make_resnet_layer4()
        self.word_fc = nn.Linear(300, 300, bias=True)
        self.head_fc = nn.Linear(2048, 300, bias=True)
        self.rank_fc = nn.Linear(300, 1, bias=True)
        self.head_dropout = nn.Dropout(p=dropout_p, inplace=True)

    def forward(self, roi_feat, word_feat):
        """

        Args:
            roi_feat: [N, R, 1024, 7, 7].
            word_feat: [N, S, 300].

        Returns:
            max_rank_score: [N, R].
            max_idx: [N, R].

        """
        N, R, *_ = roi_feat.shape
        head_feat = self.head(roi_feat.reshape(N*R, 1024, 7, 7)).reshape(N, R, 2048, 7, 7)
        head_pool = head_feat.mean(dim=(3, 4))                 # [N, R, 2048]
        head_pool = self.head_dropout(head_pool)               # [N, R, 2048]
        head_mapped = self.head_fc(head_pool)                  # [N, R, 300]
        word_mapped = self.word_fc(word_feat)                  # [N, S, 300]
        head_expanded = head_mapped.unsqueeze(2)               # [N, R, 1, 300]
        word_expanded = word_mapped.unsqueeze(1)               # [N, 1, S, 300]
        feat_merged = head_expanded * word_expanded            # [N, R, S, 300]
        feat_merged = normalize(feat_merged, dim=3)            # [N, R, S, 300]
        rank_score = self.rank_fc(feat_merged).squeeze(dim=3)  # [N, R, S]
        max_rank_score, max_idx = rank_score.max(dim=2)        # [N, R], [N, R]
        sigmoid_rank_score = torch.sigmoid(max_rank_score)
        return sigmoid_rank_score, max_idx
        # lower_bound = torch.zeros_like(max_rank_score)
        # upper_bound = torch.ones_like(max_rank_score)
        # scaled_rank_score = torch.min(torch.max(0.25 * max_rank_score + 0.5, lower_bound), upper_bound)
        # return scaled_rank_score, max_idx

    def rank_parameters(self):
        return list(self.head_fc.parameters()) + list(self.word_fc.parameters()) + list(self.rank_fc.parameters())

    def named_rank_parameters(self):
        return list(self.head_fc.named_parameters()) + list(self.word_fc.named_parameters()) \
               + list(self.rank_fc.named_parameters())


if __name__ == '__main__':

    predictor = AttVanillaPredictorV2(0.5, 0.5)
    for k, v in predictor.named_parameters():
        print(k, ':', v.shape)
