import torch
import torch.nn as nn
from torch.nn.functional import normalize, relu
from torch.nn.utils.rnn import pad_packed_sequence
from torchvision import models


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


class VanillaPredictor(nn.Module):

    def __init__(self):
        super(VanillaPredictor, self).__init__()
        # Head network with pretrained weight
        self.head = make_resnet_layer4()
        
        # Prediction branch & bbox regression branch from m-rcnn
        self.cls_score_net = nn.Linear(2048, 81, bias=True)
        self.bbox_pred_net = nn.Linear(2048, 81*4, bias=True)
        # Layers of new branch
        self.ref_cls_fc1 = nn.Linear(2048, 300, bias=True)
        self.ref_cls_fc2 = nn.Linear(300, 1, bias=True)

    def forward(self, roi_feats, word_feats):
        """

        Args:
            roi_feats: [N, R, 1024, 7, 7].
            word_feats: [N, S, 300].

        Returns:
            max_ref_score: [N, R].
            max_idx: [N, R].
            cls_score: [N, R, 81].
            bbox_pred: [N, R, 81*4].

        """
        N, R, *_ = roi_feats.shape
        head_feats = self.head(roi_feats.reshape(N*R, 1024, 7, 7)).reshape(N, R, 2048, 7, 7)
        head_pool = head_feats.mean(dim=(3, 4))         # [N, R, 2048]
        head_mapped = self.ref_cls_fc1(head_pool)       # [N, R, 300]
        head_reshaped = head_mapped.unsqueeze(2)        # [N, R, 1, 300]
        word_reshaped = word_feats.unsqueeze(1)         # [N, 1, S, 300]
        feat_merged = head_reshaped*word_reshaped       # [N, R, S, 300]
        feat_merged = normalize(feat_merged, dim=3)     # [N, R, S, 300]
        ref_score = self.ref_cls_fc2(feat_merged).squeeze(dim=3)  # [N, R, S]
        max_ref_score, max_idx = ref_score.max(dim=2)   # [N, R]
        cls_score = self.cls_score_net(head_pool)       # [N, R, 81]
        bbox_pred = self.bbox_pred_net(head_pool)       # [N, R, 81*4]
        return max_ref_score, max_idx, cls_score, bbox_pred


class LTRSubNetwork(nn.Module):
    def __init__(self, sent_feat_dim):
        super(LTRSubNetwork, self).__init__()
        self.head_fc = nn.Linear(2048, sent_feat_dim, bias=True)
        self.rank_fc = nn.Linear(sent_feat_dim, 1, bias=False)

    def forward(self, sent_feat, head_feat):
        """

        Args:
            sent_feat: Sentence feature: [sent_num, sent_feat_dim].
            head_feat: Head network output: [roi_num, head_feat].

        Returns:
            rank_score: Rank score of shape [sent_num, roi_num].

        """
        mapped_head_feat = self.head_fc(head_feat).unsqueeze(0)  # [1, roi_num, sent_feat_dim]
        mapped_head_feat = relu(mapped_head_feat, inplace=True)  # [1, roi_num, sent_feat_dim]
        reshaped_sent_feat = sent_feat.unsqueeze(1)              # [sent_num, 1, sent_feat_dim]
        merged_feat = reshaped_sent_feat * mapped_head_feat      # [sent_num, roi_num, sent_feat_dim]
        rank_score = self.rank_fc(merged_feat).squeeze(dim=2)    # [sent_num, roi_num]
        return rank_score


class LTRPredictor(nn.Module):

    RNN_H_DIM = 512

    def __init__(self):
        super(LTRPredictor, self).__init__()
        # Head network with pretrained weight
        self.head_net = make_resnet_layer4()
        # Prediction branch & Regression branch from M-RCNN
        self.cls_score_net = nn.Linear(2048, 81, bias=True)
        self.bbox_pred_net = nn.Linear(2048, 81 * 4, bias=True)
        # LTR sub-network
        self.LTR_net = LTRSubNetwork(self.RNN_H_DIM)
        # Sentence processing network
        self.rnn = nn.GRU(300, self.RNN_H_DIM, bidirectional=True)
        # self.stats = {
        #     'head_feat_mean': None, 'head_feat_std': None, 'head_feat_norm': None,
        #     'sent_feat_mean': None, 'sent_feat_std': None, 'sent_feat_norm': None
        # }

    def forward(self, roi_feat, packed_sent_feat):
        """

        Args:
            roi_feat: ROI features, [roi_num, 1024, 7, 7].
            packed_sent_feat: `PackedSequence` object, should be moved to proper device in advance.

        Returns:
            rank_score: Rank score of shape [sent_num, roi_num].

        """
        # Extract sentence feature with RNN
        _, h_n = self.rnn(packed_sent_feat)     # [2, sent_num, sent_feat_dim]
        sent_feat = h_n.sum(dim=0)              # [sent_num, sent_feat_dim]
        # Extract head feature
        head_feat = self.head_net(roi_feat)     # [roi_num, 2048, 7, 7]
        head_feat = head_feat.mean(dim=(2, 3))  # [roi_num, 2048]
        # Rank ROIs
        rank_score = self.LTR_net(sent_feat, head_feat)  # [sent_num, roi_num]
        # self.stats['head_feat_mean'] = head_feat.mean().item()
        # self.stats['head_feat_std']  = head_feat.std().item()
        # self.stats['head_feat_norm'] = head_feat.abs().mean().item()
        # self.stats['sent_feat_mean'] = sent_feat.mean().item()
        # self.stats['sent_feat_std']  = sent_feat.std().item()
        # self.stats['sent_feat_norm'] = sent_feat.abs().mean().item()
        return rank_score,                      # NOTE that the return tuple is a one-element tuple


class RNNBCEPredictor(nn.Module):

    RNN_H_DIM = 512

    def __init__(self):
        super(RNNBCEPredictor, self).__init__()
        # Head network with pretrained weight
        self.head = make_resnet_layer4()
        # Prediction branch & bbox regression branch from m-rcnn
        self.cls_score_net = nn.Linear(2048, 81, bias=True)
        self.bbox_pred_net = nn.Linear(2048, 81*4, bias=True)
        # Layers of new branch
        self.ref_cls_fc1 = nn.Linear(2048, self.RNN_H_DIM, bias=True)
        self.ref_cls_fc2 = nn.Linear(self.RNN_H_DIM, 1, bias=True)
        self.rnn = nn.GRU(300, self.RNN_H_DIM, bidirectional=True)

    def forward(self, roi_feat, packed_sent_feat):
        """

        Args:
            roi_feat: [N, R, 1024, 7, 7].
            packed_sent_feat: `PackedSequence` object, should be moved to proper device in advance.

        Returns:
            ref_score: [N, R].

        """
        # Extract sentence feature with RNN
        _, h_n = self.rnn(packed_sent_feat)  # [2, N, RNN_H_DIM]
        sent_feat = h_n.sum(dim=0)           # [N, RNN_H_DIM]

        N, R, *_ = roi_feat.shape
        head_feat = self.head(roi_feat.reshape(N * R, 1024, 7, 7)).reshape(N, R, 2048, 7, 7)
        head_pool = head_feat.mean(dim=(3, 4))          # [N, R, 2048]
        head_mapped = self.ref_cls_fc1(head_pool)       # [N, R, RNN_H_DIM]
        sent_feat = sent_feat.unsqueeze(1)              # [N, 1, RNN_H_DIM]
        feat_merged = head_mapped * sent_feat           # [N, R, RNN_H_DIM]
        feat_merged = normalize(feat_merged, dim=2)     # [N, R, RNN_H_DIM]
        ref_score = self.ref_cls_fc2(feat_merged)       # [N, R, 1]
        ref_score = ref_score.squeeze(2)                # [N, R]

        debug_info = {
            # 'head_feat_norm': head_feat.norm(p=2).item(),
            # 'sent_feat_norm': sent_feat.norm(p=2).item(),
            'ref_score_mean': ref_score.mean().item(),
            'ref_score_std': ref_score.std().item()
        }

        return ref_score, debug_info


class AttVanillaPredictor(nn.Module):

    def __init__(self, att_dropout_p, vis_dropout_p, rank_dropout_p):
        super(AttVanillaPredictor, self).__init__()
        # Head network with pretrained weight
        self.head = make_resnet_layer4()
        # Layers of new branch
        self.att_fc = nn.Linear(3072, 1, bias=True)
        self.vis_fc = nn.Linear(2048, 1024, bias=True)
        self.rank_fc = nn.Linear(1024, 1, bias=True)
        self.att_dropout = nn.Dropout(p=att_dropout_p, inplace=True)
        self.vis_dropout = nn.Dropout(p=vis_dropout_p, inplace=True)
        self.rank_dropout = nn.Dropout(p=rank_dropout_p, inplace=True)
        self.rnn = nn.GRU(300, 512, bidirectional=True, batch_first=True)

    def forward(self, roi_feat, packed_sent_feat):
        """

        Args:
            roi_feat: [N, R, 1024, 7, 7].
            packed_sent_feat: `PackedSequence` object, should be moved to proper device in advance.

        Returns:
            ref_score: [N, R].

        """
        # Extract word feature with RNN
        packed_output, _ = self.rnn(packed_sent_feat)
        rnn_out, _ = pad_packed_sequence(packed_output, batch_first=True)  # [N, S, 1024], [N]
        #print("rnn_out%s"%(rnn_out.shape))
        S = rnn_out.size(1)
        # Extract visual feature with ResNet Conv-head
        N, R, *_ = roi_feat.shape
        head_feat = self.head(roi_feat.reshape(N * R, 1024, 7, 7)).reshape(N, R, 2048, 7, 7)
        head_pool = head_feat.mean(dim=(3, 4))          # [N, R, 2048]
        # Cross-modal attention over words
        expanded_rnn_out = rnn_out.unsqueeze(dim=1).expand(-1, R, -1, -1)      # [N, R, S, 1024]
        expanded_head_pool = head_pool.unsqueeze(dim=2).expand(-1, -1, S, -1)  # [N, R, S, 2048]
        att_merged_feat = torch.cat((expanded_rnn_out, expanded_head_pool), dim=3)  # [N, R, S, 3072]
        att_score = self.att_fc(self.att_dropout(att_merged_feat))   # [N, R, S, 1]
        att_weight = torch.softmax(att_score, dim=2)                 # [N, R, S, 1]
        sent_feat = torch.sum(att_weight * expanded_rnn_out, dim=2)  # [N, R, 1024]
        # Compute rank score
        head_mapped = self.vis_fc(self.vis_dropout(head_pool))    # [N, R, 1024]
        feat_merged = head_mapped * sent_feat          # [N, R, 1024]
        feat_merged = relu(feat_merged, inplace=True)  # [N, R, 1024]
        feat_merged = normalize(feat_merged, dim=2)    # [N, R, 1024]
        ref_score = self.rank_fc(self.rank_dropout(feat_merged))  # [N, R, 1]
        ref_score = ref_score.squeeze(2)               # [N, R]

        return ref_score,
"""
## only image weighted by sent not sent weighted by image
class AttVanillaPredictorV2(nn.Module):

    def __init__(self, att_dropout_p, rank_dropout_p):
        super(AttVanillaPredictorV2, self).__init__()
        # Head network with pretrained weight
        self.head = make_resnet_layer4()
        #self.head1= make_resnet_layer4()
        # Layers of new branch
        self.vis_a_fc = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True)
        )
        self.vis_r_fc = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True)
        )
        self.att_fc = nn.Linear(1024, 1, bias=True)
        self.rank_fc = nn.Linear(1024, 1, bias=True)
        self.att_dropout = nn.Dropout(p=att_dropout_p, inplace=True)
        self.rank_dropout = nn.Dropout(p=rank_dropout_p, inplace=True)
        self.rnn = nn.GRU(300, 256, bidirectional=True, batch_first=True)
        #print("266......")
        self.vis_a_fc1 = nn.Sequential(
            nn.Linear(2053, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True)
        )
        #print("272......")
 
        #print("280......")
        self.se_img = nn.Sequential(
            #nn.AdaptiveAvgPool2d((1,1)),
            nn.Linear(1024,64,bias=True),
            
            nn.ReLU(),
            nn.Linear(64,512,bias=True),
            nn.Sigmoid()
        )

        self.rank_fc_gt = nn.Linear(512, 1, bias=True)
        #print("300......")
       
    def forward(self, roi_feat, packed_sent_feat,bert_feats,lfeat):
        """"""
        Args:
            roi_feat: [N, R, 1024, 7, 7].
            packed_sent_feat: `PackedSequence` object, should be moved to proper device in advance.
            sent_feats: [N,R,768]??

        Returns:
            ref_score: [N, R].

        """"""
        # Extract visual feature with ResNet Conv-head
        #print("276roishape%s"%(roi_feat.shape,)) # [1, 305, 1024, 7, 7]) N is batch, R is region number   [32, 32, 1024, 7, 7]
        #print("313bert_feats%s"%(bert_feats.shape,))
        #bert_feats = bert_feats.detach()
        #print("315bert_feats%s"%(bert_feats.shape,))
        #print("276packedshape%s"%len(packed_sent_feat))  # 4?
        N, R, *_ = roi_feat.shape
        head_feat = self.head(roi_feat.reshape(N * R, 1024, 7, 7)).reshape(N, R, 2048, 7, 7)
        head_pool = head_feat.mean(dim=(3, 4))  # [N, R, 2048]
        

        # add loc 
        #head_feat1 = self.head1(roi_feat.reshape(N * R, 1024, 7, 7)).reshape(N, R, 2048, 7, 7)
        #head_pool1 = head_feat1.mean(dim=(3, 4))  # [N, R, 2048]
        #bert_feats =bert_feats.detach()
        
        sent_feat_map = self.mapping_sent(bert_feats)# 512
        #print("329...bertfeat %s sent_feat_map %s "%(bert_feats.shape,sent_feat_map.shape,))
        feat_cat_loc = torch.cat([head_pool, lfeat], 2)  # N R 2048+5 = N R 2053
        feat_cat_loc_map = self.vis_a_fc1(feat_cat_loc) # N , R, 512
        sent_feat_map= torch.unsqueeze(sent_feat_map,1)
        sent_feat_map0 = sent_feat_map.expand(-1, R, -1)
        sent_img0 = torch.cat((feat_cat_loc_map, sent_feat_map0), dim=2)           # [N, R, 1280](768) [N, R, 1024](512)
        #sent_weighted_by_img = sent_feat_map0 * self.se_sent(sent_img0)  # N R 512 
        # img_to_sent_weight = torch.softmax(sent_img1, dim=2)                # [N, R, dim]
        # weighted sent feat
        #weighted_sent_feat = torch.sum(sent_feats * img_to_sent_weight, dim=2)        # [N, R, 512]   this
        # weighted image feat
        img_weighted_by_sent = feat_cat_loc_map * self.se_img(sent_img0) #[N, R, 512]
        #head_mapped = self.vis_r_fc(feat_cat_loc)
        #t1=0.06
        #score_gt_neg =torch.einsum("nrc,nrc->nr",sent_weighted_by_img,img_weighted_by_sent)
        #sent_img_merge =(sent_weighted_by_img) * img_weighted_by_sent #[N, R, 512]
        #sent_img_merge = relu(sent_img_merge, inplace=True)  # [N, R, 512]
        sent_img_merge = normalize(img_weighted_by_sent, dim=2) # 
        score_gt_neg = self.rank_fc_gt(sent_img_merge)          # [N, R, 1]
        score_gt_neg = score_gt_neg.squeeze(2)               # [N, R]
        
        
        
        # Extract word feature with RNN
        packed_output, _ = self.rnn(packed_sent_feat)
        
        rnn_out, sent_len = pad_packed_sequence(packed_output, batch_first=True)  # [N, S, 512], [N]  [32, 10, 512]
        #print("284rnn_out%s sent_len%s"%(rnn_out, sent_len))
        #print("285rnn_out shape %s sent_len%s"%(rnn_out.shape, sent_len.shape)) # ([32, 8, 512]) sent_lentorch.Size([32])
        S = rnn_out.size(1)
        sent_mask = (torch.arange(S) + 1).unsqueeze(dim=0).expand(N, -1) > sent_len.unsqueeze(dim=1)
        #print("288pre sent_mask %s"%sent_mask)
        sent_mask = sent_mask[:, None, :, None].expand(-1, R, -1, -1)  # [N, R, S, 1]
        #print("290pre sent_mask %s"%sent_mask)
        #print("290pre sent_maskshape %s"%(sent_mask.shape,))
        # Cross-modal attention over words
        att_key = self.vis_a_fc(head_pool)                          # [N, R, 512]
        att_key = att_key.unsqueeze(dim=2).expand(-1, -1, S, -1)    # [N, R, S, 512]
        att_value = rnn_out.unsqueeze(dim=1).expand(-1, R, -1, -1)  # [N, R, S, 512]
        att_feat = torch.cat((att_key, att_value), dim=3)           # [N, R, S, 1024]
        att_feat = self.att_dropout(att_feat)                       # [N, R, S, 1024]
        att_score = self.att_fc(att_feat)                           # [N, R, S, 1]
        att_score[sent_mask] = float('-inf')
        #print("300att_score[sent_mask]%s"%(att_score[sent_mask],))
        att_weight = torch.softmax(att_score, dim=2)                # [N, R, S, 1]
        sent_feat = torch.sum(att_weight * att_value, dim=2)        # [N, R, 512]
        # Compute rank score
        head_mapped = self.vis_r_fc(head_pool)         # [N, R, 512]
        feat_merged = head_mapped * sent_feat          # [N, R, 512]
        feat_merged = relu(feat_merged, inplace=True)  # [N, R, 512]
        feat_merged = normalize(feat_merged, dim=2)    # [N, R, 512]
        # add gt to token ref 
        feat_merged = torch.cat((feat_merged, sent_img_merge), dim=2)  # 1024
        feat_merged = self.rank_dropout(feat_merged)   # [N, R, 1024]
        ref_score = self.rank_fc(feat_merged)          # [N, R, 1]
        ref_score = ref_score.squeeze(2)               # [N, R]


        # add loc and gt 
       
        return score_gt_neg, ref_score,
"""
# 2 not sent map layer, direct cat image, cat---then fusion

# sip qk add  its right
#/home/wj/code/ref_nms/output/my_sipqk_ad_refcoco_0903233435_b.pth
class AttVanillaPredictorV2(nn.Module):

    def __init__(self, att_dropout_p, rank_dropout_p):
        super(AttVanillaPredictorV2, self).__init__()
        # Head network with pretrained weight
        self.head = make_resnet_layer4()
        #self.head1= make_resnet_layer4()
        # Layers of new branch  
        self.vis_a_fc = nn.Sequential(
            nn.Linear(2048, 1024, bias=True), 
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True)
        )
        self.vis_r_fc = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True)
        )
        self.att_fc = nn.Linear(1024, 1, bias=True)
        self.rank_fc = nn.Linear(512, 1, bias=True)
        self.att_dropout = nn.Dropout(p=att_dropout_p, inplace=True)
        self.rank_dropout = nn.Dropout(p=rank_dropout_p, inplace=True)
        self.mapping_word = nn.Sequential(
           nn.Linear(768,512),
           nn.ReLU(),
             # follow one stage      
         )
        
        #print("298......")
        
        #self.rank_fc_gt = nn.Linear(512, 1, bias=True)
        #print("300......")
       
    def forward(self, roi_feat, sent_len, word_feats,bert_feats,lfeat):
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
      
        # Extract word feature with RNN
         
        
        #rnn_out, sent_len = pad_packed_sequence(packed_output, batch_first=True)  # [N, S, 512], [N]  [32, 10, 512]
        #print("284................rnn_out%s sent_len%s"%(rnn_out.shape, sent_len))
        #print("285rnn_out shape %s sent_len%s"%(rnn_out.shape, sent_len.shape)) # ([32, 8, 512]) sent_lentorch.Size([32])
        S = word_feats.size(2)
        #print("793..................sent_len%s"%sent_len)
        #print("794..................S%s"%S)
        sent_mask = (torch.arange(S) + 1).unsqueeze(dim=0).expand(N, -1) > sent_len.unsqueeze(dim=1)
        #print("288pre sent_mask %s"%sent_mask)
        sent_mask = sent_mask[:, None, :, None].expand(-1, R, -1, -1)  # [N, R, S, 1]
        #print("290pre sent_mask %s"%sent_mask)
        #print("290pre sent_maskshape %s"%(sent_mask.shape,))
        # Cross-modal attention over words
        #print("741att_key%s  "%(head_pool.shape,))
        att_key = self.vis_a_fc(head_pool) 
        #print("743att_key%s  "%(att_key.shape,))                         # [N, R, 512] 
        
        att_key = att_key.unsqueeze(dim=2).expand(-1, -1, S, -1)    # [N, R, S, 512]
        #print("746att_key%s  "%(att_key.shape,))    
        #print("719.............word_feats shape%s"%(word_feats.shape,))
        #att_value = word_feats.unsqueeze(dim=1).expand(-1, R, -1, -1)  # [N, R, S, 512]
        word_feats= self.mapping_word(word_feats)
        #print("746.word_feats%s"%(word_feats.shape,))
        att_value = word_feats.expand(-1, R, -1, -1)  # [N, R, S, 512]
        #print("748.att_value%s"%(att_value.shape,))
        #print("749.att_key%s"%(att_key.shape,)) #[32, 32, 1, 512]
        att_feat = torch.cat((att_key, att_value), dim=3)           # [N, R, S, 1024]
        att_feat = self.att_dropout(att_feat)                       # [N, R, S, 1024]
        att_score = self.att_fc(att_feat)                           # [N, R, S, 1]
        att_score[sent_mask] = float('-inf')
        #print("300att_score[sent_mask]%s"%(att_score[sent_mask],))
        att_weight = torch.softmax(att_score, dim=2)                # [N, R, S, 1]
        sent_feat = torch.sum(att_weight * att_value, dim=2)        # [N, R, 512]
        # Compute rank score
        head_mapped = self.vis_r_fc(head_pool)         # [N, R, 512]
        feat_merged = head_mapped * sent_feat        # [N, R, 512]
       
        feat_merged = relu(feat_merged, inplace=True)  # [N, R, 512]
        feat_merged = normalize(feat_merged, dim=2)    # [N, R, 512]
        # add gt to token ref 
        #feat_merged = torch.cat((feat_merged,sent_img_qkv), dim=2)  # 1024
        feat_merged = self.rank_dropout(feat_merged)   # [N, R, 1024]
        ref_score = self.rank_fc(feat_merged)          # [N, R, 1]
        ref_score = ref_score.squeeze(2)               # [N, R]
         
        # add loc and gt 
        

        return ref_score

class RankPredictor(nn.Module):

    def __init__(self):
        super(RankPredictor, self).__init__()
        self.head = make_resnet_layer4()
        self.word_fc = nn.Linear(300, 300, bias=True)
        self.head_fc = nn.Linear(2048, 300, bias=True)
        self.rank_fc = nn.Linear(300, 1, bias=True)

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
        head_mapped = self.head_fc(head_pool)                  # [N, R, 300]
        word_mapped = self.word_fc(word_feat)                  # [N, S, 300]
        head_expanded = head_mapped.unsqueeze(2)               # [N, R, 1, 300]
        word_expanded = word_mapped.unsqueeze(1)               # [N, 1, S, 300]
        feat_merged = head_expanded * word_expanded            # [N, R, S, 300]
        feat_merged = normalize(feat_merged, dim=3)            # [N, R, S, 300]
        rank_score = self.rank_fc(feat_merged).squeeze(dim=3)  # [N, R, S]
        max_rank_score, max_idx = rank_score.max(dim=2)        # [N, R], [N, R]
        return max_rank_score, max_idx

    def rank_parameters(self):
        return list(self.head_fc.parameters()) + list(self.word_fc.parameters()) + list(self.rank_fc.parameters())

    def named_rank_parameters(self):
        return list(self.head_fc.named_parameters()) + list(self.word_fc.named_parameters()) \
               + list(self.rank_fc.named_parameters())


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
