import torch
import torch.nn as nn
from torch.nn.functional import normalize, relu
from torch.nn.utils.rnn import pad_packed_sequence
from torchvision import models
from torch.autograd import Variable
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

class Normalize_Scale(nn.Module):
  def __init__(self, dim, init_norm=20):
    super(Normalize_Scale, self).__init__()
    self.init_norm = init_norm
    self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

  def forward(self, bottom):
    # input is variable (n, dim)
    assert isinstance(bottom, Variable), 'bottom must be variable'
    bottom_normalized = nn.functional.normalize(bottom, p=2, dim=1)
    bottom_normalized_scaled = bottom_normalized * self.weight
    return bottom_normalized_scaled

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
        self.rnn = nn.GRU(300, 256, bidirectional=True, batch_first=True)
        #print("266......")
        """
        self.vis_a_fck = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True)
        )
        """

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
        )
        
        
        self.head_feat0_normalizer = Normalize_Scale(2048,20) 
        self.phrase_normalizer= Normalize_Scale(512, 20)
        self.attn_fuse = nn.Sequential(nn.Linear(2048+512, 512),
                                       nn.Tanh(),
                                       nn.Linear(512, 1))
                                       
        self.vis_emb_fc  = nn.Sequential(nn.Linear(2048, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Linear(512, 512),
                                     nn.BatchNorm1d(512),
                                     )
                                     
        self.lang_emb_fc = nn.Sequential(nn.Linear(512, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Linear(512, 512),
                                     nn.BatchNorm1d(512)
                                     ) 
        self.siggt = nn.Sigmoid()
        self.sigref = nn.Sigmoid()
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
        head_feat = self.head(roi_feat.reshape(N * R, 1024, 7, 7))   # (N*R, 2048, 7, 7) --->N,R,49,2048
        head_feat0 = head_feat.reshape(N, R, 2048, 7, 7)
        head_pool = head_feat0.mean(dim=(3, 4))  # [N, R, 2048]
  

 
    
    
        head_feat0 = head_feat0.view(N*R, 2048, -1) #  (N*R, 2048,49)
        #print("139head_feat....%s"%(head_feat0.shape,))
        head_feat0 = head_feat0.transpose(1,2).contiguous().view(-1,2048) # (N*R*49, 2048)
        #head_feat0 = self.vis_a_fck(head_feat0) # (N*R*49, 512)
        #print("141head_feat....%s"%(head_feat.shape,))
        head_feat0 = self.head_feat0_normalizer(Variable(head_feat0)) # (N*R*49, 512)
        #print("137.........")
        sent_feat1 = self.mapping_sent(bert_feats)# N,512
        sent_feat2  = sent_feat1.unsqueeze(1).expand(N, R, 512) # (N,R 512)
        sent_feat2 = sent_feat2.contiguous().view(-1, 512) # (N*R, 512)
        #print("139.........")
        sent_feat = self.phrase_normalizer(sent_feat1)# (N, 512)      
        sent_feat  = sent_feat.unsqueeze(1).expand(N, R, 512) # (N,R 512)
        sent_feat = sent_feat.contiguous().view(-1, 512) # (N*R, 512) 
        sent_feat  = sent_feat.unsqueeze(1).expand(N*R,49, 512) # (N*R,49, 512) 
        sent_feat = sent_feat.contiguous().view(-1, 512) # (N*R*49, 512) 
        #print("153sent_featshape%s"%(sent_feat.shape,))
    
        attn = self.attn_fuse(torch.cat([head_feat0,sent_feat], 1)) # (NxRx49, 1)
        attn = torch.softmax(attn.view(N*R, 49),dim=1) # (n*R, 49)
        attn3 = attn.unsqueeze(1)  # (n*R, 1, 49)
        weighted_visual_feats = torch.bmm(attn3, head_feat0.view(N*R, 49, -1)) # (n*R, 1, 49) * (n*R, 49, 2048)-->(n*R, 1, 2048)
        weighted_visual_feats = weighted_visual_feats.squeeze(1) # (n*R, 2048)
        #print("161 weighted_visual_feats %s"%(weighted_visual_feats.shape,))
        visual_emb = self.vis_emb_fc(weighted_visual_feats)
        lang_emb = self.lang_emb_fc(sent_feat2)
        
        # l2-normalize
        visual_emb_normalized = nn.functional.normalize(visual_emb, p=2, dim=1) # (n, jemb_dim)
        lang_emb_normalized = nn.functional.normalize(lang_emb, p=2, dim=1)     # (n, jemb_dim)  
        # compute cossim
        cossim_gt = torch.sum(visual_emb_normalized * lang_emb_normalized, 1)  # (n*R, )
        cossim_gt = cossim_gt.view(N,R)
        sig_gt = self.siggt(cossim_gt)
        
        
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
        feat_merged = head_mapped * sent_feat        # [N, R, 512]
       
        feat_merged = relu(feat_merged, inplace=True)  # [N, R, 512]
        feat_merged = normalize(feat_merged, dim=2)    # [N, R, 512]
        # add gt to token ref 
        #feat_merged = torch.cat((feat_merged,sent_img_qkv), dim=2)  # 1024
        feat_merged = self.rank_dropout(feat_merged)   # [N, R, 1024]
        ref_score = self.rank_fc(feat_merged)          # [N, R, 1]
        ref_score = ref_score.squeeze(2)               # [N, R]
        sig_ref = self.sigref(ref_score)
        ref_score =(sig_ref + sig_gt)/2
        # add loc and gt 

        return sig_gt, ref_score,

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
