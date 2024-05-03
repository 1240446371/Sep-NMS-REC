
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



class AttVanillaPredictorV2(nn.Module):

    def __init__(self, att_dropout_p, rank_dropout_p):
        super(AttVanillaPredictorV2, self).__init__()
        # Head network with pretrained weight
        self.head = make_resnet_layer4()
        # Layers of new branch
        

        self.vis_a_fc = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True)
        )
        self.se1 =  nn.Sequential(
            nn.Linear(512, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512, bias=False),
            nn.Sigmoid()
        )
        self.se2 =  nn.Sequential(
            nn.Linear(1024, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1024, bias=False),
            nn.Sigmoid()
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

    def forward(self, roi_feat, packed_sent_feat):
        """
        Args:
            roi_feat: [N, R, 1024, 7, 7].
            packed_sent_feat: `PackedSequence` object, should be moved to proper device in advance.

        Returns:
            ref_score: [N, R].

        """
        # Extract visual feature with ResNet Conv-head
        #print("276roishape%s"%(roi_feat.shape,)) # [1, 305, 1024, 7, 7]) N is batch, R is region number   [32, 32, 1024, 7, 7]
        #print("276packedshape%s"%len(packed_sent_feat))  # 4?
        N, R, *_ = roi_feat.shape #  [32, 32, 1024, 7, 7]
        #head_feat = self.head(roi_feat.reshape(N * R, 1024, 7, 7)).reshape(N, R, 2048, 7, 7)
        head_feat = self.head(roi_feat.reshape(N * R, 1024, 7, 7)) 
        head_pool = head_feat.mean(dim=(2, 3)).reshape(N, R, 2048)  # [N*R, 2048]
        #print("70headpoolshape%s"%(head_pool.shape,))
        #head_pool = (head_pool*self.se(head_pool)).reshape(N, R, 2048)
        # Extract word feature with RNN
        packed_output, _ = self.rnn(packed_sent_feat)    
        #print("283packed_output%s"%(packed_output,))
        rnn_out, sent_len = pad_packed_sequence(packed_output, batch_first=True)  # [N, S, 512], [N]  [32, 10, 512]
        #print("284rnn_out%s sent_len%s"%(rnn_out, sent_len))
        #print("285rnn_out shape %s sent_len%s"%(rnn_out.shape, sent_len.shape)) # ([32, 8, 512]) sent_lentorch.Size([32])
        S = rnn_out.size(1)
        sent_mask = (torch.arange(S) + 1).unsqueeze(dim=0).expand(N, -1) > sent_len.unsqueeze(dim=1)
        #print("289%s"%(torch.arange(S) + 1).unsqueeze(dim=0))
        #print("290%s"%((torch.arange(S) + 1).unsqueeze(dim=0).shape,))
        #print("291%s"%((torch.arange(S) + 1).unsqueeze(dim=0).expand(N, -1).shape,))
        #print("288pre sent_mask %s"%(sent_mask))
        #print("288pre sent_mask %s"%(sent_mask.shape,))
        sent_mask = sent_mask[:, None, :, None].expand(-1, R, -1, -1)  # [N, R, S, 1]
        #print("290pre sent_mask %s"%sent_mask)
        # Cross-modal attention over words
        att_key = self.vis_a_fc(head_pool)                          # [N, R, 512]
        att_key = att_key + (att_key* self.se1(att_key))
       
        att_key = att_key.unsqueeze(dim=2).expand(-1, -1, S, -1)    # [N, R, S, 512]
        att_value = rnn_out.unsqueeze(dim=1).expand(-1, R, -1, -1)  # [N, R, S, 512]
        att_feat = torch.cat((att_key, att_value), dim=3)           # [N, R, S, 1024]
        att_feat = self.att_dropout(att_feat)                       # [N, R, S, 1024]
        att_feat = att_feat+(att_feat * self.se2(att_feat))
        att_score = self.att_fc(att_feat)                           # [N, R, S, 1]
        att_score[sent_mask] = float('-inf')
        #print("300att_score[sent_mask]%s"%(att_score))
        att_weight = torch.softmax(att_score, dim=2)                # [N, R, S, 1]
        sent_feat = torch.sum(att_weight * att_value, dim=2)        # [N, R, 512]   this
        # Compute rank score
        head_mapped = self.vis_r_fc(head_pool)         # [N, R, 512]   this
        feat_merged = head_mapped * sent_feat          # [N, R, 512]
        feat_merged = relu(feat_merged, inplace=True)  # [N, R, 512]
        feat_merged = normalize(feat_merged, dim=2)    # [N, R, 512]
        feat_merged = self.rank_dropout(feat_merged)   # [N, R, 512]
        ref_score = self.rank_fc(feat_merged)          # [N, R, 1]
        ref_score = ref_score.squeeze(2)               # [N, R]

        return ref_score,


"""
class AttVanillaPredictorV2(nn.Module):

    def __init__(self, att_dropout_p, rank_dropout_p):
        super(AttVanillaPredictorV2, self).__init__()
        # Head network with pretrained weight
        self.head = make_resnet_layer4()
        # Layers of new branch
        

        self.vis_a_fc = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True)
        )
        self.se =  nn.Sequential(
            nn.Linear(2048, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2048, bias=False),
            nn.Sigmoid()
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

    def forward(self, roi_feat, packed_sent_feat):

        # Extract visual feature with ResNet Conv-head
        #print("276roishape%s"%(roi_feat.shape,)) # [1, 305, 1024, 7, 7]) N is batch, R is region number   [32, 32, 1024, 7, 7]
        #print("276packedshape%s"%len(packed_sent_feat))  # 4?
        N, R, *_ = roi_feat.shape #  [32, 32, 1024, 7, 7]
        #head_feat = self.head(roi_feat.reshape(N * R, 1024, 7, 7)).reshape(N, R, 2048, 7, 7)
        head_feat = self.head(roi_feat.reshape(N * R, 1024, 7, 7)) 
        head_pool = head_feat.mean(dim=(2, 3))  # [N*R, 2048]
        #print("70headpoolshape%s"%(head_pool.shape,))
        head_pool = (head_pool*self.se(head_pool)).reshape(N, R, 2048)
        # Extract word feature with RNN
        packed_output, _ = self.rnn(packed_sent_feat)    
        #print("283packed_output%s"%(packed_output,))
        rnn_out, sent_len = pad_packed_sequence(packed_output, batch_first=True)  # [N, S, 512], [N]  [32, 10, 512]
        #print("284rnn_out%s sent_len%s"%(rnn_out, sent_len))
        #print("285rnn_out shape %s sent_len%s"%(rnn_out.shape, sent_len.shape)) # ([32, 8, 512]) sent_lentorch.Size([32])
        S = rnn_out.size(1)
        sent_mask = (torch.arange(S) + 1).unsqueeze(dim=0).expand(N, -1) > sent_len.unsqueeze(dim=1)
        #print("289%s"%(torch.arange(S) + 1).unsqueeze(dim=0))
        #print("290%s"%((torch.arange(S) + 1).unsqueeze(dim=0).shape,))
        #print("291%s"%((torch.arange(S) + 1).unsqueeze(dim=0).expand(N, -1).shape,))
        #print("288pre sent_mask %s"%(sent_mask))
        #print("288pre sent_mask %s"%(sent_mask.shape,))
        sent_mask = sent_mask[:, None, :, None].expand(-1, R, -1, -1)  # [N, R, S, 1]
        #print("290pre sent_mask %s"%sent_mask)
        # Cross-modal attention over words
        att_key = self.vis_a_fc(head_pool)                          # [N, R, 512]
       
        att_key = att_key.unsqueeze(dim=2).expand(-1, -1, S, -1)    # [N, R, S, 512]
        att_value = rnn_out.unsqueeze(dim=1).expand(-1, R, -1, -1)  # [N, R, S, 512]
        att_feat = torch.cat((att_key, att_value), dim=3)           # [N, R, S, 1024]
        att_feat = self.att_dropout(att_feat)                       # [N, R, S, 1024]
        att_score = self.att_fc(att_feat)                           # [N, R, S, 1]
        att_score[sent_mask] = float('-inf')
        #print("300att_score[sent_mask]%s"%(att_score))
        att_weight = torch.softmax(att_score, dim=2)                # [N, R, S, 1]
        sent_feat = torch.sum(att_weight * att_value, dim=2)        # [N, R, 512]   this
        # Compute rank score
        head_mapped = self.vis_r_fc(head_pool)         # [N, R, 512]   this
        feat_merged = head_mapped * sent_feat          # [N, R, 512]
        feat_merged = relu(feat_merged, inplace=True)  # [N, R, 512]
        feat_merged = normalize(feat_merged, dim=2)    # [N, R, 512]
        feat_merged = self.rank_dropout(feat_merged)   # [N, R, 512]
        ref_score = self.rank_fc(feat_merged)          # [N, R, 1]
        ref_score = ref_score.squeeze(2)               # [N, R]

        return ref_score,
"""