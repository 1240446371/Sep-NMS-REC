import json
import pickle
import argparse
from multiprocessing import Pool
import os
import sys
root_path = os.path.abspath(__file__) 
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
import torch
from tqdm import tqdm
from torchvision.ops import nms
from torch.nn.utils.rnn import pack_padded_sequence 
#import clip
from lib.my_qkad_predictor import AttVanillaPredictorV2
from lib.my_adyb_vanilla_utils import DetEvalLoader as DetEvalLoader
from utils.constants import EVAL_SPLITS_DICT
import numpy as np
from transformers import AutoTokenizer, BertModel
import os.path as osp
# load rpn, >0.05 class confidence; then caculate the relate score, combine the class confidence, obtain the threhold
def rank_proposals(position, gpu_id, tid, refdb_path, ctxdb_path,split,m,dataset_splitby):
    # Load refdb
    with open(refdb_path) as f:
        refdb = json.load(f)
    with open(ctxdb_path) as f:  
        ctxdb = json.load(f)
    dataset_ = refdb['dataset_splitby'].split('_')[0]
    # Load pre-trained model
   # device = torch.device('cuda', gpu_id)
    device=torch.device('cuda',gpu_id)

    #with open('/home/wj/code/ref_nms/output/neg045_32yb004_{}_{}_{}.json'.format(m, dataset_, tid), 'r') as f:
    #json_pth ="/home/wj/code/ref_nms/output/05refyb_att_vanilla_refcoco+_0624154853.json"
    #json_pth ="/home/wj/code/ref_nms/output/08yb_att_vanilla_refcoco+_0714160018.json"
    #json_pth ="/home/wj/code/ref_nms/output/att_vanilla_refcoco_1019204514.json"
    json_pth ="/home_data/wj/ref_nms/output/my_sipsepqk_ad_att_vanilla_refcoco+_0913191409.json"
    #json_pth = "/home/wj/code/ref_nms/output/yb003_att_vanilla_refcocog_0316163239.json"

    with open(json_pth.format(m, dataset_, tid), 'r') as f:
    #with open("/home/wj/code/ref_nms/output/yb006_att_vanilla_refcoco+.json") as f:
        #print("jsonpath%s"%f)
        model_info = json.load(f)
    predictor = AttVanillaPredictorV2(att_dropout_p=model_info['config']['ATT_DROPOUT_P'],
                                      rank_dropout_p=model_info['config']['RANK_DROPOUT_P'])
    
    #model_path = '/home/wj/code/ref_nms/output/{}_yb008_{}_{}_b.pth'.format(m, dataset_, tid)
    #model_path = '/home/wj/code/ref_nms/output/att_vanilla_yb008_refcoco+_0110152420_b.pth'
    #model_path ="/home/wj/code/ref_nms/output/att_vanilla_yb008ckpt_refcoco+_0110152420_2.pth" # 3,4,5
    #model_path ="/home/wj/code/ref_nms/output/att_vanilla_yb004_refcocog_0112101456_b.pth"
    #model_path ='/home/wj/code/ref_nms/output/att_vanilla_yb008_refcoco+_0112095046_b.pth'
    #model_path ='/home/wj/code/ref_nms/output/neg045_32att_vanilla_yb004_refcocog_0318182906_b.pth'
    #/home/wj/code/ref_nms/output/neg045_24yb004_att_vanilla_refcocog_0318142357.json
    #model_path ="/home/wj/code/ref_nms/output/att_vanilla_yb006ckpt_refcoco+_5.pth"
    #model_path  = "/home/wj/code/ref_nms/output/05refyb_att_vanilla_refcoco+_0624154853_b.pth"
    #model_path  = "/home/wj/code/ref_nms/output/048refyb_att_vanilla_refcoco+_0625104037_b.pth"
    #model_path  = "/home/wj/code/ref_nms/output/048refyb_att_vanilla_refcoco_0626110852_b.pth"
    #model_path  ="/home/wj/code/ref_nms/output/03yb_att_vanilla_refcocog_0713144804_b.pth"
    #model_path  ="/home/wj/code/ref_nms/output/att_vanilla_refcoco+_1019213349_b.pth"
    #model_path  ="/home/wj/code/ref_nms/output/att_vanilla_refcoco_1019204514_b.pth"
    #model_path  = "/home/wj/code/ref_nms/output/att_vanilla_refcocog_0720222934_b.pth"
    #model_path  = "/home/wj/code/ref_nms/output/my_sipqk_ad_att_vanilla_refcoco+_unc_0913094949_b.pth"
    #model_path  = "/home/wj/code/ref_nms/output/my_sipqk_ad_refcoco_0909181253_b.pth"
    model_path  = "/home_data/wj/ref_nms/output/my_sipqk_ad_att_vanilla_refcoco+_unc_0913094949_b.pth"
 


    

    
    predictor.load_state_dict(torch.load(model_path,map_location=device))
    predictor.to(device)
    predictor.eval()
    
    data_json =  osp.join('/home/wj/code/MAttNet_master/cache/prepro',dataset_splitby, 'data.json') 
    #ctxdb_path ="/home/wj/code/ref_nms/cache/ybclip_ctxdb_ann_04sen{}.json".format(dataset_splitby)
    #ctxdb_path = "/home/wj/code/ref_nms/cache/ybclip_ctxdb_ann_08senrefcoco_unc.json"
 
    info = json.load(open(data_json))  
    images = info['images'] 
    Images = {image['image_id']: image for image in images}
    
    bert_model_name = 'bert-base-uncased'
    tokenizer=AutoTokenizer.from_pretrained(bert_model_name)
    with torch.no_grad():
      textmodel = BertModel.from_pretrained(bert_model_name)
      textmodel.eval()
    loader = DetEvalLoader(tokenizer,textmodel,Images,refdb, ctxdb,split, gpu_id)
    #print("41loadok")
    # Rank proposals
    
    exp_to_proposals_gt= {}
    exp_to_proposals_ref= {}
    exp_to_proposals_all= {}
    loader = DetEvalLoader(tokenizer,textmodel,Images,refdb,ctxdb,split, gpu_id)
    tqdm_loader = tqdm(loader, desc='scoring {}'.format(split), ascii=True, position=position)
    t_num=0
    
    for exp_id,lfeats, pos_feat, sent_feat, bert_feats, pos_box, pos_score, cls_num_list,rpn_clip_score in tqdm_loader:
        # Compute rank score
        #t_num+=1
        #print("71exp_id, pos_feat, sent_feat, pos_box, pos_score, cls_num_list ,rpn_clip_score%s"%(exp_id, pos_feat, sent_feat, pos_box, pos_score, cls_num_list ,rpn_clip_score))
        packed_sent_feats = pack_padded_sequence(sent_feat, torch.tensor([sent_feat.size(1)]),
                                                 enforce_sorted=False, batch_first=True)  # save first sequence, then second sequence
        with torch.no_grad():
            #print("54pos_featshape%s packed_sent_feats.shape%s"%(pos_feat.shape,packed_sent_feats.shape))
            gt_score , rank_score, = predictor(pos_feat, packed_sent_feats,bert_feats,lfeats)   # [1, n_box]
            #print("56rankscoreshape%s"%(rank_score.shape,))  # [1,n_posbox]
        # Normalize rank score
        #print("53rankscore%s"%rank_score)
        #print("54rankscore[0]%s"%rank_score[0])
        rank_score = torch.sigmoid(rank_score[0])#[n_box]
        gt_score =torch.sigmoid(gt_score[0])
        #print("54rank_score%s"%rank_score)
        #print("85rank_score shape%s"%(rank_score.shape,))  #torch.Size([304])
        #print("86rpn_clip_score shape%s"%(rpn_clip_score.dtype))
        # Split scores and boxes category-wise
        rpn_clip_score=torch.from_numpy(rpn_clip_score.astype(np.float32)).to(device)
        #print("89%s"%(rank_score.dtype))
        rpn_clip_list =torch.split(rpn_clip_score, cls_num_list, dim=0)
        
        rank_score_list = torch.split(rank_score, cls_num_list, dim=0)  # tuple
        gt_score_list = torch.split(gt_score, cls_num_list, dim=0)
        #gt_ad_rank_score_list = torch.split(yb_ad_gt_rank, cls_num_list, dim=0)
        #print("65lenrank_scorelist %s"%(len(rank_score_list)))  #80
        pos_box_list = torch.split(pos_box, cls_num_list, dim=0)
        pos_score_list = torch.split(pos_score, cls_num_list, dim=0)    # len()=80
       
       
        # Combine score and do NMS category-wise
      
        proposals_gt = []
        proposals_ref = []
        proposals_all = []
        cls_idx = 0
        for cls_gt, cls_rank ,cls_clip, cls_pos_box, cls_pos_score in zip(gt_score_list,rank_score_list, rpn_clip_list, pos_box_list, pos_score_list):
            cls_idx += 1
            # No positive box under this category
            #print("76cls_rank_score%s"%(cls_rank_score,))  #one cat box score
            #if cls_rank.size(0) == 0 and cls_gt.size(0) == 0 and cls_gt_ad_rank.size(0) == 0: # size(0)some is 0, and some is 16 or other
            if cls_gt.size(0) == 0 and cls_rank.size(0) == 0 and cls_clip.size(0) == 0:
                continue
       
            final_score_rank1 = (cls_rank+ (cls_gt)+ cls_clip) * cls_pos_score   
            
                   
         
            keeprank1 = nms(cls_pos_box, final_score_rank1, iou_threshold=0.3)
            
         
            cls_kept_box_all = cls_pos_box[keeprank1]
            cls_kept_score_all = final_score_rank1[keeprank1]
            
         
              
            for box2, score2 in zip(cls_kept_box_all, cls_kept_score_all):
                proposals_all.append({'score': score2.item(), 'box': box2.tolist(), 'cls_idx': cls_idx})

        assert cls_idx == 80

        exp_to_proposals_all[exp_id] = proposals_all
    return exp_to_proposals_all


def error_callback(e):
    print('\n\n\n\nERROR in subprocess:', e, '\n\n\n\n')


def main(args):
    dataset_splitby = '{}_{}'.format(args.dataset, args.split_by)
    eval_splits = EVAL_SPLITS_DICT[dataset_splitby]
    #refdb_path = '/home/wj/code/ref_nms/cache/std_refdb_{}.json'.format(dataset_splitby)
    #ctxdb_path = "/home/wj/code/ref_nms/cache/score_ybclip_ctxdb_ann_senrefcoco+_unc.json"
    #ctxdb_path = "/home/wj/code/ref_nms/cache/score_ybclip_ctxdb_ann_senrefcoco_unc.json"
    refdb_path = '/home_data/wj/ref_nms/cache/std_refdb_{}.json'.format(dataset_splitby)
    #ctxdb_path = "/home/wj/code/ref_nms/cache/score_ybclip_ctxdb_ann_senrefcoco+_unc.json"
    ctxdb_path = "/data1/wj/ref_nms/cache/score_ybclip_ctxdb_ann_senrefcoco+_unc.json"
    #ctxdb_path = "/home/wj/code/ref_nms/cache/score_ybclip_ctxdb_ann_senrefcocog_umd.json"

    print('about to rank proposals via multiprocessing, good luck ~')
    results = {}
    with Pool(processes=len(eval_splits)) as pool:
        for idx, split in enumerate(eval_splits):
            sub_args = (idx, args.gpu_id, args.tid, refdb_path,ctxdb_path, split, args.m,dataset_splitby)
            #print("132sub_args%s"%(sub_args,))
            results[split] = pool.apply_async(rank_proposals, sub_args, error_callback=error_callback)
        pool.close()
        pool.join()
    proposal_dict = {}
   
    for split in eval_splits:
        print("137split%s"%split)
        assert results[split].successful()
        print('subprocess for {} split succeeded, fetching results...'.format(split))
      
        proposal_dict[split] = results[split].get()
    #save_path = '/home/wj/code/ref_nms/cache/proposalsnocls_{}_{}_{}.pkl'.format(args.m, args.dataset, args.tid)
    #save_path = '/home/wj/code/ref_nms/cache/ref_ad_003yb_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path1 = '/home/wj/code/ref_nms/cache/qkadgtcube_ad_008yb_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    #save_path2 = '/home/wj/code/ref_nms/cache/qkadcubeall_ad_008yb_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    save_path = '/data1/wj/ref_nms/cache/qkad008yb_clsall_proposals_{}_{}.pkl'.format(args.m, args.dataset)
    
    #print('saving proposals to {}...'.format(save_path))
    
    #with open(save_path1, 'wb') as f:
    #    pickle.dump(proposal_dict1, f)
    #with open(save_path2, 'wb') as f:
    #    pickle.dump(proposal_dict2, f)  
    with open(save_path, 'wb') as f:
        pickle.dump(proposal_dict, f)
    print('all done ~')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', default='refcoco+')
    parser.add_argument('--split_by', default='unc')
    parser.add_argument('--tid', type=str, default='0720222934')# coco+ 0112095046 0318182906
    parser.add_argument('--m', type=str, default='att_vanilla')
    main(parser.parse_args())
