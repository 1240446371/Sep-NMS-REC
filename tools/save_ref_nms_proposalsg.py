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

#from lib.my_se_predictor import AttVanillaPredictorV2
from lib.my_qkadctx_predictor import AttVanillaPredictorV2
from lib.vanilla_utilsg import DetEvalLoader
from utils.constants import EVAL_SPLITS_DICT

#from lib.my_vanilla_utils import DetEvalLoader

def rank_proposals(position, gpu_id, tid, refdb_path, split, m):
    # Load refdb
    with open(refdb_path) as f:
        refdb = json.load(f)
    dataset_ = refdb['dataset_splitby'].split('_')[0]
    # Load pre-trained model
    device = torch.device('cuda', gpu_id)

    #json_pth ="/home_data/wj/ref_nms/output/my_att_qkvcat_att_vanilla_refcocog_0919232856.json"
    json_pth ="/home_data/wj/ref_nms/output/my_sipsepqk_ad_att_vanilla_refcoco+_0913191409.json"
    #with open(json_pth.format(m, dataset_, tid), 'r') as f:
    with open(json_pth,'r') as f:
        #print("jsonpath%s"%f)
        model_info = json.load(f) 
        
    predictor = AttVanillaPredictorV2(att_dropout_p=model_info['config']['ATT_DROPOUT_P'],
                                      rank_dropout_p=model_info['config']['RANK_DROPOUT_P'])
    
    #model_path = '/home_data/wj/ref_nms/output/{}_{}_{}_b.pth'.format(m, dataset_, tid)
    #model_path = "/home_data/wj/ref_nms/output/myse_att_vanilla_0923160805_b.pth"
    #model_path = "/home_data/wj/ref_nms/output/my2se_att_vanilla_0923230030_b.pth"
    model_path = "/home_data/wj/ref_nms/output/my_sipqkad_gtctxrefcoco+_1028133418_b.pth"


    #print("model_path%s"%model_path)
    predictor.load_state_dict(torch.load(model_path))
    predictor.to(device)
    predictor.eval()
    # Rank proposals
    exp_to_proposals = {}
    loader = DetEvalLoader(refdb, split, gpu_id)
    tqdm_loader = tqdm(loader, desc='scoring {}'.format(split), ascii=True, position=position)
    for exp_id, pos_feat, sent_feat, pos_box, pos_score, cls_num_list in tqdm_loader:
        # Compute rank score
        packed_sent_feats = pack_padded_sequence(sent_feat, torch.tensor([sent_feat.size(1)]),
                                                 enforce_sorted=False, batch_first=True)
        with torch.no_grad():
            rank_score, *_ = predictor(pos_feat, packed_sent_feats)  # [1, *]
        # Normalize rank score
        rank_score = torch.sigmoid(rank_score[0])
        # Split scores and boxes category-wise
        rank_score_list = torch.split(rank_score, cls_num_list, dim=0)
        pos_box_list = torch.split(pos_box, cls_num_list, dim=0)
        pos_score_list = torch.split(pos_score, cls_num_list, dim=0)
        # Combine score and do NMS category-wise
        proposals = []
        cls_idx = 0
        for cls_rank_score, cls_pos_box, cls_pos_score in zip(rank_score_list, pos_box_list, pos_score_list):
            cls_idx += 1
            # No positive box under this category
            if cls_rank_score.size(0) == 0:
                continue
            final_score = cls_rank_score * cls_pos_score
            keep = nms(cls_pos_box, final_score, iou_threshold=0.3)
            cls_kept_box = cls_pos_box[keep]
            cls_kept_score = final_score[keep]
            for box, score in zip(cls_kept_box, cls_kept_score):
                proposals.append({'score': score.item(), 'box': box.tolist(), 'cls_idx': cls_idx})
        assert cls_idx == 80
        exp_to_proposals[exp_id] = proposals
    return exp_to_proposals


def error_callback(e):
    print('\n\n\n\nERROR in subprocess:', e, '\n\n\n\n')


def main(args):
    dataset_splitby = '{}_{}'.format(args.dataset, args.split_by)
    eval_splits = EVAL_SPLITS_DICT[dataset_splitby]
    refdb_path = '/home_data/wj/ref_nms/cache/std_refdb_{}.json'.format(dataset_splitby)
    print('about to rank proposals via multiprocessing, good luck ~')
    results = {}
    with Pool(processes=len(eval_splits)) as pool:
        for idx, split in enumerate(eval_splits):
            sub_args = (idx, args.gpu_id, args.tid, refdb_path, split, args.m)
            results[split] = pool.apply_async(rank_proposals, sub_args, error_callback=error_callback)
        pool.close()
        pool.join()
    proposal_dict = {}
    for split in eval_splits:
        assert results[split].successful()
        print('subprocess for {} split succeeded, fetching results...'.format(split))
        proposal_dict[split] = results[split].get()
    #save_path = '/data1/wj/ref_nms/cache/proposals_{}_{}_{}.pkl'.format(args.m, args.dataset, args.tid)
    save_path = '/home_data/wj/ref_nms/cache/my_sipqkad_gtctx_proposals_{}_{}_{}.pkl'.format(args.m, args.dataset, args.tid)
    print('saving proposals to {}...'.format(save_path))
    with open(save_path, 'wb') as f:
        pickle.dump(proposal_dict, f)
    print('all done ~')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, default=2)
    parser.add_argument('--dataset', default='refcoco+')
    parser.add_argument('--split-by', default='unc')
    parser.add_argument('--tid', type=str, default='1028133418')
    parser.add_argument('--m', type=str, default='att_vanilla')
    main(parser.parse_args())
