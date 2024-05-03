import argparse
import pickle
import json
import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from utils.hit_rate_utils import CtxHitRateEvaluator
from utils.hit_rate_utils import GtCtxHitRateEvaluator
from utils.constants import EVAL_SPLITS_DICT
from lib.refer import REFER


def threshold_with_confidence(exp_to_proposals, conf):
    results = {}
    for exp_id, proposals in exp_to_proposals.items():
        assert len(proposals) >= 1
        sorted_proposals = sorted(proposals, key=lambda p: p['score'], reverse=True)
        thresh_proposals = [sorted_proposals[0]]
        for prop in sorted_proposals[1:]:
            if prop['score'] > conf:
                thresh_proposals.append(prop)
            else:
                break
        results[exp_id] = thresh_proposals
    return results


def main(args):
    dataset_splitby = '{}_{}'.format(args.dataset, args.split_by)
    eval_splits = EVAL_SPLITS_DICT[dataset_splitby]
    # Load proposals
    #used_proposal_path = 'cache/proposals_{}_{}_{}.pkl'.format(args.m, args.dataset, args.tid)
    #used_proposal_path = '/home_data/wj/ref_nms/cache/yb07proposals_att_vanilla_refcoco+_0316212105.pkl'
    #used_proposal_path = "/data1/wj/ref_nms/cache/refncet0007_t102proposals_att_vanilla_refcoco.pkl"
    #used_proposal_path = "/data1/wj/ref_nms/cache/refncet002_t103proposals_att_vanilla_refcoco.pkl"
    #used_proposal_path = "/home_data/wj/ref_nms/cache/my_se_proposals_att_vanilla_refcocog_0923160805.pkl"
    #used_proposal_path = "/data1/wj/ref_nms/cache/my_1qkad_gt_proposals_att_vanilla_refcoco+.pkl"
    #used_proposal_path ="/data1/wj/ref_nms/cache/my_qkad_gt_proposals_att_vanilla_refcoco+.pkl"
    #used_proposal_path ="/data1/wj/ref_nms/cache/my_sepqkad_gt_proposals_att_vanilla_refcoco+.pkl"
    #used_proposal_path ="/data1/wj/ref_nms/cache/my_1qkad_adgtall3__proposals_att_vanilla_refcoco+.pkl"
    
    #used_proposal_path="/data1/wj/ref_nms/cache/my_qkad_adgtall__proposals_att_vanilla_refcoco+.pkl"
    #used_proposal_path="/data1/wj/ref_nms/cache/my_qkad_gtctx_proposals_att_vanilla_refcoco+.pkl"
    #used_proposal_path= "/data1/wj/ref_nms/cache/my_Sigad_adatall_bertall_proposals_att_vanilla_refcoco+.pkl"
    #used_proposal_path= "/data1/wj/ref_nms/cache/my_Sigad_adatall_bertall_proposals_att_vanilla_refcoco+.pkl"
    #used_proposal_path="/data1/wj/ref_nms/cache/my_bert_proposals_att_vanilla_refcocog.pkl"
    used_proposal_path = "/data1/wj/ref_nms/cache/my_bert_proposals_att_vanilla_refcoco.pkl"
    


    print('loading {} proposals from {}...'.format(args.m, used_proposal_path))
    with open(used_proposal_path, 'rb') as f:
        used_proposal_dict = pickle.load(f)
    # Load refer
    refer = REFER('/home_data/wj/ref_nms/data/refer', dataset=args.dataset, splitBy=args.split_by)
    # Load ctxdb
    json_pth = "/home_data/wj/ref_nms/cache/std_ctxdb_refcoco_unc.json"
    #"/home_data/wj/ref_nms/cache/std_ctxdb_refcoco+_unc.json"
    # "/data1/wj/ref_nms/cache/ybclip_ctxdb_ann_08senrefcoco+_unc.json"
    #"/home_data/wj/ref_nms/cache/ybclip_ctxdb_ann_07senrefcoco+_unc.json"
    #json = "/home_data/wj/ref_nms/cache/std_ctxdb_refcoco_unc.json"
    #json_pth = "/home_data/wj/ref_nms/cache/std_ctxdb_refcocog_umd.json"
    #with open("/data1/wj/ref_nms/cache/ybclip_ctxdb_ann_08senrefcoco+_unc.json", 'r') as f:
    with open(json_pth, 'r') as f:
        ctxdb = json.load(f)
    
    """
    with open("/home_data/wj/ref_nms/cache/ybclip_ctxdb_ann_07senrefcoco+_unc.json", 'r') as f:
        ctxdb = json.load(f)
    """
    """
    with open('cache/std_ctxdb_{}.json'.format(dataset_splitby), 'r') as f:
        ctxdb = json.load(f)
    """
    # Evaluate hit rate
    print('Context object recall on', dataset_splitby)
    #evaluator = CtxHitRateEvaluator(refer, ctxdb, top_N=None, threshold=args.thresh)
    evaluator = CtxHitRateEvaluator(refer, ctxdb, top_N=None, threshold=args.thresh)
    for split in eval_splits:
        exp_to_proposals = used_proposal_dict[split]
        exp_to_proposals = threshold_with_confidence(exp_to_proposals, args.conf)
        proposal_per_ref, hit_rate = evaluator.eval_hit_rate(split, exp_to_proposals)
        print('[{:5s}] hit rate: {:.2f} @ {:.2f}'.format(split, hit_rate*100, proposal_per_ref))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=str)
    parser.add_argument('--dataset', default='refcoco')
    parser.add_argument('--split-by', default='unc')
    parser.add_argument('--tid', type=str)
    parser.add_argument('--top-N', type=int)
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--conf', type=float, default=0.12)
    main(parser.parse_args())
