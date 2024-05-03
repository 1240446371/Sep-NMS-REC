import argparse
import pickle
import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from utils.hit_rate_utils import NewHitRateEvaluator
from utils.constants import EVAL_SPLITS_DICT
from lib.refer import REFER


def threshold_with_confidence(exp_to_proposals, conf):
    results = {}
    for exp_id, proposals in exp_to_proposals.items():
    #for exp_id, proposals in exp_to_proposals:
        print("exp_to_proposals.items%s"%(exp_to_proposals.items(),))
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

# "/home_data/wj/ref_nms/cache/std_ctxdb_refcoco_unc.json"
# "/data1/wj/ref_nms/cache/proposals_att_vanilla_refcoco_1019204514.pkl"
def main(args):
    dataset_splitby = '{}_{}'.format(args.dataset, args.split_by)
    eval_splits = EVAL_SPLITS_DICT[dataset_splitby]

    # Load proposals
    #proposal_path = '/data1/wj/ref_nms/cache/proposals_{}_{}_{}.pkl'.format(args.m, args.dataset, args.tid)
    split = 'val'
    proposal_path = '/data1/wj/ref_nms/cache/clipproposals_{}_{}.pkl'.format( dataset_splitby,split)
    #proposal_path = '/home_data/wj/ref_nms/cache/std_ctxdb_refcoco_unc.json'
    print('loading {} proposals from {}...'.format(args.m, proposal_path))
    with open(proposal_path, 'rb') as f:
        proposal_dict = pickle.load(f)
        #print("proposal_dict%s"%(proposal_dict,))
    # Load refer
    refer = REFER('/home_data/wj/ref_nms/data/refer', dataset=args.dataset, splitBy=args.split_by)
    # Evaluate hit rate
    print('Hit rate on {}\n'.format(dataset_splitby))
    evaluator = NewHitRateEvaluator(refer, top_N=None, threshold=args.thresh)
    print('conf: {:.3f}'.format(args.conf))
    for split in eval_splits:
        split = 'val'
        exp_to_proposals = (proposal_dict[split]).get()
        print("exp_to_proposals%s"%(exp_to_proposals,))
        exp_to_proposals = threshold_with_confidence(exp_to_proposals, args.conf)
        proposal_per_ref, hit_rate = evaluator.eval_hit_rate(split, exp_to_proposals)
        print('[{:5s}] hit rate: {:.2f} @ {:.2f}'.format(split, hit_rate*100, proposal_per_ref))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=str, default='att_vanilla')
    parser.add_argument('--dataset', default='refcoco+')
    parser.add_argument('--split-by', default='unc')
    parser.add_argument('--tid', type=str,default='1019213349' )# 1019213349,1019204514
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--conf', type=float,default=0.5 )
    main(parser.parse_args())
