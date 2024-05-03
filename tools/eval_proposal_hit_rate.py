import argparse
import pickle
import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from utils.hit_rate_utils import NewHitRateEvaluator
#from utils.constants import EVAL_SPLITS_DICTtestA
from utils.constants import EVAL_SPLITS_DICT
from lib.refer import REFER


def threshold_with_confidence(exp_to_proposals, conf):
    results = {}
    for exp_id, proposals in exp_to_proposals.items():
    #for exp_id, proposals in exp_to_proposals:
        #print("exp_to_proposals.items%s"%(exp_to_proposals.items(),))
        #print("exp_id %s proposal %s"%(exp_id,proposals))
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
    #eval_splits = EVAL_SPLITS_DICTtestA[dataset_splitby]
    eval_splits = EVAL_SPLITS_DICT[dataset_splitby]

    # Load proposals
    #proposal_path = '/data1/wj/ref_nms/cache/proposals_{}_{}_{}.pkl'.format(args.m, args.dataset, args.tid)
    #proposal_path = '/home_data/wj/ref_nms/cache/yb07proposals_att_vanilla_refcoco+_0316212105.pkl'
    #split = 'testB'
    #proposal_path = '/data1/wj/ref_nms/cache/clipproposals_{}_{}.pkl'.format( dataset_splitby,split)
    #proposal_path = '/home_data/wj/ref_nms/cache/std_ctxdb_refcoco_unc.json'
    # "/data1/wj/ref_nms/cache/ybclip_ctxdb_ann_01senrefcoco_unc.json"
    #proposal_path ="/data1/wj/ref_nms/cache/refncet002_t103proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/clipproposals_att_vanilla_refcoco_1019204514.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/refnce_t0005t1007proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/refncet0007_t102proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/refnce_t1006_t1006_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/home/wj/code/ref_nms/cache/refnce_t1006_att_vanilla_refcocog_0530222413.pkl"
    #proposal_path ="/home/wj/code/ref_nms/cache/clipproposals_att_vanilla_refcoco_1019204514.pkl
    #proposal_path = "/data1/wj/ref_nms/cache/my_refsip_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_refsip_adgtall_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_refsip_gt_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_refsipsep_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_refsipsep_adgtall_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_refsipsep_gt_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_refsipsep_adgt02_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_refsip_adgt049_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_refsip_adgt049nms_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_refsip_adgt04_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/0my_refsipsep_adgt049nms_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/0my_refsipsep_adgt05_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/0my_refsip_adgt049_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_qkvcat_gt_proposals_att_vanilla_refcoco.pkl"
    
    #proposal_path ="/data1/wj/ref_nms/cache/my_qkvcat_ref_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_qkvad_ref_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_qkvad_gt_proposals_att_vanilla_refcoco.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_sipcat_ref_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_sipcat_gt_proposals_att_vanilla_refcoco.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_sipsep_gt_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_sipsep_adgtall_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_qkad_gt_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_sepqkad_ref_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_sipcat_adgt01_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_att_qkvcat_adgtall_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/home_data/wj/ref_nms/cache/my_se_proposals_att_vanilla_refcocog_0923160805.pkl"
   
    #proposal_path = "/home_data/wj/ref_nms/cache/my_2se_proposals_att_vanilla_refcocog_0923230030.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_att_qkvcat_gt_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_1qkad_gt_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_1qkad_gt_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_sepqkad_gt_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_sepqkad_ad3gtall_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_sipqk_gt_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_qkad_adgtall__proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_qkad_gtctx_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_qkad_gtctx_proposals_att_vanilla_refcoco.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_1linearbert_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_ref_bertall_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_adatall_bertall_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_qkad_clsnms_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_qkad_gtall_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_qkad_gt001mulcls03_score_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_Sigad_gt_bertall_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_adatall_bertall_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_qkad_gtnms02_adcls_proposals_att_vanilla_refcoco+.pkl"
    
    #proposal_path ="/data1/wj/ref_nms/cache/my_qkad_gt005andcls005_proposals_att_vanilla_refcocog.pkl"

    #proposal_path = "/data1/wj/ref_nms/cache/my_qkad_gt001to0_nms02_adcls_proposals_att_vanilla_refcocog.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_qkad_gt001to0_nms_adcls02to0_proposals_att_vanilla_refcocog.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_qkad_gt_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/qkadgtcls_ad_003yb_proposals_att_vanilla_refcocog.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_bert_proposals_att_vanilla_refcocog.pkl"
    proposal_path = "/data1/wj/ref_nms/cache/my_Sigad_gt_bertall_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_Sigad_gtfx3(3)_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_gtfx2(4)_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_gtfx1(6)_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_gtfxqcut(0-3_4_5-9)_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_gtfxqcut(0_9)_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_gtfxqcut(0-1_2-5)_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_gtfxqcut(0-8_9_10-20)_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_adatall_bertall_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_gtall_bertall_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_bert_proposals_att_vanilla_refcoco.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_cocoin_gtsqrt_bertall_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_Sigad_cocoin_gtcbrt_bertall_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_cocoin_clssquar_bertall_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_cocoin_clssquar_out_clssqrt_bertall_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_Sigad_gtfxqcut(0.01_10)_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_Sigad_gtfxqcut(0.01_0.5_10)_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_Sigad_gtfxqcut(0.01_0.5_6)_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_Sigad_gt0.01_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_gtsqrt0.01_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_clssqrt0.01_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_gt0.01_3sqrt_weight_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_gt0.01_3cube_weight_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_gt0.01_3cube_weight_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_Sigad_cls0.5_gtcls_testAproposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_Sigad_cls0.5_gtnms_testAproposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_Sigad_cls0.3_gtcls_testAproposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_Sigad_cls0.2_gtlfcls_testAproposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_Sigad_cls0.3_gt(1+cls)_testAproposals_att_vanilla_refcoco+.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_Sigad_gt_bertall_proposals_att_vanilla_refcoco.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_Sigad_adatall_bertall_proposals_att_vanilla_refcoco.pkl"
    #proposal_path = "/data1/wj/ref_nms/cache/my_Sigad_cls0.3_gt+clsc30_testAproposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_gt0.01_3cube_weight_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_Sigad_gt0.01_adclsc100_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_qkad_gtcube_proposals_att_vanilla_refcoco+.pkl"
    #proposal_path ="/data1/wj/ref_nms/cache/my_bert_proposals_att_vanilla_refcocog.pkl"
    proposal_path = "/data1/wj/ref_nms/cache/my_bert_proposals_att_vanilla_refcoco.pkl"









   













    
    print('loading {} proposals from {}...'.format(args.m, proposal_path))
    with open(proposal_path, 'rb') as f:
        proposal_dict = pickle.load(f)
        #print("proposal_dict%s"%(proposal_dict,))
    # Load refer
    refer = REFER('/home_data/wj/ref_nms/data/refer', dataset=args.dataset, splitBy=args.split_by)
    # Evaluate hit rate
    print('Hit rate on {}\n'.format(dataset_splitby))
    #evaluator = NewHitRateEvaluator(refer, top_N=None, threshold=args.thresh)  #thresh how to use thresh?
    evaluator = NewHitRateEvaluator(refer, top_N=None, threshold=args.thresh)  #thresh how to use thresh?
    #conf = 0.1
    print('conf: {:.3f}'.format(args.conf))
    for split in eval_splits:
        #split = 'testB'
        exp_to_proposals = proposal_dict[split]
        #print("53exp_to_proposals%s"%type(exp_to_proposals))
        exp_to_proposals = threshold_with_confidence(exp_to_proposals, args.conf)  # first
        proposal_per_ref, hit_rate = evaluator.eval_hit_rate(split, exp_to_proposals)
        print('[{:5s}] hit rate: {:.2f} @ {:.2f}'.format(split, hit_rate*100, proposal_per_ref))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=str, default='att_vanilla')
    parser.add_argument('--dataset', default='refcoco') 
    parser.add_argument('--split-by', default='unc')
    parser.add_argument('--tid', type=str,default='0316212105' )# 1019213349,1019204514
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--conf', type=float, default=0.12) 
    main(parser.parse_args())
 