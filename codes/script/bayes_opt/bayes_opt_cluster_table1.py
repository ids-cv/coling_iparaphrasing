import sys
import GPyOpt

sys.path.append('./codes/script/detection')
sys.path.append('./codes/script/eval')
from cluster_paraphrase import feat_based_cluster
from cluster_paraphrase import score_based_cluster
from eval_cluster import eval_ari_score

def wrapper_feat(params, method, eval_type):
    gs_file = 'ari_data/gt_cluster-label_val_%s.csv' % eval_type
    up_file = 'data/entity/val/uniquePhrases'
    feat_file = 'data/entity/val/textFeats_%s.npy' % method
    tmp_file = 'tmp/botmp_%s_%s.csv' % (method, eval_type)
    
    clutter = params[0]
    res_df = feat_based_cluster(up_file, gs_file, feat_file,
                               clutter=clutter, damping=0.5)
    score = eval_ari_score(gs_file, res_df)
    return -score

def wrapper_score_based(params, method, eval_type):
    gs_file = 'ari_data/gt_cluster-label_val_%s.csv' % eval_type
    pair_score_file = 'models/%s/res_val.csv' % (method)
    affinity_dir = 'output/af/%s/val/%s/' % (method, eval_type)
    
    clutter = params[0]
    res_df = score_based_cluster(gs_file, pair_score_file, affinity_dir,
                        clutter=clutter,damping=0.5)
    score = eval_ari_score(gs_file, res_df)
    return -score


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('paraphrase detection')
    parser.add_argument('--method', type=str, default='fv+pca', help='feature types (fv+pca or fv+cca)')
    parser.add_argument('--eval_type', type=str, default='all', help='evaluation type [all, single, multiple]')
    parser.add_argument('--max_iter', '-i', type=int, default=100,
                        help='maximum iteration for Bayesian optimization')
    args = parser.parse_args()

    bounds = []
    if args.method in ['fv+pca', 'fv+cca']:
        bounds = [{'name': 'clutter', 'type': 'continuous', 'domain': (0., 100.)}]
        wrapper = lambda x: wrapper_feat(x, args.method, args.eval_type) 
    else:
        bounds = [{'name': 'clutter', 'type': 'continuous', 'domain': (0., 100.)}]
        wrapper = lambda x: wrapper_score_based(x, args.method, args.eval_type)
        
    prob = GPyOpt.methods.BayesianOptimization(
        wrapper,
        bounds,
        acquisition_type='EI',
        normalize_Y=True,
        acquisition_weight=2
    )
    
    report_file = 'bo_output/report_%s+%s' % (args.method, args.eval_type)
    evaluations_file = 'bo_output/eval_%s+%s' % (args.method, args.eval_type)
    prob.run_optimization(args.max_iter, report_file=report_file, evaluations_file=evaluations_file)
