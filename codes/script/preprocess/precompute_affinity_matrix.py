import sys, os
sys.path.append('codes/script/detection')
from cluster_paraphrase import save_score_based_affinity

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('paraphrase detection')
    parser.add_argument('--method', type=str, default=None, help='method name')
    parser.add_argument('--pair_score_file', type=str, default=None, help='pair score file')
    parser.add_argument('--eval_type', type=str, default='all', help='evaluation type [all, single, multiple]')
    parser.add_argument('--split', type=str, default=None, help='data split [val, test]')
    args = parser.parse_args()
    
    gs_file = 'ari_data/gt_cluster-label_%s_%s.csv' % (args.split, args.eval_type)
    output_dir = 'output/af/%s/%s/%s/' % (args.method, args.split, args.eval_type)
    
    if args.pair_score_file is None:
        pair_score = 'output/res_%s_%s.csv' % (args.method, args.split)
    else:
        pair_score = args.pair_score_file
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_score_based_affinity(gs_file, pair_score, output_dir)