import os
import sys
sys.path.append('script/detection/')
from phrase_only import find_best_thres, test_paraphrase_detection

def main():
    base_dir = sys.argv[1]

    phr_score_file = os.path.join(base_dir, 'res_val.csv')
    thres = find_best_thres(phr_score_file, os.path.join(base_dir, 'pr_curv_val.png'))

    test_paraphrase_detection(os.path.join(base_dir, 'res_test.csv'), thres, saveto=os.path.join(base_dir, 'eval_w_tuned_thres_test'))


if __name__ == '__main__':
    main()