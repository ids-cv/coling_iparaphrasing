import numpy as np
import json
from itertools import combinations
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def preprocess_Mp2r(Mp2r, top_k):
    Mp2r[Mp2r==0] = np.nan # fill nan to non-score cells
    sort_i = Mp2r.argsort(axis=1)
    
    # select top-k regions
    for i, s in enumerate(sort_i):
        Mp2r[i, s[top_k:]] = np.nan
    smpl = np.logical_not(np.isnan(Mp2r).all(axis=0))
    smpl, = np.where(smpl)
    Mp2r = Mp2r[:, smpl]
    
    # distance to prob
    Mp2r = np.exp(- Mp2r)
    Mp2r[np.isnan(Mp2r)] = 0
    Mp2r /= Mp2r.sum(axis=1, keepdims=True)
    return Mp2r, smpl

def compute_phrase_feature_score(gt_pair_file, feat_type='fv+pca', split='val'):
    X = np.load('data/pl-clc_cca/entity/%s/textFeats_%s.npy' % (split, feat_type))

    p2i_dict = {}
    with open('data/pl-clc_cca/entity/%s/uniquePhrases'%split) as f:
        for i, line in enumerate(f):
            p2i_dict[line.rstrip()] = i

    gt_df = pd.read_csv(gt_pair_file)
    images = pd.unique(gt_df.image.values)
    scores = []
    for im in images:
        sub_df = gt_df[gt_df.image == im]
        pi_1 = [p2i_dict[p] for p in sub_df.phrase1]
        pi_2 = [p2i_dict[p] for p in sub_df.phrase2]

        X1 = X[pi_1]
        X1 / np.linalg.norm(X1, axis=1, keepdims=True)
        X2 = X[pi_2]
        X2 / np.linalg.norm(X2, axis=1, keepdims=True)
        score = (X1 * X2).sum(axis=1) / 2. + .5 
        scores.append(score.ravel())
    
    return np.hstack(scores)

def compute_localization_based_score(gt_pair_file, top_k = 30, split='val'):
    gt_df = pd.read_csv(gt_pair_file)
    images = pd.unique(gt_df.image.values)
    scores = []
    for im in images:
        X = np.load('data/pl-clc_cca/convert/phrase_region_score/cca+/%s/%i.npy'%(split, im))
        X_, _  = preprocess_Mp2r(X, top_k=top_k)
        Y = np.dot(X_, X_.T)

        graph_str = json.load(open('data/pl-clc_cca/convert/phrase_graph/fv+cca/%s/%i.json'%(split, im)))
        p2i = {p:i for i, p in enumerate(graph_str['phrases'])}

        sub_df = gt_df[gt_df.image == im]

        for _, row in sub_df.iterrows():
            phr1 = row['phrase1'].decode('utf-8')
            phr2 = row['phrase2'].decode('utf-8')
            score = Y[p2i[phr1], p2i[phr2]]
            scores.append(score)
    return scores

def compute_score(gt_pair_file, feat_type=None, top_k = 30, split='val'):
    
    score = compute_localization_based_score(gt_pair_file, top_k, split)
    
    if feat_type is not None:
        phr_score = compute_phrase_feature_score(gt_pair_file, feat_type=feat_type, split=split)
        score = np.asarray(score) + np.asarray(phr_score)
        score /= 2.

    df = pd.read_csv(gt_pair_file)
    df['score'] = score

    return df

def detect_paraphrase(phr_score_file):
    df = pd.read_csv(phr_score_file)
    y_true = df.ytrue
    y_pred = df.score
    return y_true, y_pred


def plot_prcurv(y_true, y_pred):
    precision, recall, thres = precision_recall_curve(y_true, y_pred)

    plt.clf()
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                    color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    return precision, recall, thres

def find_best_thres(phr_score_file, saveto='phrase_FV_res_val'):
    y_true, y_pred = detect_paraphrase(phr_score_file)

    # evaluate
    precision, recall, thres = plot_prcurv(y_true, y_pred)

    f1 = 2. * (precision * recall) / (precision + recall)
    f1[np.isnan(f1)] = 0.

    best_idx = f1.argmax()
    best_prec, best_rec, best_thres = precision[best_idx], recall[best_idx], thres[best_idx]

    plt.scatter(best_rec, best_prec)

    summary = 'best thres=%.4f\nf1=%.4f\nprec=%.4f\nrec=%.4f'%(best_thres, f1[best_idx], best_prec, best_rec)
    plt.text(.75, .8, summary)
    print(summary)
    
    plt.savefig((saveto+'.png'))

    return best_thres


def test_paraphrase_detection(phr_score_file, thres, saveto='phrase_FV_res_test'):
    y_true, y_pred = detect_paraphrase(phr_score_file)

    # evaluate
    precision, recall, _ = plot_prcurv(y_true, y_pred)

    best_prec = precision_score(y_true, y_pred > thres)
    best_rec = recall_score(y_true, y_pred > thres)
    f1 = f1_score(y_true, y_pred > thres)

    plt.scatter(best_rec, best_prec)
    summary = 'predefined thres=%.4f\nf1=%.4f\nprec=%.4f\nrec=%.4f'%(thres, f1, best_prec, best_rec)
    plt.text(.75, .8, summary)
    print(summary)
    plt.savefig((saveto+'.png'))

    # save results
    df = pd.read_csv(phr_score_file)
    df['ypred'] = y_pred > thres
    df.to_csv(saveto+'.csv')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_type', type=str, default=None)
    args = parser.parse_args()
    top_k = 30
    feat_type = args.feat_type

    phr_score_file = 'outputs/phrase_localization_based_phrase_pair_score_%s_val.csv'% feat_type
    df = compute_score('data/pl-clc_cca/convert/phrase_pair_remove_trivial_match_val.csv', feat_type, top_k, 'val')
    df.to_csv(phr_score_file)

    thres = find_best_thres(phr_score_file, saveto='outputs/phrase_localization_based_phrase_pair_score_%s_val'%feat_type)
    
    phr_score_file = 'outputs/phrase_localization_based_phrase_pair_score_%s_test.csv'%feat_type
    df = compute_score('data/pl-clc_cca/convert/phrase_pair_remove_trivial_match_test.csv', feat_type, top_k, 'test')
    df.to_csv(phr_score_file)

    test_paraphrase_detection(phr_score_file, thres, saveto='outputs/phrase_localization_based_phrase_pair_score_%s_test'%feat_type)

if __name__ == '__main__':
    main()