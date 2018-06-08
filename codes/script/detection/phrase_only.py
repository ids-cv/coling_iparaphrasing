import numpy as np
# import networkx as nx
from itertools import combinations, product
from parse import parse
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import recall_score, precision_score, f1_score
import os
from collections import defaultdict
import re
import pandas as pd

def load_sentence(txt_file):
    sents = []
    with open(txt_file, 'r') as f:
        for line in f:
            entities = []
            for item in re.findall(r'\[.*?\]', line):
                phr_id, category, phrase = parse('[/EN#{:d}/{} {}]', item)
                if category != 'notvisual':
                    entities.append({'id':phr_id, 'category': category, 'org_phrase': phrase})
            sents.append(entities)
    return sents

def load_preprocessed(line):

    items = parse('sen_{:d}: {} ## === {} ## ', line)
    if items is None:
        return None
    sent_id, ent_id, phr = items

    ent_id = ent_id.split(' ## ')
    ent_id = map(int, ent_id)
    phr = phr.split(' ## ')

    return {eid: p for eid, p in zip(ent_id, phr) if p != ''}
    
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

def save_results(result, saveto):
    img_id = [r[0] for r in result]
    phrase1 = ['/'.join((r[1], r[5])) for r in result]
    phrase2 = ['/'.join((r[2], r[6])) for r in result]
    y_pred = [r[3] for r in result]
    y_true = [r[4] for r in result]

    df = pd.DataFrame({'image': img_id, 'phrase1': phrase1, 'phrase2': phrase2, 'y_pred': y_pred, 'y_true': y_true})
    df.to_csv(saveto)

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

    # save_results(result, saveto+'.csv')

def main(feat_type):
    phr_score_file = 'output/tmp/phrase_score_%s_%s.csv'%(feat_type, 'val')
    print(phr_score_file)
    thres = find_best_thres(phr_score_file, saveto='%s_val'%feat_type)
    
    phr_score_file = 'output/tmp/phrase_score_%s_%s.csv'%(feat_type, 'test')
    print(phr_score_file)
    test_paraphrase_detection(phr_score_file, thres, saveto='%s_test'%feat_type)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_type', type=str, default='fv+cca', help="['fv+pca', 'fv+cca', 'fv']")
    args = parser.parse_args()

    main(args.feat_type)