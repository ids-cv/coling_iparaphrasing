import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import recall_score, precision_score, f1_score
import json
import progressbar

import sys
# sys.path.append('3rd_party/pygco/')
sys.path.append('/home/chu/tools/pygco')
# sys.path.append('/Users/chu/Documents/work/tools/pygco')
reload(sys)
sys.setdefaultencoding('utf-8')
import pygco
import codecs

if sys.version_info < (3, 0):
    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
    sys.stdin = codecs.getreader('UTF-8')(sys.stdin)

def get_region2label_table(X, clutter, damping, metric='cosine'):
    '''
    metric: cosine | iou
    '''
    # compute affinity
    if metric == 'cosine':
        A = cosine_similarity(X)
        A = A / 2. + .5
    elif metric == 'iou':
        raise RuntimeError
    
    pref = np.percentile(A, clutter)

    # bbox clustering
    af = AffinityPropagation(preference=pref, affinity='precomputed', damping=damping)
    af.fit(A)

    # p(l|r)
    # mat of N_label x N_region
    Tcr = A[:, af.cluster_centers_indices_]
    Tcr /= Tcr.sum(axis=1, keepdims=True)
    Tcr = Tcr.T
    
    return Tcr

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

def detect_paraphrase(
    phrase_pair_file,
    graph_dir,
    p2r_dir,
    rfeat_dir,
    p2r_method,
    r2r_method,
    top_k = 5,
    clutter = 90,
    damping = 0.5,
    k = .1,
    thresh = .58,
    saveto = None,
    wo_phrase_connection=False):

    phrase_pair_df = pd.read_csv(phrase_pair_file, encoding = "utf-8")

    bar = progressbar.ProgressBar()
    y_true = []
    y_pred = []
    items = []

    for fn in bar(os.listdir(graph_dir)):
        graph_data = json.load(open(os.path.join(graph_dir, fn)))
        edges = np.asarray(graph_data['edges'])
        weights = np.asarray(graph_data['weights'])
        weights = weights / 2. + .5
        phrase = graph_data['phrases']
        
        # preprocess graph
        edges = edges[weights > thresh]
        weights = weights[weights > thresh]
        
        Mp2r = np.load(os.path.join(p2r_dir, fn[:-4]+'npy'))
        Xr = np.load(os.path.join(rfeat_dir, fn[:-4]+'npy'))
        r_ids = np.arange(len(Xr))
        
        Mp2r, smpl = preprocess_Mp2r(Mp2r, top_k)
        Xr = Xr[smpl]
        Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
        r_ids[smpl]
        
        # cluster regions
        Tcr = get_region2label_table(Xr, clutter, damping, metric=r2r_method)
        
        # p(l | i)
        # Table. size N_label x N_phrase
        Tli = np.dot(Tcr, Mp2r.T)

        unary = - np.log(Tli.T)
        pairwise = (1 - np.eye(unary.shape[-1]))

        # optimize labels
        init_labels = unary.argmin(axis=1)
        N_label = len(Tcr)
        
        if wo_phrase_connection:
            labels = init_labels
        elif (len(edges) > 0) and (N_label > 1):
            labels = pygco.cut_general_graph(edges, k * weights, unary,  pairwise, n_iter=-1, algorithm='swap')
        else: # if there is no edges between any phrases or only one cluster is found
            labels = init_labels

        # eval results
        p2i = {p:i for i, p in enumerate(phrase)}
        sub_df = phrase_pair_df[phrase_pair_df.image == int(fn[:-5])]

        for p in phrase:
            items.append((int(fn[:-5]), p, labels[p2i[p]]))

        for _, row in sub_df.iterrows():
            p1 = row['phrase1']
            p2 = row['phrase2']
            y_true.append(row['ytrue'])
            y_pred.append(labels[p2i[p1]] == labels[p2i[p2]])

    if saveto:
        res_df = pd.DataFrame(items, columns=['image', 'phrase', 'label'])
        res_df.to_csv(saveto)

    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return f1, prec, rec

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('paraphrase detection')
    parser.add_argument('--validation', '-v', action='store_true')
    parser.add_argument('--test', '-t', action='store_true')
    parser.add_argument('--k', '-k', type=float, default=1.,
    help = 'parameter to controll effects of text-text connections')
    parser.add_argument('--clutter', '-c', type=float, default=95,
    help = 'parameter to controll cluster numbers.')
    parser.add_argument('--damping', type=float, default=0.5,
    help='parameter to controll the extent to which the current value is maintained relative to incoming values.')
    parser.add_argument('--max_n', '-n', type=int, default=None,
    help = 'maximum number of images to be tested')
    parser.add_argument('--saveto', '-s', type=str, default=None,
    help='file name to write detected paraphrases. default None')
    parser.add_argument('--data_path', '-d', type=str, default=None,
                        help='data path')
    parser.add_argument('--region_num', '-r', type=int, default=10,
                        help='number of candidate regions per phrase')
    parser.add_argument('--thresh', '-th', type=float, default=.58,
                        help='threshold for phrase similarity')
    parser.add_argument('--phrase_off', action='store_true')
    args = parser.parse_args()
    
    if args.validation:
        split = 'val'
    elif args.test:
        split = 'test'
    else:
        raise RuntimeError('set option -v or -t (validation or test)')

    phrase_pair_file = '/home/mayu-ot/Documents/iparaphrasing/data/pl-clc_cca/convert/phrase_pair_remove_trivial_match_%s.csv'%split
    graph_dir = '/home/mayu-ot/Documents/iparaphrasing/data/pl-clc_cca/convert/phrase_graph/fv+cca/%s/'%split
    p2r_dir = '/home/mayu-ot/Documents/iparaphrasing/data/pl-clc_cca/convert/phrase_region_score/cca+/%s/'%split
    rfeat_dir = '/home/mayu-ot/Documents/iparaphrasing/data/pl-clc_cca/region/fv+cca/%s/'%split

    f1, prec, rec = detect_paraphrase(phrase_pair_file, graph_dir, p2r_dir, rfeat_dir, p2r_method='cca+',
                                      r2r_method='cosine', top_k=args.region_num, clutter=args.clutter,
                                      damping=args.damping, k=args.k, thresh=.58, saveto=args.saveto,
                                      wo_phrase_connection=args.phrase_off)

    print('f1: %.2f  prec: %.2f  rec: %.2f'%(f1*100, prec*100, rec*100))