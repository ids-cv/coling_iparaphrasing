import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.cluster import AffinityPropagation
import progressbar
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import io
import codecs
import random

if sys.version_info < (3, 0):
    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
    sys.stdin = codecs.getreader('UTF-8')(sys.stdin)

def feat_based_cluster(uniquePhrases, gold, phraseFeats, clutter, damping, affinity='cosine', saveto=None):
    phrase_dic = read_phrase(uniquePhrases)
    gt_df = pd.read_csv(gold, encoding = "utf-8")
    feat_matrix = np.load(phraseFeats)

    img_list = gt_df.image.unique()
    bar = progressbar.ProgressBar()
    result = []
    for img in bar(img_list):
        gt_df_img = gt_df.query('image == %i' % img)

        feats = []
        phrases = []
        for _, item in gt_df_img.iterrows():
            phrase = item.phrase
            phrases.append(phrase)
            feats.append(feat_matrix[phrase_dic[phrase]])

        feats = np.vstack(feats)

        labels = []
        if len(phrases) > 1:
            if affinity == 'cosine':
                similarity = cosine_similarity(feats)
                pref = np.percentile(similarity, clutter)
                af = AffinityPropagation(preference=pref, affinity='precomputed', damping=damping)
                labels = af.fit_predict(similarity)
                    
            elif affinity == 'euclidean':
                distance = -euclidean_distances(feats, squared=True)
                pref = np.percentile(distance, clutter)
                af = AffinityPropagation(preference=pref, damping=damping)
                labels = af.fit_predict(feats)
            else:
                raise RuntimeError('invalid affinity metric')
            if np.isnan(labels).any(): # when af did not converge
                    labels = np.arange(labels.size)
        else:
            labels = [1]

        for i in range(len(phrases)):
            result.append({'image': img, 'phrase': phrases[i], 'label': labels[i]})
    return pd.DataFrame(result)

def score_based_cluster(gold, pair_score, affinity_save_path, clutter, damping):
    gt_df = pd.read_csv(gold, encoding = "utf-8")

    img_list = gt_df.image.unique()

    bar = progressbar.ProgressBar()
    result = []
    for img in bar(img_list):
        gt_df_img = gt_df.query('image == %i' % img)

        phrases = []
        for _, item in gt_df_img.iterrows():
            phrase = item.phrase
            phrases.append(phrase)

        scores = np.load(affinity_save_path+'/'+str(img)+'.npy')
        if scores.size > 1:
            pref = np.percentile(scores, clutter)
            af = AffinityPropagation(preference=pref, affinity='precomputed', damping=damping)
            labels = af.fit_predict(scores)
            if np.isnan(labels).any(): # when af did not converge
                    labels = np.arange(labels.size)
        else:
            labels = [1]
            
        for i in range(len(phrases)):
            result.append({'image': img, 'phrase': phrases[i], 'label': labels[i]})
    return pd.DataFrame(result)

def save_score_based_affinity(gold, pair_score, affinity_save_path):
    gt_df = pd.read_csv(gold, encoding="utf-8")
    phair_score_df = pd.read_csv(pair_score, encoding="utf-8")

    img_list = gt_df.image.unique()
    
    # bar = progressbar.ProgressBar()

    # for img in bar(img_list):
    for img in img_list:
        gt_df_img = gt_df.query('image == %i' % img)
        phair_score_df_img = phair_score_df.query('image == %i' % img)

        phrases = []
        scores = []

        for _, item in gt_df_img.iterrows():
            phrase1 = item.phrase
            phrases.append(phrase1)
            score_list = []
            for _, item in gt_df_img.iterrows():
                phrase2 = item.phrase
                if phrase1 == phrase2:
                    score_list.append(1)
                else:
                    score = 0.0
                    phair_score_df_img_phrase_pair_direct = phair_score_df_img[(phair_score_df_img['phrase1'] == phrase1)
                                                                               & (phair_score_df_img['phrase2'] == phrase2)]
                    phair_score_df_img_phrase_pair_inverse = phair_score_df_img[(phair_score_df_img['phrase2'] == phrase1)
                                                                                & (phair_score_df_img['phrase1'] == phrase2)]
                    for _, item in phair_score_df_img_phrase_pair_direct.iterrows():
                        score = item.score
                    for _, item in phair_score_df_img_phrase_pair_inverse.iterrows():
                        score = item.score
                    score_list.append(score)
            scores.append(score_list)

        scores = np.vstack(scores)
        np.save(affinity_save_path+'/'+str(img), scores)

def lower_bound(gold, saveto, method):
    gt_df = pd.read_csv(gold, encoding="utf-8")
    img_list = gt_df.image.unique()
    bar = progressbar.ProgressBar()
    result = []
    for img in bar(img_list):
        gt_df_img = gt_df.query('image == %i' % img)
        phrases = []
        labels = []
        label = 0
        for _, item in gt_df_img.iterrows():
            phrase = item.phrase
            phrases.append(phrase)

            if method == 'uniform':
                label += 1
            elif method == 'random':
                label = random.randint(1, len(gt_df_img))
            elif method == 'unique':
                label = 1

            labels.append(label)

        for i in range(len(phrases)):
            result.append({'image': img, 'phrase': phrases[i], 'label': labels[i]})

    save(result, saveto)

def read_phrase(phrase_file):
    phrase_dic = {}
    line_num = 0
    with io.open(phrase_file, encoding = "utf-8") as f:
        for phrase in f:
            phrase_dic[phrase.strip()] = line_num
            line_num += 1
    return phrase_dic

def save(result, saveto):
    if saveto is not None:
        res_df = pd.DataFrame({
            'image': [r['image'] for r in result],
            'phrase': [r['phrase'] for r in result],
            'label': [r['label'] for r in result]
        })
        res_df.to_csv(saveto)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('paraphrase detection')
    parser.add_argument('--gold', type=str, default=None, help='ground truth file')
    parser.add_argument('--uniquePhrases', type=str, default=None, help='unique phrases file')
    parser.add_argument('--phraseFeats', type=str, default=None, help='phrase features file')
    parser.add_argument('--clutter', type=float, default=5,
                        help='parameter to controll cluster numbers.')
    parser.add_argument('--damping', type=float, default=0.5,
                        help='parameter to controll the extent to which the current value is maintained relative to incoming values.')
    parser.add_argument('--affinity', type=str, default='cosine',
                        help='set to the affinity to either cosine or euclidean for feature based clustering')
    parser.add_argument('--lower_bound', type=str, default=None,
                        help='compute the culstering lower bound, set either uniform, random or unique')
    parser.add_argument('--pair_score', type=str, default=None,
                        help='set a file containing phrase pair scores')
    parser.add_argument('--score_based_mode', type=str, default='cluster',
                        help='set to either "cluster" for clustering, to "save" for saving affinity')
    parser.add_argument('--affinity_save_path', type=str, default=None,
                        help='save affinity matrix under this path')
    parser.add_argument('--saveto', type=str, default=None,
                        help='file name to save the predict labels. default None')
    args = parser.parse_args()

    if args.lower_bound:
        lower_bound(gold=args.gold, saveto=args.saveto, method=args.lower_bound)
    elif args.pair_score:
        if args.score_based_mode == 'save':
            save_score_based_affinity(gold=args.gold, pair_score=args.pair_score,
                                      affinity_save_path=args.affinity_save_path)
        elif args.score_based_mode == 'cluster':
            score_based_cluster(gold=args.gold, pair_score=args.pair_score, affinity_save_path=args.affinity_save_path,
                clutter=args.clutter, damping=args.damping, saveto=args.saveto)
    else:
        feat_based_cluster(uniquePhrases=args.uniquePhrases, gold=args.gold, phraseFeats=args.phraseFeats,
            clutter=args.clutter, damping=args.damping, affinity=args.affinity, saveto=args.saveto)


