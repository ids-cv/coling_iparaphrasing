import os
import pandas as pd
import numpy as np

import sys, os
sys.path.append('codes/script/training/')
from train_paraphrase_classifier import EntityDatasetWordID
from chainer.iterators import SerialIterator
from sklearn.preprocessing import normalize


def wea_converter(batch):
    xp1 = [b[0] for b in batch]
    xp2 = [b[1] for b in batch]
    l = [b[2] for b in batch]
    return xp1, xp2, l

class PhraseEmb(object):
        
    def predict(self, split):
        raise NotImplementedError
    
    def save_score_file(self, split, out_file):
        df = self.predict(split)
        df.to_csv(out_file)
        
class WEA(PhraseEmb):

    def compute_score(self, split):
        phrase_file = 'data/phrase_pair_remove_trivial_match_%s.csv'
        word_dict_file = 'data/entity/word_dict'

        val = EntityDatasetWordID(phrase_file%split, word_dict_file)
        val_iter = SerialIterator(val, batch_size=100, repeat=False, shuffle=False)

        w_vec = np.load('data/entity/word_vec.npy')
        gt = []
        score = []

        for batch in val_iter:
            xp1, xp2, l = wea_converter(batch)

            avr1 = [np.mean(w_vec[x], axis=0) for x in xp1]
            avr2 = [np.mean(w_vec[x], axis=0) for x in xp2]
            avr1 = np.vstack(avr1)
            avr2 = np.vstack(avr2)
            score.append((normalize(avr1) * normalize(avr2)).sum(axis=1))
            gt += l
        return np.hstack(score), gt

    def predict(self, split):
        t_score, t_gt = self.compute_score(split)

        # write output file
        df = pd.read_csv('data/phrase_pair_remove_trivial_match_%s.csv'%split)
        df['score'] = (t_score / 2. + .5)
        return df
    
class FV_PCA(PhraseEmb):
    def __init__(self):
        self.feat_file = 'data/entity/%s/textFeats_fv+pca.npy'
        
    def compute_phrase2phrase_score(self, phrase_pair_file, uni_phrase_file, feat_file):
        with open(uni_phrase_file, 'r') as f:
            all_phrase = f.read()
            all_phrase = all_phrase.split()

        phrase_dict = {k: i for i, k in enumerate(all_phrase)}

        # load phrase feature
        Xp = np.load(feat_file)

        df = pd.read_csv(phrase_pair_file)

        score = []
        for _, row in df.iterrows():
            phrase1 = row['phrase1']
            phrase2 = row['phrase2']
            score.append((Xp[phrase_dict[phrase1]] * Xp[phrase_dict[phrase2]]).sum())

        df['score'] = pd.Series(score)
        return df
    
    def predict(self, split):
        df = self.compute_phrase2phrase_score('data/phrase_pair_remove_trivial_match_%s.csv'%split,
                                 'data/entity/%s/uniquePhrases'%split,
                                 self.feat_file % split)
        return df

class FV_CCA(FV_PCA):
    def __init__(self):
        self.feat_file = 'data/entity/%s/textFeats_fv+cca.npy'
        
def get_model_ensemble_df(res_file1, res_file2):

    res1 = pd.read_csv(res_file1)
    res2 = pd.read_csv(res_file2)

    p_score = res1.score.values
    i_score = res2.score.values

    ens_score = (p_score + i_score) * .5
    ens_res = res1.copy()
    ens_res.score = ens_score
    
    return ens_res