
import os
import sys
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import pairwise_distances

# sys.path.append('3rd_party/pygco/')
sys.path.append('/home/chu/tools/pygco')
sys.path.append('/Users/chu/Documents/work/tools/pygco')
import pygco



from sklearn.metrics import euclidean_distances
from itertools import combinations
import progressbar

def cluster_bbox(X, clutter=1):
    A = -euclidean_distances(X, squared=True)
    pref = np.percentile(A, clutter)
    
    # bbox clustering
    af = AffinityPropagation(preference=pref)
    af.fit(X)
    
    # membership function
    if(af.cluster_centers_.ndim == 3):
        af.cluster_centers_ = af.cluster_centers_.reshape((af.cluster_centers_.shape[0], -1))
    M_mem = np.sqrt(pairwise_distances(X, af.cluster_centers_))
    M_mem = np.exp(- M_mem)
    M_mem /= M_mem.sum(axis=1, keepdims=True)
    return M_mem, af.cluster_centers_
        
class ParaphraseDetector(object):
    
    def __init__(self, image, t2t_dataframe, t2r_dataframe, region_num=5, clutter=1, visual=False):
        '''
        image: id of image
        t2t_dataframe: pandas DataFrame. includes phrase2phrase scores.
        t2r_dataframe: pandas DataFrame. includes phrase2region scores.

        (Attributes)
        G:
        a networkx graph containing phrases, regions, and concepts (omegas).
        
        candidates:
        possible paraphrases for the image
        '''
        self.clutter = clutter
        self.cluster_center = None
        self.region_num = region_num
        self.visual = visual

        self.G = nx.DiGraph()
        
        self.t2t_df = t2t_dataframe.query('image == %i' % image)
        self.t2r_df = t2r_dataframe.query('image == %i' % image)
        
        self.candidates = []
        
        # register phrases
        self.register_phrases()

        # register regions
        self.register_regions()

    def register_phrases(self):
        for _, item in self.t2t_df.iterrows():
            # if visual is set: skip non-visual entities
            if(self.visual and (item.region1 == 0 or item.region2 == 0)):
                continue
            phr1 = item.phrase1
            phr2 = item.phrase2

            if phr1 == phr2:  # skip exact match
                continue

            if not self.G.has_node(phr1):
                self.G.add_node(phr1, ntype='phrase')

            if not self.G.has_node(phr2):
                self.G.add_node(phr2, ntype='phrase')

            if self.G.has_edge(phr1, phr2) or self.G.has_edge(phr2, phr1):
                continue

            weight = self.get_t2t_weight(phr1, phr2)
            self.G.add_edge(phr1, phr2, weight=weight)

            self.candidates.append((phr1, phr2))

            # remove edges with zero-weight
            zero_edges = [(n1, n2) for n1, n2, d in self.G.edges(data=True) if d['weight'] == 0]
            self.G.remove_edges_from(zero_edges)

    def register_regions(self):
        phr_list = self.t2r_df.phrase.unique()

        # print self.region_num
        for phr in phr_list:
            t2r_df_phr = self.t2r_df.query('phrase == """%s"""' % phr)
            # print phr
            count = 1
            for id, item in t2r_df_phr.iterrows():
                # print id
                bbox_id = item.bbox
                phr = item.phrase

                if not self.G.has_node(bbox_id):
                    bbox = item[['ymin', 'xmin', 'ymax', 'xmax']].values
                    bbox = bbox.astype(np.float)
                    self.G.add_node(bbox_id, cord=bbox, ntype='region')

                if not self.G.has_edge(phr, bbox_id):
                    weight = self.get_t2r_weight(phr, bbox_id)
                    self.G.add_edge(phr, bbox_id, weight=weight)

                count += 1
                if count > self.region_num:
                    count = 1
                    break

        # MEMO: same phrase in different sentences have different bbox_ids and scores. currently it simply
        # uses the phrase appears first in the data and its bbox_ids and scores
        # normalize t2r scores
        for phr in self.get_phrases():
            adj_e = [edge for edge in self.G.adj[phr].items() if self.G.node[edge[0]]['ntype'] == 'region']
            z = sum([e[1]['weight'] for e in adj_e])
            for e in adj_e:
                self.G[phr][e[0]]['weight'] = e[1]['weight'] / z

    def _cluster_regions(self):
        r_nodes = []
        X = []
        for node, ntype in self.G.nodes.data('ntype'):
            
            if ntype == 'region':
                cord = self.G.node[node]['cord']
                X.append(cord)
                r_nodes.append(node)
                
        X = np.vstack(X)
        # update membership func
        Mmem, cc_bbox = cluster_bbox(X, self.clutter)
        self.cluster_center = cc_bbox
        
        N_w =Mmem.shape[1]
        Ws = ['w%i'%i for i in range(N_w)]
        # when only visual, w_nv should not be needed
        # but it is possible that a scene/other entity is not localized to any region
        # therefore, we simply keep this node
        Ws.append('w_nv') # dummy node for non-visual concept
        self.G.add_nodes_from(Ws, ntype='omega')
        
        for node, mem in zip(r_nodes, Mmem):
            for w, m in zip(Ws, mem):
                self.G.add_edge(node, w, weight=m)
    
    def get_cluster_centers(self):
        '''
        return representative bboxies corresponding to each omega
        '''
        if self.cluster_center is None:
            raise RuntimeError('cluster centers are not computed')
        return self.cluster_center
    
    def get_phrases(self):
        '''
        return all phrases annotated to the image
        '''
        return [node for node, ntype in self.G.nodes.data('ntype') if ntype == 'phrase']
    
    def get_regions(self):
        '''
        return all region bboxies
        '''
        return [node for node, ntype in self.G.nodes.data('ntype') if ntype == 'region']
    
    def get_omegas(self):
        '''
        return names of all concepts (omegas) corresponding to region clusters.
        '''
        return [node for node, ntype in self.G.nodes.data('ntype') if ntype == 'omega']
    
    def _setup_t2w(self):
        phrases = self.get_phrases()
        omegas = self.get_omegas()
        
        for phr, omega in product(phrases, omegas):
            weight = self.get_t2w_weight(phr, omega)
            self.G.add_edge(phr, omega, weight=weight)
    
    def get_phrase_omega_graph(self):
        '''
        prepare phrase-omega graph.
        resulting graph is used as input of graph cut method
        '''
        self._cluster_regions()
        self._setup_t2w()
        
        G = self.G.copy()
        for node, ntype in self.G.nodes.data('ntype'):
            if ntype == 'region':
                G.remove_node(node)
        return G
        
    def get_t2w_weight(self, phr, omega):
        '''
        compute p( omega | phr )
        '''
        r_nodes = [node for node in self.G.adj[phr] if self.G.node[node]['ntype'] == 'region']

        # if no regions are registered, connect to non-visual omega
        if (len(r_nodes) == 0) and (omega == 'w_nv'):
            return 1
        
        p = 0
        for rn in r_nodes:
            if self.G.has_edge(rn, omega):
                p += self.G[phr][rn]['weight'] * self.G[rn][omega]['weight']
        return p
            
    
    def get_t2t_weight(self, phr1, phr2):
        '''
        get translation prob.
        '''
        phrase_pair = (phr1, phr2)
        
        res = self.t2t_df.query('(phrase1 == """%s""") and (phrase2 == """%s""")' % phrase_pair)
        if res.empty:
            res = self.t2t_df.query('(phrase2 == """%s""") and (phrase1 == """%s""")' % phrase_pair)
        return res.score.iloc[0]
    
    def get_t2r_weight(self, phr, bbox_id):
        '''
        get a score between a phrase and an image region.
        the score is converted by np.exp(-score).
        '''
        # get text to region scores
        match = self.t2r_df.query('(bbox == %i) and (phrase == """%s""")' % (bbox_id, phr))
        
        score = match.score.values[0] # at this point, a smaller score means higher probability of correspondence
        score = np.exp( - score)
        
        return score
    
    def cvrt2gco(self, Gtw):
        '''
        convert networkx graph to input arrays for gco.
        '''
        eps = 2e-5
        
        phrases = self.get_phrases()
        node2int = {node: i for node, i in zip(phrases, range(len(phrases)))}
        
        # extract edges and weights
        edges = []
        weights = []
        for n1, n2 in Gtw.edges:
            if all([Gtw.nodes[n]['ntype'] == 'phrase' for n in (n1, n2)]):
                e = (node2int[n1], node2int[n2])
                edges.append((min(e), max(e)))
                weights.append(Gtw[n1][n2]['weight'])
                
        edges = np.asarray(edges, dtype=np.int32)
        weights = np.asarray(weights, dtype=np.float64)
        
        # compute unary_cost
        omegas = self.get_omegas()
        unary_cost = np.zeros((len(node2int), len(omegas)), dtype=np.float64)
        
        for node, idx in node2int.items():
            unary = np.asarray([Gtw[node][w]['weight'] for w in omegas])

            unary_cost[idx] = unary / sum(unary)
        
        # equaly penalize different labels in neighbors
        pairwise = (1 - np.eye(len(omegas)))
        weights = np.exp(weights)
        unary_cost = - np.log(unary_cost + eps)
        
        return edges, weights, unary_cost, pairwise, node2int
    
    def iscandidates(self, pairs):
        '''
        filter paraphrases
        '''
        candidates = [set(pair) for pair in self.candidates]
        res = []
        for pair in pairs:
            spair = set(pair)
            if any([spair == cpair for cpair in candidates]):
                res.append(pair)
        return res
    
def detect_paraphrase(k=.8, clutter=50, region_num=5, split='val', data_path=None, max_n=None, saveto=None, verbose=False, visual=False):
    '''
    k: float. parameter to balance the data term and the smooth term.
    clutter: float in range of [0,100]. control the number of clusters. larger number results in more clusters.
    '''

    t2t_df = pd.read_csv(data_path+'/alignment_%s.csv' % split)
    t2r_df = pd.read_csv(data_path+'/entity_region_scores_%s.csv' % split)

    if split == 'test':
        with open(data_path+'/filtered_test_id.txt', 'r') as f:
            img_list = [int(line.strip()) for line in f]
    else:
        img_list = t2t_df.image.unique()

    # test first max_n images
    img_list = img_list[:max_n]
    # img_list = [102851549]
        
    result = []

    # setup graph
    bar = progressbar.ProgressBar()

    N_gt = 0 # number of ground truth paraphrases

    for img in bar(img_list):
        p_detector = ParaphraseDetector(img, t2t_df, t2r_df, region_num, clutter=clutter, visual=visual)
        Gtw = p_detector.get_phrase_omega_graph()
        edges, weights, unary, pairwise, node2int = p_detector.cvrt2gco(Gtw)
        init_labels = unary.argmin(axis=1)

        # optimize
        if edges.size: # if there is no edges between any phrases
            labels = pygco.cut_general_graph(edges, weights, unary, k * pairwise, init_labels=init_labels, n_iter=-1, algorithm='swap')
        else:
            labels = init_labels

        detected_phrase = []
        for l in np.unique(labels):
            nodes = [n for n in node2int if labels[node2int[n]] == l]
            detected = p_detector.iscandidates(combinations(nodes, 2)) # remove phrase pair from the same sentence
            detected_phrase += detected
        
        

        # check if each detected paraphrase is correct
        df = t2t_df.query('image == %i' % img)

        for phrase_pair in detected_phrase:
            res = df.query('(phrase1 == """%s""") and (phrase2 == """%s""")' % phrase_pair)
            occurrence = len(res)
            gt_count = sum(res.region1 == res.region2)
            
            res = df.query('(phrase2 == """%s""") and (phrase1 == """%s""")' % phrase_pair)
            occurrence += len(res)
            gt_count += sum(res.region1 == res.region2)

            result.append({'image': img, 'phrase': phrase_pair, 'occurrence': occurrence, 'score': gt_count})

        # count ground truth paraphrase
        if visual:
            N_gt += len(df.query('(phrase1 != phrase2) and (region1 == region2) and (region1 != 0) and (region2 != 0)'))
        else:
            N_gt += len(df.query('(phrase1 != phrase2) and (region1 == region2)' ))
    
    # compute evaluation scores
    N_ans = sum([r['occurrence'] for r in result])
    N_tp = sum([r['score'] for r in result])
    prec = N_tp * 1. / N_ans
    rec = N_tp * 1. / N_gt
    f1 = 2. * (prec * rec) / (prec + rec)

    if verbose:
        print('No. prediction:', N_ans, ', No. true positive', N_tp, ', No. gt positives', N_gt)
        print('prec = %.4f rec = %.4f f1 = %.4f' % (prec, rec, f1))

    if saveto is not None:
        res_df = pd.DataFrame({
                                'image': [r['image'] for r in result],
                                'phrase1': [r['phrase'][0] for r in result],
                                'phrase2': [r['phrase'][1] for r in result],
                                'occurrence': [r['occurrence'] for r in result],
                                'score': [r['score'] for r in result]
                                })
                                
        res_df.to_csv(saveto)

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
    parser.add_argument('--max_n', '-n', type=int, default=None,
    help = 'maximum number of images to be tested')
    parser.add_argument('--saveto', '-s', type=str, default=None,
    help='file name to write detected paraphrases. default None')
    parser.add_argument('--data_path', '-d', type=str, default=None,
                        help='data path')
    parser.add_argument('--region_num', '-r', type=int, default=None,
                        help='number of candidate regions per phrase')
    parser.add_argument('--visual', action='store_true',
                        help='only evaluate visual entities')
    args = parser.parse_args()
    
    if args.validation:
        split = 'val'
    elif args.test:
        split = 'test'
    else:
        raise RuntimeError('set option -v or -t (validation or test)')
    
    detect_paraphrase(k=args.k, clutter=args.clutter, region_num=args.region_num, split=split, data_path=args.data_path,
                      max_n=args.max_n, saveto=args.saveto, verbose=True, visual=args.visual)
