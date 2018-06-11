import matplotlib
matplotlib.use('Agg')
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
import numpy as np
import os
from chainer.iterators import SerialIterator
from chainer import function
from chainer import cuda
from chainer.dataset.convert import concat_examples
import pandas as pd
from collections import defaultdict
from chainer import dataset
from imageio import  imread
from chainer import initializers
from chainer.dataset import iterator
from chainer.dataset.convert import to_device
import six.moves.cPickle as pickle
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime as dt

class SampleManager(iterator.Iterator):
    def __init__(self, dataset, batch_size, p_batch_ratio, repeat=True, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.p_batch_ratio = p_batch_ratio
        self._repeat = repeat
        self._shuffle = shuffle
        
        # split dataset into positive and negative samples
        # This process will be slow if the dataset is not preloaded
        self._porder = [i for i in range(len(self.dataset)) if self.dataset._label[i]]
        self._norder = [i for i in range(len(self.dataset)) if not self.dataset._label[i]]
        print('Positive: %i, Negative: %i' % (len(self._porder), len(self._norder)))
        
        self.reset()
        
    def reset(self):
        if self._shuffle:
            np.random.shuffle(self._porder)
            np.random.shuffle(self._norder)

        self.p_current_position = 0
        self.n_current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        p_batch_size = int(self.batch_size * self.p_batch_ratio)
        n_batch_size = self.batch_size - p_batch_size
        
        i_p = self.p_current_position
        i_p_end = i_p + p_batch_size
        
        i_n = self.n_current_position
        i_n_end = i_n + n_batch_size
        
        Np = len(self._porder)
        Nn = len(self._norder)

        p_batch = self.dataset[self._porder[i_p:i_p_end]]
        n_batch = self.dataset[self._norder[i_n:i_n_end]]

        if i_p_end >= Np:
            if self._repeat:
                rest = i_p_end - Np
                
                if self._shuffle:
                    np.random.shuffle(self._porder)
                    np.random.shuffle(self._norder)
                if rest > 0:
                    p_batch.extend([self.dataset[index]
                                      for index in self._porder[:rest]])
                self.p_current_position = rest
            else:
                self.p_current_position = 0
                self.n_current_position = 0

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.p_current_position = i_p_end
            self.n_current_position = i_n_end

        # reset negative pairs when the reading head reaches the end of negative samples
        if i_n_end >= Nn:
            if self._repeat:
                rest = i_n_end - Nn
                
                if self._shuffle:
                    np.random.shuffle(self._norder)
                if rest > 0:
                    n_batch.extend([self.dataset[index]
                                      for index in self._norder[:rest]])
                self.n_current_position = rest
            else:
                self.n_current_position = 0

        return p_batch + n_batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.p_current_position * 1. / len(self._porder)

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        self.p_current_position = serializer('p_current_position',
                                           self.p_current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        self.p_batch_ratio = serializer('p_batch_ratio', self.p_batch_ratio)
        if self._porder is not None:
            try:
                serializer('porder', self._porder)
                serializer('norder', self._norder)
            except KeyError:
                serializer('_porder', self._porder)
                serializer('_norder', self._norder)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / len(self.dataset)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.

class EntityDatasetBase(dataset.DatasetMixin):
    def __init__(self, data_file, san_check=False, skip=None):
        df = pd.read_csv(data_file)

        self._image_id = df.image.values
        self._phrase1 = df.phrase1.values
        self._phrase2 = df.phrase2.values
        self._label = df.ytrue.values

        if san_check:
            print('sanity chack mode')
            self._image_id = self._image_id[::2000]
            self._phrase1 = self._phrase1[::2000]
            self._phrase2 = self._phrase2[::2000]
            self._label = self._label[::2000]

        if skip is not None:
            print('sample data with skip %i'%skip)
            self._image_id = self._image_id[::skip]
            self._phrase1 = self._phrase1[::skip]
            self._phrase2 = self._phrase2[::skip]
            self._label = self._label[::skip]  

    def __len__(self):
        return self._label.size
    
    def _get_entity(self, i):
        raise NotImplementedError
    
    def _get_label(self, i):
        return self._label[i]
    
    def get_example(self, i):
        
        # get phrase feature
        x1, x2 = self._get_entity(i)
        
        # get label
        y = self._get_label(i)
        
        return x1, x2, y

class EntityDatasetWordID(EntityDatasetBase):
    def __init__(self, data_file, word_dict_file, san_check=False, skip=None):
        super(EntityDatasetWordID, self).__init__(data_file, san_check=san_check, skip=skip)
        
        self._word_dict = pickle.load(open(word_dict_file, 'rb'), encoding='latin1')
    
    def _get_entity(self, i):
        x1 = [self._word_dict[w] for w in self._phrase1[i].split('+')]
        x2 = [self._word_dict[w] for w in self._phrase2[i].split('+')]
        return x1, x2
    

class EntityDatasetPhraseFeat(EntityDatasetBase):
    def __init__(self, data_file, phrase_feature_file, unique_phrase_file, san_check=False, skip=None):
        super(EntityDatasetPhraseFeat, self).__init__(data_file, san_check=san_check, skip=skip)
        
        phrase2id_dict = defaultdict(lambda: -1)
        with open(unique_phrase_file) as f:
            for i, line in enumerate(f):
                phrase2id_dict[line.rstrip()] = i
        
        self._p2i_dict = phrase2id_dict
        self._feat = np.load(phrase_feature_file).astype(np.float32)

    def _get_entity(self, i):
        x1 = self._feat[self._p2i_dict[self._phrase1[i]]]
        x2 = self._feat[self._p2i_dict[self._phrase2[i]]]
        return x1, x2

class ImageEntityDatasetBase(EntityDatasetBase):
    def __init__(self, data_file, img_root, san_check=False, preload=False, skip=None):
        super(ImageEntityDatasetBase, self).__init__(data_file, san_check=san_check, skip=skip)
        self._root = img_root
        self._image_arr = None

        if preload:
            unique_image = pd.unique(self._image_id)
            self._image_idx = {im_id: i for i, im_id in enumerate(unique_image)}
            print('loading %i images ...' % len(unique_image))
            self._image_arr = [imread(os.path.join(self._root, '%i.jpg'% x)).astype(np.float32) for x in unique_image]
            print('complete')
    
    def _get_image(self, i):
        if self._image_arr is None:
            path = os.path.join(self._root, '%i.jpg'%self._image_id[i])
            image = imread(path).astype(np.float32)
        else:
            image = self._image_arr[self._image_idx[self._image_id[i]]]

        if image.ndim == 2:
            # image is greyscale
            image = image[:, :, numpy.newaxis]
        
        image = image.transpose(2, 0, 1)
        return image
    
    def get_example(self, i):
        # get image
        image = self._get_image(i)
        
        # get phrase feature
        x1, x2 = self._get_entity(i)
        
        # get label
        y = self._get_label(i)
        
        return image, x1, x2, y

class ImageEntityDatasetWordID(ImageEntityDatasetBase):
    def __init__(self, data_file, word_dict_file, img_root, san_check=False, preload=False, skip=None):
        super(ImageEntityDatasetWordID, self).__init__(data_file, img_root, san_check=san_check, preload=preload, skip=skip)
        
        self._word_dict = pickle.load(open(word_dict_file, 'rb'), encoding='latin1')
    
    def _get_entity(self, i):
        x1 = [self._word_dict[w] for w in self._phrase1[i].split('+')]
        x2 = [self._word_dict[w] for w in self._phrase2[i].split('+')]
        return x1, x2
    

class ImageEntityDatasetPhraseFeat(ImageEntityDatasetBase):
    def __init__(self, data_file, phrase_feature_file, unique_phrase_file, img_root, san_check=False, preload=False, skip=None):
        super(ImageEntityDatasetPhraseFeat, self).__init__(data_file, img_root, san_check=san_check, preload=preload, skip=skip)
        
        phrase2id_dict = defaultdict(lambda: -1)
        with open(unique_phrase_file) as f:
            for i, line in enumerate(f):
                phrase2id_dict[line.rstrip()] = i
        
        self._p2i_dict = phrase2id_dict
        self._feat = np.load(phrase_feature_file).astype(np.float32)

    def _get_entity(self, i):
        x1 = self._feat[self._p2i_dict[self._phrase1[i]]]
        x2 = self._feat[self._p2i_dict[self._phrase2[i]]]
        return x1, x2
    
def my_converter(batch, device=None):
    Xim = [b[0] for b in batch]
    # # Comment out to enable data augmentation
    Xp1 = [b[1] for b in batch]
    Xp2 = [b[2] for b in batch]
    Y = [b[-1] for b in batch]
    
    Xp1 = np.vstack(Xp1)
    Xp2 = np.vstack(Xp2)
    Y = np.asarray(Y, np.int32)[:, None]

    Xp1 = to_device(device, Xp1)
    Xp2 = to_device(device, Xp2)
    Y = to_device(device, Y)

    return Xim, Xp1, Xp2, Y

def my_ip_converter_wordvec(batch, device=None):
    Xim = [b[0] for b in batch]
    Xp1 = [to_device(device, np.asarray(b[1], np.int32)) for b in batch]
    Xp2 = [to_device(device, np.asarray(b[2], np.int32)) for b in batch]

    Y = [b[-1] for b in batch]
    Y = np.asarray(Y, np.int32)[:, None]
    Y = to_device(device, Y)
    return Xim, Xp1, Xp2, Y

def my_p_converter_wordvec(batch, device=None):
    Xp1 = [to_device(device, np.asarray(b[0], np.int32)) for b in batch]
    Xp2 = [to_device(device, np.asarray(b[1], np.int32)) for b in batch]

    Y = [b[-1] for b in batch]
    Y = to_device(device, np.asarray(Y, np.int32)[:, None])
    return Xp1, Xp2, Y

def my_p_converter_pfeat(batch, device=None):
    Xp1 = [b[0] for b in batch]
    Xp2 = [b[1] for b in batch]
    Y = [b[2] for b in batch]
    
    Xp1 = np.vstack(Xp1)
    Xp2 = np.vstack(Xp2)
    Y = np.asarray(Y, np.int32)[:, None]

    Xp1 = to_device(device, Xp1)
    Xp2 = to_device(device, Xp2)
    Y = to_device(device, Y)

    return Xp1, Xp2, Y

class BinaryClassificationSummary(function.Function):
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, t = inputs

        y = y.ravel()
        true = t.ravel()
        pred = (y > 0)
        support = xp.sum(true)
        
        gtp_mask = xp.where(true)
        relevant = xp.sum(pred)
        tp = pred[gtp_mask].sum()
        
        if (support == 0) or (relevant == 0) or (tp == 0):
            return xp.array(0.), xp.array(0.), xp.array(0.)

        prec = tp * 1. / relevant
        recall = tp * 1. / support
        f1 = 2. * (prec * recall) / (prec + recall)
        
        return prec, recall, f1

def binary_classification_summary(y, t):
    return BinaryClassificationSummary()(y, t)

class PNetBase(chainer.Chain):
    def __init__(self):
        super(PNetBase, self).__init__()

        with self.init_scope():
            # classifier
            self.cls_w = L.Linear(None, 128, nobias=True, initialW=initializers.HeNormal())
            self.cls_u = L.Linear(None, 128, initialW=initializers.HeNormal())
            self.cls = L.Linear(None, 1, initialW=initializers.LeCunNormal())
        
    def classification_layer(self, h1, h2):
        h = F.relu(self.cls_w(h1) + self.cls_u(h2))
        return self.cls(h)
    
    def encode_phrase(self, X):
        raise NotImplementedError
    
    def predict(self, Xp1, Xp2):
        h1 = self.encode_phrase(Xp1)
        h2 = self.encode_phrase(Xp2)
        
        h = self.classification_layer(h1, h2)
        return F.sigmoid(h)

    def __call__(self, Xp1, Xp2, Y):
        h1 = self.encode_phrase(Xp1)
        h2 = self.encode_phrase(Xp2)
        
        h = self.classification_layer(h1, h2)
        loss = F.sigmoid_cross_entropy(h, Y)
        
        precision, recall, fbeta = binary_classification_summary(h, Y)
        chainer.report({'loss': loss, 'precision': precision, 'recall': recall, 'f1': fbeta}, self)
        
        return loss

class PNetFV(PNetBase):
    def __init__(self):
        super(PNetFV, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(None, 1000)

    def encode_phrase(self, X):
        return F.relu(self.fc(X))

class PNetAvr(PNetBase):
    def __init__(self, w_vec):
        super(PNetAvr, self).__init__()
        self.add_persistent('w_vec', w_vec)

    def encode_phrase(self, X):
        X = [F.mean(F.embed_id(x, self.w_vec), axis=0, keepdims=True) for x in X]
        X = F.vstack(X)
        return X

class IPNetBase(PNetBase):
    def __init__(self, pretrained_model):
        super(IPNetBase, self).__init__()

        if isinstance(pretrained_model, L.VGG16Layers):
            self.D = 14**2
            self.C = 512
            o_layer='conv5_3'
        elif isinstance(pretrained_model, L.ResNet50Layers):
            self.D = 7**2
            self.C = 2048
            o_layer='res5'

        self._ext_fun = lambda x: self.vis_cnn.extract(x, layers=[o_layer])[o_layer]

        with self.init_scope():
            self.vis_cnn = pretrained_model
            self.bn = L.BatchNormalization(self.C)

            # attention module
            self.u_att = L.Linear(None, self.C, nobias=True, initialW=initializers.HeNormal())
            self.w_att = L.Linear(None, self.C, initialW=initializers.HeNormal())
            self.w = L.Linear(None, 1, initialW=initializers.LeCunNormal())
            
            # fuse layer
            self.fuse1 = L.Linear(None, 128, nobias=True, initialW=initializers.HeNormal())
            self.fuse2 = L.Linear(None, 128, initialW=initializers.HeNormal())
            
    def _extract_preprocessed(self, images, layers):
        x = concat_examples(images)
        x = chainer.Variable(self.vis_cnn.xp.asarray(x))
        return self.vis_cnn(x, layers=layers)

    def project_features(self, features):
        features_flat = features.transpose((0, 2, 3, 1)).reshape((-1, self.C)) # (N*196, self.C)
        features_proj = self.u_att(features_flat)
        features_proj = features_proj.reshape((-1, self.D, self.C)) # (N, self.D, self.C)
        return features_proj
        
    def attention_layer(self, features, features_proj, Xp):
        h = F.expand_dims(self.w_att(Xp), 1)
        features_proj = F.normalize(features_proj, axis=-1)
        h = F.normalize(h, axis=-1)
        h_att = F.relu(features_proj + F.broadcast_to(h, features_proj.shape)) # (N, self.D, self.C) + (N, 1, self.C)
        out_att = self.w(F.reshape(h_att, (-1, self.C))) # (Nxself.D, self.C) -> (Nxself.D, 1)
        out_att = F.reshape(out_att, (-1, self.D)) # (N, self.D)
        alpha = F.softmax(out_att) # (N, self.D)
        context = F.sum(features * F.broadcast_to(F.expand_dims(alpha, 1), features.shape), axis=2) # (N, self.C, self.D) * (N, 1, self.D)
        return context, alpha
    
    def fuse_layer(self, context, phr_features):
        context = F.normalize(context, axis=1)
        phr_features = F.normalize(phr_features, axis=1)
        return F.relu(self.fuse1(context) + self.fuse2(phr_features))
    
    def encode_phrase(self, X):
        raise NotImplementedError
    
    def predict(self, Xim, Xp1, Xp2):
        Xp1 = self.encode_phrase(Xp1)
        Xp2 = self.encode_phrase(Xp2)
        
        # extract feature map from cnn
        with function.no_backprop_mode(), chainer.using_config('train', False):
            features = self._ext_fun(Xim)
        
        features = self.bn(features)
        features_proj = self.project_features(features)
        features = F.reshape(features, (-1, self.C, self.D))
        
        # get context
        context1, alpha1 = self.attention_layer(features, features_proj, Xp1)
        context2, alpha2 = self.attention_layer(features, features_proj, Xp2)
        
        h1 = self.fuse_layer(context1, Xp1)
        h2 = self.fuse_layer(context2, Xp2)
        
        h = self.classification_layer(h1, h2)
        return F.sigmoid(h)

    def __call__(self, Xim, Xp1, Xp2, Y):
        Xp1 = self.encode_phrase(Xp1)
        Xp2 = self.encode_phrase(Xp2)
        
        # extract feature map from cnn
        with function.no_backprop_mode(), chainer.using_config('train', False):
            features = self._ext_fun(Xim)
        
        features = self.bn(features)
        features_proj = self.project_features(features)
        features = F.reshape(features, (-1, self.C, self.D))
        
        # get context
        context1, alpha1 = self.attention_layer(features, features_proj, Xp1)
        context2, alpha2 = self.attention_layer(features, features_proj, Xp2)
        
        h1 = self.fuse_layer(context1, Xp1)
        h2 = self.fuse_layer(context2, Xp2)
        
        h = self.classification_layer(h1, h2)
        loss = F.sigmoid_cross_entropy(h, Y)
        
        precision, recall, fbeta = binary_classification_summary(h, Y)
        chainer.report({'loss': loss, 'precision': precision, 'recall': recall, 'f1': fbeta}, self)
        
        return loss

class IPNetFV(IPNetBase):
    def __init__(self, pretrained_model):
        super(IPNetFV, self).__init__(pretrained_model)
        with self.init_scope():
            self.fc = L.Linear(None, 1000)

    def encode_phrase(self, X):
        return F.relu(self.fc(X))

class IPNetAvr(IPNetBase):
    def __init__(self, pretrained_model, w_vec):
        super(IPNetAvr, self).__init__(pretrained_model)
        self.add_persistent('w_vec', w_vec)
    
    def encode_phrase(self, X):
        X = [F.mean(F.embed_id(x, self.w_vec), axis=0, keepdims=True) for x in X]
        return F.vstack(X)

def observe_pr(iterator_name='main', observation_key='pr'):
    return extensions.observe_value(
        observation_key,
        lambda trainer: trainer.updater._iterators['main'].p_batch_ratio)

def get_dataset(phrase_net, image_net=None, split='val', skip=None, preload=False, san_check=False):
    print(phrase_net, image_net, split)
    wo_image = (image_net is None)
    
    if not wo_image:
        if 'FlickrIMG_ROOT' not in os.environ:
            raise RuntimeError('set environmental variable FlickrIMG_ROOT')

        IMG_ROOT = os.environ['FlickrIMG_ROOT']
        img_root = IMG_ROOT

    
    phrase_file = 'data/phrase_pair_remove_trivial_match_%s.csv'
    word_dict_file = 'data/entity/word_dict'
    phrase_feature_file = 'data/entity/%s/textFeats_%s.npy'% (split, phrase_net)
    unique_phrase_file = 'data/entity/%s/uniquePhrases' % split

    if phrase_net in ['rnn', 'avr']:
        if wo_image:
            dataset = EntityDatasetWordID(phrase_file%split, word_dict_file,san_check=san_check, skip=skip)
            conv_f = my_p_converter_wordvec
        else:
            dataset = ImageEntityDatasetWordID(phrase_file%split,
                                    word_dict_file, img_root, san_check=san_check, preload=preload, skip=skip)
            conv_f = my_ip_converter_wordvec
            
    elif phrase_net in ['fv', 'fv+cca', 'fv+pca']:
        if wo_image:
            dataset = EntityDatasetPhraseFeat(phrase_file % split,phrase_feature_file, unique_phrase_file,san_check=san_check, skip=skip)
            conv_f = my_p_converter_pfeat
        else:
            dataset = ImageEntityDatasetPhraseFeat(phrase_file%split,
                                    phrase_feature_file,
                                    unique_phrase_file, img_root, san_check=san_check, preload=preload, skip=skip)
            conv_f = my_converter
    else:
        raise RuntimeError

    return dataset, conv_f

def setup_model(phrase_net, image_net):
    if image_net == 'vgg':
        vis_cnn = L.VGG16Layers()
    elif image_net == 'resnet':
        vis_cnn = L.ResNet50Layers()
    else:
        pass

    wo_image = (image_net is None)

    if phrase_net in ['rnn', 'avr']:
        w_vec = np.load('data/entity/word_vec.npy')
    
    if phrase_net in ['fv', 'fv+cca', 'fv+pca']:
        if image_net is None:
            model = PNetFV()
        elif image_net in ['vgg', 'resnet']:
            model = IPNetFV(vis_cnn)
        else:
            raise RuntimeError
    elif phrase_net == 'rnn':
        model = PNetGRU(w_vec) if wo_image else IPNetGRU(vis_cnn, w_vec)
    elif phrase_net == 'avr':
        model = PNetAvr(w_vec) if wo_image else IPNetAvr(vis_cnn, w_vec)
    else:
        raise NotImplementedError
        
    return model

def get_prediction(model_dir, split, device=None):
    model_dir = model_dir+'/' if model_dir[-1] != '/' else model_dir

    setting = json.load(open(model_dir+'settings.json'))
    image_net = setting['image_net']
    phrase_net = setting['phrase_net']
    # img_preprocessed = setting['img_preprocessed']

    model = setup_model(phrase_net, image_net)

    chainer.serializers.load_npz(model_dir+'model', model)
    if device is not None:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    wo_image = (image_net is None)
    test, conv_f = get_dataset(phrase_net, image_net=image_net, split=split, preload=True)
    test_iter = SerialIterator(test, batch_size=300, repeat=False, shuffle=False)
    
    s_i = 0
    e_i = 0
    pred = np.zeros((len(test),), dtype=np.float32)
    
    with function.no_backprop_mode(), chainer.using_config('train', False):
       
        for i, batch in enumerate(test_iter):
            inputs = conv_f(batch, device)
            score = model.predict(*inputs[:-1])
            score.to_cpu()

            e_i = s_i + len(batch)
            pred[s_i:e_i] = score.data.ravel()
            s_i = e_i

    df = pd.DataFrame({
        'image': test._image_id,
        'phrase1': test._phrase1,
        'phrase2': test._phrase2,
        'ytrue': test._label,
        'score': pred,
    })
    
    return df

def train(args):
    san_check = args.san_check
    epoch = args.epoch
    lr = args.lr
    b_size = args.b_size
    device = args.device
    w_decay = args.w_decay
    image_net = args.image_net
    phrase_net = args.phrase_net
    preload = args.preload
    wo_image = (image_net is None)

    out_base = 'checkpoint/'
    time_stamp = dt.now().strftime("%Y%m%d-%H%M%S")
    saveto = out_base + '{}{}-{}_{}/'.format(
        'sc_' * san_check,
        phrase_net,
        image_net,
        time_stamp)
    os.makedirs(saveto)
    json.dump(vars(args), open(saveto+'settings.json', 'w'))

    print('setup dataset...')
    train, conv_f = get_dataset(phrase_net, image_net=image_net, split='train', preload=preload, san_check=args.san_check)
    val, _ = get_dataset(phrase_net, image_net=image_net, split='val', skip=10*4, preload=preload, san_check=args.san_check)

    train_iter = SampleManager(train, b_size, p_batch_ratio=.15)
    val_iter = SerialIterator(val, b_size, shuffle=False, repeat=False)

    print('setup a model ...')
    chainer.cuda.get_device_from_id(device).use()
    model = setup_model(phrase_net, image_net)
    model.to_gpu()

    optimizer = chainer.optimizers.Adam(alpha=lr)
    optimizer.setup(model)

    if hasattr(model, 'vis_cnn'):
        model.vis_cnn.disable_update() # This line protects vgg paramters from weight decay.

    if w_decay:
        optimizer.add_hook(
            chainer.optimizer.WeightDecay(w_decay), 'hook_dec')

    updater = training.StandardUpdater(train_iter, optimizer,converter=conv_f, device=device)
    trainer = training.Trainer(updater, (epoch, 'epoch'), saveto)

    val_interval = (1, 'epoch') if san_check else (1000, 'iteration')
    log_interval = (1, 'iteration') if san_check else (10, 'iteration')
    plot_interval = (1, 'iteration') if san_check else (10, 'iteration')
    dataset_interval = (1, 'iteration') if san_check else (1000, 'iteration')

    trainer.extend(extensions.Evaluator(val_iter, model, converter=conv_f, device=device),
                trigger=val_interval)
    
    if not san_check:
        trainer.extend(extensions.ExponentialShift(
            'alpha', 0.5), trigger=(1, 'epoch'))
    
    # # Comment out to enable visualization of a computational graph.
    # trainer.extend(extensions.dump_graph('main/loss'))

    if not san_check:
        ## Comment out next line to save a checkpoint at each epoch, which enable you to restart training loop from the saved point. Note that saving a checkpoint may cost a few minutes.
        trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
        trainer.extend(extensions.snapshot_object(model, 'model_{.updater.iteration}'), trigger=val_interval)
        trainer.extend(extensions.snapshot_object(
            model, 'model'), trigger=training.triggers.MaxValueTrigger('validation/main/f1', trigger=val_interval))
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/f1', 'validation/main/f1', 'pr', 'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], file_name='loss.png', trigger=plot_interval))
    trainer.extend(extensions.PlotReport(['main/f1', 'validation/main/f1'], file_name='f1.png', trigger=plot_interval))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    print('start training')
    trainer.run()

    chainer.serializers.save_npz(saveto+'final_model', model)

def evaluate(model_dir, split, device=None):
    df = get_prediction(model_dir, split, device)
    print('writing predicted results to '+model_dir+'res_%s.csv'%split)
    df.to_csv(model_dir+'res_%s.csv'%split)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='training script for a paraphrase classifier')
    parser.add_argument('--lr', '-lr', type=float, default=0.01,
                        help='learning rate <float> (default 0.01)')
    parser.add_argument('--device', '-d', type=int, default=None,
                        help='gpu device id <int> (default None(cpu mode))')
    parser.add_argument('--b_size', '-b', type=int, default=100,
                        help='minibatch size <int> (default 100)')
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='maximum epoch <int> (default 5)')
    parser.add_argument('--san_check', '-sc', action='store_true',
                        help='sanity check mode')
    parser.add_argument('--w_decay', '-wd', type=float, default=0.0001,
                        help='weight decay <float> (default 0.0001)')
    parser.add_argument('--preload', action='store_true',
                        help='load images beforehand.')
    parser.add_argument('--resume', type=str, default=None,
                        help='file name of a snapshot <str>. you can restart the training at the checkpoint.')
    parser.add_argument('--phrase_net', type=str, default='avr',
                        help='phrase features <str>: fv+cca, fv+pca, avr')
    parser.add_argument('--image_net', type=str, default=None,
                        help='network to encode images <str>: vgg, resnet, if none only phrases are used.')
    parser.add_argument('--eval', type=str, default=None,
                        help='path to a directory of a model, which will be evaluated <str>')
    args = parser.parse_args()

    if args.eval is not None:
        # evaluate(args.eval, split='val', device=args.device)
        evaluate(args.eval, split='test', device=args.device)
    else:
        train(args)

if __name__ == '__main__':
    main()