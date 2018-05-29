# IParaphrasing
This is training and evaluation scripts of our work, **iParaphrasing: Extracting Visually Grounded Paraphrases via an Image**.
# Dependencies
This code was tested on the environment below:
- python 2.7.13
- chainer 3.0.0
- pandas 0.20.3
- imageio 1.5.0
- scikit-learn 0.19.1

# How to train the model
First, set `IMG_ROOT` in the `script/training/train_paraphrase_classifier.py` to the directory where Flickr30K images are downloaded.
To train a model SNN+image (FV+CCA) in our paper, run
```sh
$ cd iparaphrase
$ python script/training/train_paraphrase_classifier.py -di 0 --phrase_net fv+cca --image_net vgg
```

```
$ python script/training/train_paraphrase_classifier.py -help
usage: train_paraphrase_classifier.py [-h] [--lr LR] [--device DEVICE]
                                      [--b_size B_SIZE] [--epoch EPOCH]
                                      [--san_check] [--w_decay W_DECAY]
                                      [--preload] [--resume RESUME]
                                      [--phrase_net PHRASE_NET]
                                      [--image_net IMAGE_NET] [--eval EVAL]

training script for a paraphrase classifier

optional arguments:
  -h, --help            show this help message and exit
  --lr LR, -lr LR       learning rate <float>
  --device DEVICE, -d DEVICE
                        gpu device id <int>
  --b_size B_SIZE, -b B_SIZE
                        minibatch size <int> (default 10)
  --epoch EPOCH, -e EPOCH
                        maximum epoch <int>
  --san_check, -sc      sanity check mode
  --w_decay W_DECAY, -wd W_DECAY
                        weight decay <float>
  --preload             load images beforehand.
  --resume RESUME       file name of a snapshot <str>. you can restart the
                        training at the checkpoint.
  --phrase_net PHRASE_NET
                        phrase features <str>: fv+cca, avr
  --image_net IMAGE_NET
                        network to encode images <str>: vgg, resnet, if none
                        only phrases are used.
  --eval EVAL           path to an output directory <str>. the model will be
                        evaluated.


```

# Dataset
The dataset **Flickr30K Entities** is in this [link](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/).