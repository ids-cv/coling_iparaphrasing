# iParaphrasing
scripts for our COLING paper "iParaphrasing: Extracting Visually Grounded Paraphrases via an Image"

## Dependencies
This scripts are tested on

- python 3.6
- numpy 1.13.3
- scipy 1.0.0
- scikit-learn 0.19.1
- pandas 0.21.0
- chainer 4.0.0
- GPy 1.9.2
- GPuOpt 1.2.5

## Data

Get Flickr30K entities dataset [here](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/).

Other materials can be downloaded from [here](https://figshare.com/projects/_COLING18_iParaphrasing/34637) (figshare).

Download data.zip, ari_data.zip and models.zip, then extract zip files under `coling_iparaphrasing`.

## Train a model

Run the command below in `coling_iparaphrasing` directory．

```
FlickrIMG_ROOT=/path/to/flickr30k-images/ python codes/script/training/train_paraphrase_classifier.py -d 0 --image_net vgg --phrase_net fv+cca
```

By default, the output model and log files will be stored under `checkpoint/generated_name/`

For more details, run

```
python codes/script/training/train_paraphrase_classifier.py --help
```

## Get prediction
```
FlickrIMG_ROOT=/path/to/flickr30k-images/ python codes/script/training/train_paraphrase_classifier.py -d 0 --eval /path/to/output/directory/
```

Prediction results will be written to `res_test.csv` in the model directory.

## Evaluate prediction
First, run 
```
codes/script/shell/prepare_eval.sh
```
See codes/notebook/\[COLING\] Table 1.ipynb and codes/notebook/\[COLING\] Table 1-ARI scores.ipynb