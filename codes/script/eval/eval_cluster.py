import pandas as pd
import progressbar
from sklearn import metrics
import sys
import codecs

if sys.version_info < (3, 0):
    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
    sys.stdin = codecs.getreader('UTF-8')(sys.stdin)

def eval_ari_score(gold, predict_df):
    gt_df = pd.read_csv(gold, encoding = "utf-8")

    img_list = gt_df.image.unique()
    bar = progressbar.ProgressBar()

    score = 0
    for img in bar(img_list):
        gt_df_img = gt_df.query('image == %i' % img)
        predict_df_img = predict_df.query('image == %i' % img)

        gt_label_dic = {}
        id_category_dic = {}
        gt_label = 0
        for _, item in gt_df_img.iterrows():
            phrase = item.phrase
            id_category = item.label
            if id_category in id_category_dic:
                gt_label_dic[phrase] = id_category_dic[id_category]
            else:
                gt_label += 1
                id_category_dic[id_category] = gt_label
                gt_label_dic[phrase] = gt_label

        predict_label_list = []
        gt_label_list = []
        for _, item in predict_df_img.iterrows():
            phrase = item.phrase
            label = item.label

            if phrase in gt_label_dic:
                predict_label_list.append(label)
                gt_label_list.append(gt_label_dic[phrase])
        score += metrics.adjusted_rand_score(gt_label_list, predict_label_list)

    return score / len(img_list)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('paraphrase detection')
    parser.add_argument('--gold', '-g', type=str, default=None, help='ground truth file')
    parser.add_argument('--predict', '-p', type=str, default=None, help='predict file')
    args = parser.parse_args()

    score = eval(gold=args.gold, predict=args.predict)
    print("Adjusted Rand index: %.4f" % score)