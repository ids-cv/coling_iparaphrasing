import os
import xml.etree.ElementTree as ET
import re
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import gzip
import pandas as pd

def add_bbox(bb, ax, h, w, caption=None):
    y_min, x_min, y_max, x_max = bb
    
    xy = (x_min, y_min)
    width = x_max - x_min
    height = y_max - y_min
    ax.add_patch(plt.Rectangle(
        xy, width, height, fill=False, edgecolor='yellow', linewidth=2))

    if caption is None:
        return

    ax.text(bb[0], bb[1],
            caption,
            style='italic',
            bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 2})


def load_bbox(xml_f):
    tree = ET.parse(xml_f)
    root = tree.getroot()
    bbox_name = []
    for child in root.findall('object'):
        names = [n.text for n in child.findall('name')]
        bnbbox = child.find('bndbox')
        if bnbbox is not None:
            x_min, y_min, x_max, y_max = [int(item.text) for item in bnbbox]
            bb = y_min, x_min, y_max, x_max
            bbox_name.append((bb, names))

    return bbox_name


def load_entity(txt_f):
    phrase = {}

    with open(txt_f) as f:
        for line in f:
            start_i = [i for i, c in enumerate(line) if c == '[']
            end_i = [i for i, c in enumerate(line) if c == ']']

            for si, ei in zip(start_i, end_i):
                s = line[si + 1:ei]
                m = re.search(r'(?<=/EN#)[0-9]+', s)
                r_id = m.group(0)
                phr = s[s.find(' ') + 1:]

                phrase.setdefault(r_id, []).append(phr)
    return phrase


def load_region(img, bbox):
    y_min, x_min, y_max, x_max = bbox

    if isinstance(img, np.ndarray):
        return img[y_min:y_max, x_min:x_max]

    img = plt.imread(img)
    return img[y_min:y_max, x_min:x_max]

def load_pt(pt):
    pt_mul_all_dic = {}
    pt_prob_str_dic = {}
    # print('pt:', pt)
    f = gzip.open(pt, 'r')
    for line in f.readlines():
        ele_list = line.split(' ||| ')
        score_list = ele_list[2].split()
        pt_prob_str_dic.setdefault(ele_list[0], {})[ele_list[1]] = ele_list[2]
        score_mul = float(score_list[0]) * float(score_list[2])
        # score_mul = 1
        # for score in score_list:
        #     score_mul *= float(score)
        pt_mul_all_dic.setdefault(ele_list[0], {})[ele_list[1]] = score_mul
        # print(pt_dic[ele_list[0]][ele_list[1]])
    f.close()
    return pt_mul_all_dic, pt_prob_str_dic

def judge_entity_pair(entity_1, entity_2, pt_dic):
    flag = False
    if(entity_1 in pt_dic):
        if(entity_2 in pt_dic[entity_1]):
            # print(entity_1, '#', entity_2, '#', pt_dic[entity_1][entity_2])
            flag = True
    # if(flag == False):
    #     print(entity_1, '#', entity_2, '#')
    return flag

def load_entity_per_sen(txt_f):
    entity_dic = {}

    line_num = 0
    with open(txt_f) as f:
        for line in f:
            start_i = [i for i, c in enumerate(line) if c == '[']
            end_i = [i for i, c in enumerate(line) if c == ']']

            for si, ei in zip(start_i, end_i):
                s = line[si + 1:ei]
                # m = re.search(r'(?<=/EN#)[0-9]+', s)
                m = re.search(r'(?<=/EN#)[\S]+', s)
                r_id = m.group(0)
                entity = s[s.find(' ') + 1:]
                # entity_dic[line_num][r_id] = entity.lower()
                if(line_num in entity_dic):
                    if (r_id in entity_dic[line_num]):
                        entity_ = entity_dic[line_num][r_id]
                        entity_[entity.lower()] = 1
                        entity_dic[line_num][r_id] = entity_
                    else:
                        entity_ = {}
                        entity_[entity.lower()] = 1
                        entity_dic.setdefault(line_num, {})[r_id] = entity_
                else:
                    entity_ = {}
                    entity_[entity.lower()] = 1
                    entity_dic.setdefault(line_num, {})[r_id] = entity_
            line_num += 1
    return entity_dic

def load_convert(csv):
    data = pd.read_csv(csv)
    convert_list = {}
    for _, item in data.iterrows():
        phr1_pair = item.phrase1.split('/')
        phr2_pair = item.phrase2.split('/')
        image_id = item.image
        convert_list.setdefault(str(image_id), {})[phr1_pair[0].lower()] = phr1_pair[1]
        convert_list.setdefault(str(image_id), {})[phr2_pair[0].lower()] = phr2_pair[1]
    return convert_list

def show_samples(flickr30k_entities_dir, flickr30k_images_dir):
    # annotation file
    xml_files = os.listdir(flickr30k_entities_dir + '/Annotations/')
    xml_f = random.choice(xml_files)
    # xml_f = '4567003374.xml'
    print(xml_f)

    # load bbox and entities
    bbox_name = load_bbox(flickr30k_entities_dir + '/Annotations/' + xml_f)
    phrase = load_entity(flickr30k_entities_dir +
                         '/Sentences/' + xml_f[:-4] + '.txt')

    # display image with bbox2entities
    im = plt.imread(flickr30k_images_dir + '/' + xml_f[:-4] + '.jpg')
    plt.imshow(im)
    bb_id = 0
    for bb, name in bbox_name:
        add_bbox(bb, plt.gca(), str(bb_id))

        for n in name:
            print('BBoxID %i:' % bb_id, ', '.join(phrase[n]))

        bb_id += 1

    plt.axis('off')
    plt.show()


def cmdline(arguments=None):
    parser = argparse.ArgumentParser(description="Show Flickr30k entities examples.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--flickr30k_entities_dir",
                        help="set Flickr30k entities directory")
    parser.add_argument("--flickr30k_images_dir",
                        help="set Flickr30k images directory")
    args = parser.parse_args(args=arguments)
    for _ in range(1):
        show_samples(args.flickr30k_entities_dir, args.flickr30k_images_dir)
    plt.close()

if __name__ == '__main__':
    cmdline()
