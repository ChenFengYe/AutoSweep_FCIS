# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Haochen Zhang, Yi Li, Haozhi Qi
# --------------------------------------------------------

#
import _init_paths

import argparse
import os
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np

# get config

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))

test_dir = './demo/onekeyGeonet/test/'
# label_dir = './data/ObjectSnap/Refine/seg/'
# result_dir = './demo/2017.8.6_geonet_training_data_color/'
# result_dir = './demo/result/'
result_dir = './demo/onekeyGeonet/result/'
#update_config(cur_path + '/../experiments/fcis/cfgs/resnet_v1_101_ObjectSnap_synthetic_all_fcis_end2end_ohem.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx

print "use mxnet at", mx.__file__
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_masks import show_masks
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper
from mask.mask_transform import gpu_mask_voting, cpu_mask_voting
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Train FCIS Network')
    # general
    # configuration file is required
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(cur_path + '/../experiments/fcis/cfgs/' +args.cfg)

    args = parser.parse_args()
    return args

def checkLabel(label):
    label_flag = [];
    for r in range(label.shape[0]):
        for c in range(label.shape[1]):
            if label[r, c, 1] == 0 and label[r, c, 2] == 255:  # cubo
                label_flag.append(1)
            elif label[r, c, 1] != 0 and label[r, c, 2] == 255:  # cufa
                label_flag.append(2)
            elif label[r, c, 1] != 0 and label[r, c, 2] == 200:  # cybo
                label_flag.append(3)
            elif label[r, c, 1] != 0 and label[r, c, 2] == 200:  # cyfa
                label_flag.append(4)
            elif label[r, c, 1] != 0 and label[r, c, 2] == 150:  # grip
                label_flag.append(5)
    return np.unique(label_flag)

def main():
    args = parse_args()
    print(config.config_name)
    # get symbol
    ctx_id = [int(i) for i in config.gpus.split(',')]
    #pprint.pprint(config)

    sym_instance = eval(config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    # set up class names
    num_classes = config.dataset.NUM_CLASSES
    if num_classes == 6:
        classes = ['__background__',  # always index 0
                   '1', '2', '3', '4', '5']
    elif num_classes == 5:
        classes = ['__background__',  # always index 0
                   '1', '2', '3', '4']
    elif num_classes == 3:
        classes = ['__background__',  # always index 0
                   '1', '2']

    # classes = ['__background__',  # always index 0
    #            'CubeBody', 'CubeFace', 'CylinderBody', 'CylinderFace', 'Grip']

    # load demo data
    image_names = []
    names_dirs = os.listdir(cur_path + '/../' + test_dir)
    for im_name in names_dirs:
        if im_name[-4:] == '.jpg':
            image_names.append(im_name)
    print('Number of Images: %d \n'%(len(image_names)))

    data = []

    # next [3001: len(image_names)]
    for im_name in image_names[0:len(image_names)]:
        assert os.path.exists(cur_path + '/../' + test_dir + im_name), (
            '%s does not exist'.format('../demo/' + im_name))
        im = cv2.imread(cur_path + '/../' + test_dir + im_name, cv2.IMREAD_COLOR | long(128))
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        # print "before scale: "
        # print im.shape
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        # print "after scale: "
        # print im.shape
        # im_scale = 1.0
        # print "scale ratio: "
        # print im_scale
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        # print im_tensor.shape
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})

    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    model_path = cur_path + '/../' + config.output_path + '/' + config.config_name + '/' + config.dataset.image_set + '/' + config.TRAIN.model_prefix
    arg_params, aux_params = load_param(model_path, config.TEST.test_epoch, process=True)
    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(ctx_id[0])], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # warm up
    for i in xrange(2):
        data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
        _, _, _, _ = im_detect(predictor, data_batch, data_names, scales, config)

    # 12342_mask Log
    # LogTxt = open('LogRecorder.txt', 'w')

    # test
    for idx, im_name in enumerate(image_names[0:len(image_names)]):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        tic()
        scores, boxes, masks, data_dict = im_detect(predictor, data_batch, data_names, [1.0], config)
        im_shapes = [data_batch.data[i][0].shape[2:4] for i in xrange(len(data_batch.data))]
        # print im_shapes

        if not config.TEST.USE_MASK_MERGE:
            all_boxes = [[] for _ in xrange(num_classes)]
            all_masks = [[] for _ in xrange(num_classes)]
            nms = py_nms_wrapper(config.TEST.NMS)
            for j in range(1, num_classes):
                indexes = np.where(scores[0][:, j] > 0.7)[0]
                cls_scores = scores[0][indexes, j, np.newaxis]
                cls_masks = masks[0][indexes, 1, :, :]
                try:
                    if config.CLASS_AGNOSTIC:
                        cls_boxes = boxes[0][indexes, :]
                    else:
                        raise Exception()
                except:
                    cls_boxes = boxes[0][indexes, j * 4:(j + 1) * 4]

                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                all_boxes[j] = cls_dets[keep, :]
                all_masks[j] = cls_masks[keep, :]
            dets = [all_boxes[j] for j in range(1, num_classes)]
            masks = [all_masks[j] for j in range(1, num_classes)]
        else:
            masks = masks[0][:, 1:, :, :]
            result_masks, result_dets = gpu_mask_voting(masks, boxes[0], scores[0], num_classes,
                                                        100, im_shapes[0][1], im_shapes[0][0],
                                                        config.TEST.NMS, config.TEST.MASK_MERGE_THRESH,
                                                        config.BINARY_THRESH, ctx_id[0])

            dets = [result_dets[j] for j in range(1, num_classes)]
            masks = [result_masks[j][:, 0, :, :] for j in range(1, num_classes)]
        print 'testing {} {:.4f}s'.format(im_name, toc())
        # visualize
        for i in xrange(len(dets)):
            keep = np.where(dets[i][:, -1] > 0.7)
            dets[i] = dets[i][keep]
            masks[i] = masks[i][keep]
        im = cv2.imread(cur_path + '/../' + test_dir + im_name)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = show_masks(im, dets, masks, classes, config, 1.0 / scales[0], False)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        savePath = cur_path + '/../'+ result_dir + im_name[0:-4]+'.png';
        cv2.imwrite(savePath, im)

    print 'done'


if __name__ == '__main__':
    main()

