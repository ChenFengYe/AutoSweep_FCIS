# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Haochen Zhang
# --------------------------------------------------------

import numpy as np
import utils.image as image
import matplotlib.pyplot as plt
import random
import cv2


def show_masks(im, detections, masks, class_names, cfg, scale=1.0, show = True):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    # plt.cla()
    # plt.axis("off")
    # plt.imshow(im)
    im = np.zeros(im.shape,dtype=np.uint8)

    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        # cube no cylinder label
        # elif im_name[0:4] == 'cube' and (j == 3 or j == 4):
        #     continue
        # # cylinder no cube label
        # elif im_name[0:8] == 'cylinder' and (j ==1 or j == 2):
        #     continue

        #liyuwei
        k = j-1

        dets = detections[k]
        msks = masks[k]
        i_instance = 0
        for det, msk in zip(dets, msks):
            # color = (random.random(), random.random(), random.random()) * 256  # generate a random color
            i_instance = i_instance + 1
            color = get_instance_color(name, i_instance)
            bbox = det[:4] * scale
            cod = bbox.astype(int)
            if im[cod[1]:cod[3], cod[0]:cod[2], 0].size > 0:
                msk = cv2.resize(msk, im[cod[1]:cod[3]+1, cod[0]:cod[2]+1, 0].T.shape)
                bimsk = msk >= cfg.BINARY_THRESH
                bimsk = bimsk.astype(int)
                bimsk = np.repeat(bimsk[:, :, np.newaxis], 3, axis=2)
                clmsk = np.ones(bimsk.shape) * bimsk
                clmsk[:, :, 0] = clmsk[:, :, 0] * color[0]
                clmsk[:, :, 1] = clmsk[:, :, 1] * color[1]
                clmsk[:, :, 2] = clmsk[:, :, 2] * color[2]
                for r in range(cod[1], cod[3]+1):
                    for c in range(cod[0], cod[2]+1):
                        if np.any(clmsk[r-cod[1], c-cod[0],:] > [0, 0, 0]):
                            im[r, c, :] = clmsk[r-cod[1], c-cod[0], :]
                            # im[cod[1]:cod[3] + 1, cod[0]:cod[2] + 1, :] = im[cod[1]:cod[3] + 1, cod[0]:cod[2] + 1, :] + clmsk
    if show:
        plt.imshow(im)
        plt.show()
    return im

def get_instance_color(class_name, i_instance):
    if class_name == '1':
        color = (i_instance*10, 0, 255)
    elif class_name == '2':
        color = (i_instance*10, i_instance*10, 255)
    elif class_name == '3':
        color = (i_instance*10, 0, 200)
    elif class_name == '4':
        color = (i_instance*10, i_instance*10, 200)
    elif class_name == '5':
        color = (i_instance * 10, 0, 150)

    return color