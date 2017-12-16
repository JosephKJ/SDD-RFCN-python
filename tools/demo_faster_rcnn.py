#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES = ('__background__',  # always index 0
                         'pedestrian', 'biker', 'skater', 'car',
                         'bus', 'cart')

NETS = {'ResNet-101': ('ResNet-101',
                  'resnet101_rfcn_final.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel'),
        'SDD': ('ResNet-101',
                  'resnet101_rfcn_ohem_iter_110000.caffemodel')}


def vis_detections(im, class_name, dets,image_name, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig('./img_output/ddo_80_ee/out_' + class_name + '_' + image_name, bbox_inches='tight')

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join('/home/joseph/Dataset/iith_drone_data/convocation', image_name)
    im_file = os.path.join('/home/joseph/Dataset/iith_drone_data/orijinal_resolution', image_name)
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        if cls == 'pedestrian':
            vis_detections(im, cls, dets, image_name, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        choices=NETS.keys(), default='SDD')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, 'VGG16',
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join('/home/joseph/workspace/SDD-RFCN-python'
                              '/output/faster_rcnn_end2end_sdd/sdd_trainval','Enhanced_annotation_all_from_one_vgg16_faster_rcnn_sdd_iter_70000.caffemodel')
                              # '/output/faster_rcnn_end2end_sdd/sdd_trainval','rpn_base_size_8_all_from_one_vgg16_faster_rcnn_sdd_iter_70000.caffemodel')
    # caffemodel = os.path.join('/home/joseph/workspace/SDD-RFCN-python'
    #                               '/output/faster_rcnn_end2end_sdd/sdd_trainval','vgg16_faster_rcnn_sdd_iter_30000.caffemodel')
                                  # '/output/faster_rcnn_end2end_sdd/sdd_trainval','all_from_one_vgg16_faster_rcnn_sdd_iter_70000.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    # im_names = ['iith_convo_883.jpg']
    # im_names = ['iith_06_100.jpg', 'iith_06_250.jpg', 'iith_06_500.jpg', 'iith_06_750.jpg', 'iith_06_1000.jpg',
    # 'iith_06_1250.jpg', 'iith_06_1500.jpg', 'iith_06_1750.jpg', 'iith_06_2000.jpg', 'iith_06_2500.jpg']
    # im_names = ['bookstore_video0_12345.jpg']
    # im_names = ['iith_convo_' + str(i) + '.jpg' for i in range(1, 884)]
    im_names = ['iith_06_' + str(i) + '.jpg' for i in range(1, 1200)]

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    for im_name in im_names:
        print 'Demo for {}'.format(im_name)
        demo(net, im_name)
    print 'With enhanced dataset.'

    # plt.show()
