#!/usr/bin/env python

"""
Create a semantic segmentation of an image.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from objectness.utils import semantic_segment_image
from objectness.utils import get_rgb_from_color
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from objectness.map import HeatMap


CLASSES = ('__background__','pedestrian', 'biker', 'skater', 'car', 'bus', 'cart')

NETS = {'SDD': ('ResNet-101','resnet101_rfcn_ohem_iter_110000.caffemodel')}


class Detections:
    def __init__(self, image):
        image = image[:, :, (2, 1, 0)]
        self.image = image

    def plot(self, heat_map_obj, class_name, dets, image_name, thresh=0.5, show_semantic_info=True, show_detection_info=True):
        """
        Plot the semantic segmentation information and the detections
        :param heat_map_obj:
        :param class_name:
        :param dets:
        :param image_name:
        :param thresh:
        :param show_semantic_info:
        :return:
        """

        color_label = {'pedestrian': 'red', 'biker': 'green', 'skater': 'blue',
                       'car': 'black', 'bus': 'white', 'cart': 'violet'}

        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            if show_semantic_info:
                patch = self.image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                semantic_data, iou = semantic_segment_image(heat_map_obj, patch, color_label[class_name])
                self.image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = semantic_data
                print iou

            if show_detection_info:
                bgr_img = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(bgr_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), get_rgb_from_color(color_label[class_name])[::-1], 2)
                self.image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    def get_image(self):
        return self.image

    def save_image(self, path):
        cv2.imwrite(path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))

    def show_image(self):
        plt.imshow(self.image)
        plt.show()


def get_detections(heat_map_obj, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'sdd', 'JPEGImages', image_name)
    im = cv2.imread(im_file)
    image_name = image_name.split('.')[0]

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    conf_threshold = 0.8
    nms_threshold = 0.3
    detection_object = Detections(im)
    for cls_ind, cls in enumerate(CLASSES[1:]):
        print 'Plotting ', cls
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        detections = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(detections, nms_threshold)
        detections = detections[keep, :]
        detection_object.plot(heat_map_obj, cls, detections, image_name, thresh=conf_threshold, show_detection_info=False)

    detection_object.show_image()
    # detection_object.save_image('/home/cs17mtech01001/workspace/SDD-RFCN-python/data/full_images/semantic_det_'+image_name+'.png')


def parse_args():
    """
    Parse input arguments.
    :return: args
    """
    parser = argparse.ArgumentParser(description='Segment images from SDD.')
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

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0], 'rfcn_end2end', 'test_agnostic.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'rfcn_models', NETS[args.demo_net][1])

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

    hm = HeatMap()
    print '\n\nLoaded VGG, for segmentation.'

    # Warmup on a dummy image
    img = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, img)

    im_names = ['bookstore_video0_9500.jpg']
    for im_name in im_names:
        print 'Getting detections of {}'.format(im_name)
        get_detections(hm, net, im_name)

    # from_frameno = 9000
    # to_frameno = 10000
    # for i in range(from_frameno, to_frameno):
    #     img_name = 'bookstore_video0_' + str(i) + '.jpg'
    #     print 'Getting detections of ', img_name
    #     get_detections(hm, net, img_name)

    print 'Done.'
