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

def vis_detections(im, class_name, dets, thresh=0.5):
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
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

class Detections:
    def __init__(self, image):
        image = image[:, :, (2, 1, 0)]
        self.image = image
        self.ious = []
        self.obj_scores = []

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
                semantic_data, iou, obj_score = semantic_segment_image(heat_map_obj, patch, color_label[class_name])
                # if obj_score > .1 and iou > .2:
                if obj_score > .2:
                    self.image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = semantic_data
                    self.ious.append(iou)
                    self.obj_scores.append(obj_score)

            if show_detection_info:
                bgr_img = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(bgr_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), get_rgb_from_color(color_label[class_name])[::-1], 2)
                self.image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    def get_image(self):
        return self.image

    def save_image(self, path):
        cv2.imwrite(path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))

    def show_image(self):
        frame = plt.gca()
        frame.axes.get_xaxis().set_ticks([])
        frame.axes.get_yaxis().set_ticks([])
        plt.imshow(self.image)
        plt.show()

    def get_mean_iou(self):
        print '>>>'
        print np.min(self.obj_scores)
        print np.mean(self.obj_scores)
        print '<<<'
        return np.mean(self.ious)


def get_detections(heat_map_obj, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'sdd', 'JPEGImages', image_name)
    # im_file = os.path.join('/home/joseph/Dataset/iith_drone_data/convocation', image_name)
    im_file = os.path.join('/home/joseph/Dataset/iith_drone_data/orijinal_resolution', image_name)

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
    timer = Timer()
    timer.tic()
    detection_object = Detections(im)
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        detections = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(detections, nms_threshold)
        detections = detections[keep, :]
        if cls == 'pedestrian':
            detection_object.plot(heat_map_obj, cls, detections, image_name, thresh=conf_threshold, show_detection_info=False, show_semantic_info=True)
        # vis_detections(im, cls, detections)
    timer.toc()
    # print 'Instance Segmentation took {:.3f}s '.format(timer.total_time)
    # detection_object.show_image()
    # print 'Mean IoU:', detection_object.get_mean_iou()
    # detection_object.show_image()
    detection_object.save_image('/home/joseph/drdo/ee_drone/semantic_det_'+image_name+'.png')


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

    # prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0], 'rfcn_end2end', 'test_agnostic.prototxt')
    # caffemodel = os.path.join(cfg.DATA_DIR, 'rfcn_models', NETS[args.demo_net][1])

    prototxt = os.path.join(cfg.MODELS_DIR, 'VGG16',
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join('/home/joseph/workspace/SDD-RFCN-python'
                              '/output/faster_rcnn_end2end_sdd/sdd_trainval',
                              'Enhanced_annotation_all_from_one_vgg16_faster_rcnn_sdd_iter_70000.caffemodel')

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

    # im_names = ['bookstore_video0_9500.jpg']
    # im_names = ['nexus_video1_9982.jpg']
    # im_names = ['little_video1_1750.jpg']
    # im_names = ['iith_convo_883.jpg']
    # im_names = ['iith_06_100.jpg']
    im_names = ['iith_06_' + str(i) + '.jpg' for i in range(856, 1200)]



    for im_name in im_names:
        print 'Getting detections of {}'.format(im_name)
        get_detections(hm, net, im_name)

    # from_frameno = 9000
    # to_frameno = 10000
    # for i in range(from_frameno, to_frameno):
    #     img_name = 'bookstore_video0_' + str(i) + '.jpg'
    #     print 'Getting detections of ', img_name
    #     get_detections(hm, net, img_name)
    plt.show()
    print 'Done.'
