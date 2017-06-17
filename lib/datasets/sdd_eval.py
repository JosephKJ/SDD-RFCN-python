# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np

def parse_sdd_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_dict = dict()
        obj_dict['name'] = obj.find('name').text
        obj_dict['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [int(float(bbox.find('xmin').text)),
                            int(float(bbox.find('ymin').text)),
                            int(float(bbox.find('xmax').text)),
                            int(float(bbox.find('ymax').text))]
        objects.append(obj_dict)
    return objects

def sdd_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
#
# def sdd_eval(detpath,
#              annopath,
#              imagesetfile,
#              classname,
#              cachedir,
#              ovthresh=0.5,
#              use_07_metric=False):
#     """rec, prec, ap = voc_eval(detpath,
#                                 annopath,
#                                 imagesetfile,
#                                 classname,
#                                 [ovthresh],
#                                 [use_07_metric])
#
#     Top level function that does the PASCAL VOC evaluation.
#
#     detpath: Path to detections
#         detpath.format(classname) should produce the detection results file.
#     annopath: Path to annotations
#         annopath.format(imagename) should be the xml annotations file.
#     imagesetfile: Text file containing the list of images, one image per line.
#     classname: Category name (duh)
#     cachedir: Directory for caching the annotations
#     [ovthresh]: Overlap threshold (default = 0.5)
#     [use_07_metric]: Whether to use VOC07's 11 point AP computation
#         (default False)
#     """
#     # assumes detections are in detpath.format(classname)
#     # assumes annotations are in annopath.format(imagename)
#     # assumes imagesetfile is a text file with each line an image name
#     # cachedir caches the annotations in a pickle file
#
#     print '*******(S)*******'
#     print detpath
#     print annopath
#     print imagesetfile
#     print classname
#     print cachedir
#     print '*******(E)*******'
#
#     # first load gt
#     if not os.path.isdir(cachedir):
#         os.mkdir(cachedir)
#     cachefile = os.path.join(cachedir, 'annots.pkl')
#     # read list of images
#     with open(imagesetfile, 'r') as f:
#         lines = f.readlines()
#     imagenames = [x.strip() for x in lines]
#
#     if not os.path.isfile(cachefile):
#         # load annots
#         recs = {}
#         for i, imagename in enumerate(imagenames):
#             recs[imagename] = parse_rec(annopath.format(imagename))
#             if i % 100 == 0:
#                 print 'Reading annotation for {:d}/{:d}'.format(
#                     i + 1, len(imagenames))
#         # save
#         print 'Saving cached annotations to {:s}'.format(cachefile)
#         with open(cachefile, 'w') as f:
#             cPickle.dump(recs, f)
#     else:
#         # load
#         with open(cachefile, 'r') as f:
#             recs = cPickle.load(f)
#
#     # extract gt objects for this class
#     class_recs = {}
#     npos = 1
#     for imagename in imagenames:
#         R = [obj for obj in recs[imagename] if obj['name'] == classname]
#         bbox = np.array([x['bbox'] for x in R])
#         difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
#         det = [False] * len(R)
#         npos = npos + sum(~difficult)
#         class_recs[imagename] = {'bbox': bbox,
#                                  'difficult': difficult,
#                                  'det': det}
#
#     # read dets
#     detfile = detpath.format(classname)
#     with open(detfile, 'r') as f:
#         lines = f.readlines()
#
#     splitlines = [x.strip().split(' ') for x in lines]
#     image_ids = [x[0] for x in splitlines]
#     confidence = np.array([float(x[1]) for x in splitlines])
#     BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
#
#     # sort by confidence
#     sorted_ind = np.argsort(-confidence)
#     sorted_scores = np.sort(-confidence)
#     BB = BB[sorted_ind, :]
#     image_ids = [image_ids[x] for x in sorted_ind]
#
#     # go down dets and mark TPs and FPs
#     nd = len(image_ids)
#     tp = np.zeros(nd)
#     fp = np.zeros(nd)
#     for d in range(nd):
#         R = class_recs[image_ids[d]]
#         bb = BB[d, :].astype(float)
#         ovmax = -np.inf
#         BBGT = R['bbox'].astype(float)
#
#         if BBGT.size > 0:
#             # compute overlaps
#             # intersection
#             ixmin = np.maximum(BBGT[:, 0], bb[0])
#             iymin = np.maximum(BBGT[:, 1], bb[1])
#             ixmax = np.minimum(BBGT[:, 2], bb[2])
#             iymax = np.minimum(BBGT[:, 3], bb[3])
#             iw = np.maximum(ixmax - ixmin + 1., 0.)
#             ih = np.maximum(iymax - iymin + 1., 0.)
#             inters = iw * ih
#
#             # union
#             uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
#                    (BBGT[:, 2] - BBGT[:, 0] + 1.) *
#                    (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
#
#             overlaps = inters / uni
#             ovmax = np.max(overlaps)
#             jmax = np.argmax(overlaps)
#
#         if ovmax > ovthresh:
#             if not R['difficult'][jmax]:
#                 if not R['det'][jmax]:
#                     tp[d] = 1.
#                     R['det'][jmax] = 1
#                 else:
#                     fp[d] = 1.
#         else:
#             fp[d] = 1.
#
#     # compute precision recall
#     fp = np.cumsum(fp)
#     tp = np.cumsum(tp)
#     rec = tp / float(npos)
#     # avoid divide by zero in case the first detection matches a difficult
#     # ground truth
#     prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
#     ap = voc_ap(rec, prec, use_07_metric)
#
#     return rec, prec, ap


def sdd_eval(detpath, annopath, imageset_file, classname, annocache, ovthresh=0.5, use_07_metric=False):
    """
    pascal voc evaluation
    :param detpath: detection results detpath.format(classname)
    :param annopath: annotations annopath.format(classname)
    :param imageset_file: text file containing list of images
    :param classname: category name
    :param annocache: caching annotations
    :param ovthresh: overlap threshold
    :param use_07_metric: whether to use voc07's 11 point ap computation
    :return: rec, prec, ap
    """
    annocache = os.path.join(annocache, 'annotations.pkl')
    with open(imageset_file, 'r') as f:
        lines = f.readlines()
    image_filenames = [x.strip() for x in lines]

    # load annotations from cache
    if not os.path.isfile(annocache):
        recs = {}
        for ind, image_filename in enumerate(image_filenames):
            recs[image_filename] = parse_sdd_rec(annopath.format(image_filename))
            if ind % 100 == 0:
                print 'reading annotations for {:d}/{:d}'.format(ind + 1, len(image_filenames))
        print 'saving annotations cache to {:s}'.format(annocache)
        with open(annocache, 'wb') as f:
            cPickle.dump(recs, f, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        with open(annocache, 'rb') as f:
            recs = cPickle.load(f)

    # extract objects in :param classname:
    class_recs = {}
    npos = 0
    for image_filename in image_filenames:
        objects = [obj for obj in recs[image_filename] if obj['name'].lower() == classname]
        bbox = np.array([x['bbox'] for x in objects])
        difficult = np.array([x['difficult'] for x in objects]).astype(np.bool)
        det = [False] * len(objects)  # stand for detected
        npos = npos + sum(~difficult)
        class_recs[image_filename] = {'bbox': bbox,
                                      'difficult': difficult,
                                      'det': det}

    # read detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    bbox = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    if bbox.shape[0] > 0:
        sorted_inds = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        bbox = bbox[sorted_inds, :]
        image_ids = [image_ids[x] for x in sorted_inds]

    # go down detections and mark true positives and false positives
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        r = class_recs[image_ids[d]]
        bb = bbox[d, :].astype(float)
        ovmax = -np.inf
        bbgt = r['bbox'].astype(float)

        if bbgt.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(bbgt[:, 0], bb[0])
            iymin = np.maximum(bbgt[:, 1], bb[1])
            ixmax = np.minimum(bbgt[:, 2], bb[2])
            iymax = np.minimum(bbgt[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                   (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not r['difficult'][jmax]:
                if not r['det'][jmax]:
                    tp[d] = 1.
                    r['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid division by zero in case first detection matches a difficult ground ruth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = sdd_ap(rec, prec, use_07_metric)

    return rec, prec, ap