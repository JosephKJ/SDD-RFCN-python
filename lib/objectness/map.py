import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

import _init_paths
import caffe

caffe_root = os.path.join(os.path.dirname(__file__), '..', '..', 'caffe')


def get_map(image):
    print 'Inside get_map.'
    # caffe.set_device(0)
    # caffe.set_mode_gpu()

    caffe.set_mode_cpu()

    model_def = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/deploy.prototxt')
    model_weights = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

    net = caffe.Net(model_def, model_weights, caffe.TEST)

    print 'Network Loaded Successfully.'

    print '\nParameters\n'
    for layer_name, param in net.params.iteritems():
        print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

    print '\nBlobs\n'
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape)



def display_image(image):
    plt.axis('off')
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    print('Inside Main.')
    image_path = os.path.join(caffe_root, 'examples/images/cat.jpg')
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # display_image(img)
    get_map(img)

