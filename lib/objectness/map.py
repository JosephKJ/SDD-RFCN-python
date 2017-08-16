import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

import _init_paths
import caffe

caffe_root = os.path.join(os.path.dirname(__file__), '..', '..', 'caffe')


def get_map(image, verbose=False):

    caffe.set_device(5)
    caffe.set_mode_gpu()

    # caffe.set_mode_cpu()

    # Loading the caffe model
    model_def = os.path.join(os.path.dirname(__file__), 'networks/caffenet_deploy_conv_only.prototxt')
    model_weights = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    image_shape = image.shape

    # Image Pre-processing
    transformer = caffe.io.Transformer({'data': (10, 3, image_shape[0], image_shape[1])})
    transformer.set_transpose('data', (2, 0, 1)) # Channel, Height, Width
    transformer.set_channel_swap('data', (2, 1, 0)) # The model expexts BGR
    transformed_image = transformer.preprocess('data', img)

    # Performing Forward Pass
    net.blobs['data'].reshape(10, 3, image_shape[0], image_shape[1])
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()

    # Extracting the feature map
    feat = net.blobs['conv5'].data[0]
    # visualize(feat, 'conv5')

    # Generating the mask
    feature_sum = np.sum(feat, axis=0)
    np.set_printoptions(threshold='nan')
    feature_sum = (255 * (feature_sum - np.min(feature_sum)) / np.ptp(feature_sum)).astype(int)
    feature_sum = np.ma.masked_where(feature_sum <= feature_sum.mean()+20, feature_sum)
    if verbose:
        print feature_sum

    plt.imshow(feature_sum)
    plt.axis('off')
    plt.show()

    # feat = net.blobs['pool5'].data[0]
    # visualize(feat, 'pool5')


def visualize(data, label):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.axis('off')
    # plt.savefig(os.path.join(caffe_root, 'activations', label+'.png'))
    plt.show()


def display_image(image):
    plt.axis('off')
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    print('Inside Main.')
    image_path = os.path.join(caffe_root, 'examples/images/cat.jpg')
    # image_path = os.path.join('/home/cs17mtech01001/workspace/SDD-RFCN-python/data/detections/bookstore_video0_9500_hr_bc_pedestrian_2.png')
    image_path = os.path.join('/home/cs17mtech01001/workspace/SDD-RFCN-python/data/detections/bookstore_video0_9500_hr_bc_skater_0.png')
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # display_image(img)
    get_map(img)

