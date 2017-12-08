import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import matplotlib.gridspec as gridspec


import _init_paths
import caffe

caffe_root = os.path.join(os.path.dirname(__file__), '..', '..', 'caffe')


class HeatMap:
    def __init__(self, arch='VGG16'):
        # caffe.set_device(5)
        # caffe.set_mode_gpu()
        # caffe.set_mode_cpu()

        # Loading the caffe model
        if arch == 'CaffeNet':
            model_def = os.path.join(os.path.dirname(__file__), 'networks/caffenet_deploy_conv_only.prototxt')
            model_weights = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
        elif arch == 'VGG16':
            model_def = os.path.join(os.path.dirname(__file__), 'networks/VGG_ILSVRC_16_layers_conv_only.prototxt')
            model_weights = os.path.join(os.path.dirname(__file__), 'networks/VGG_ILSVRC_16_layers.caffemodel')
        elif arch == 'ResNet-50':
            model_def = os.path.join(os.path.dirname(__file__), 'networks/ResNet-50-deploy.prototxt')
            model_weights = os.path.join(os.path.dirname(__file__), 'networks/ResNet-50-model.caffemodel')
        elif arch == 'ResNet-101':
            model_def = os.path.join(os.path.dirname(__file__), 'networks/ResNet-101-deploy.prototxt')
            model_weights = os.path.join(os.path.dirname(__file__), 'networks/ResNet-101-model.caffemodel')

        self.net = caffe.Net(model_def, model_weights, caffe.TEST)
        self.arch = arch

    def get_map(self, image, verbose=False, layer_name='res3a'):

        image_shape = image.shape

        # Image Pre-processing
        transformer = caffe.io.Transformer({'data': (10, 3, image_shape[0], image_shape[1])})
        transformer.set_transpose('data', (2, 0, 1)) # Channel, Height, Width
        transformer.set_channel_swap('data', (2, 1, 0)) # The model expexts BGR
        transformed_image = transformer.preprocess('data', image)

        # Performing Forward Pass
        net = self.net
        net.blobs['data'].reshape(1, 3, image_shape[0], image_shape[1])
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()

        # Extracting the feature map
        if self.arch == 'CaffeNet':
            feat = net.blobs['conv5'].data[0]
        elif self.arch == 'VGG16':
            feat = net.blobs['conv5_3'].data[0]
        elif self.arch == 'ResNet-50':
            feat = net.blobs[layer_name].data[0]
        elif self.arch == 'ResNet-101':
            feat = net.blobs[layer_name].data[0]

        # Generating the mask
        feature_sum = np.sum(feat, axis=0)
        np.set_printoptions(threshold='nan')
        feature_sum = (255 * (feature_sum - np.min(feature_sum)) / np.ptp(feature_sum)).astype(int)
        threshold = feature_sum.mean()+20
        feature_sum = np.ma.masked_where(feature_sum <= threshold, feature_sum)

        # # Scaling the map to the input image size.
        # feature_sum = scipy.misc.imresize(feature_sum, 2.0, interp='bicubic')
        feature_sum = scipy.misc.imresize(feature_sum, image_shape, interp='bicubic')
        feature_sum = np.ma.masked_where(feature_sum <= threshold, feature_sum)

        # Scaling the map to the input image size.

        if verbose:
            print feature_sum

        # Visualizing the activations
        print feat.shape
        filtered = []
        for f in feat:
            if np.count_nonzero(f) > 120:
                f = (255 * (f - np.min(f)) / np.ptp(f)).astype(int)
                print f
                f = cv2.cvtColor(cv2.applyColorMap(np.uint8(f), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                filtered.append(f)


        filtered = np.array(filtered)
        print filtered.shape
        # self.vis_square(filtered)
        # self.display_images(filtered)
        self.save_images(filtered, '/home/joseph/output_grid.png')
        return feature_sum

    def display_images(self, images):
        plt.figure(figsize=(20, 10))
        columns = 20
        for i, image in enumerate(images):
            plt.subplot(len(images) / columns + 1, columns, i + 1)
            frame = plt.gca()
            frame.axes.get_xaxis().set_ticks([])
            frame.axes.get_yaxis().set_ticks([])
            plt.imshow(image)
        plt.show()

    def save_images(self, images, name):
        plt.figure(figsize=(20, 10))
        columns = 16
        for i, image in enumerate(images):
            plt.subplot(len(images) / columns + 1, columns, i + 1)
            frame = plt.gca()
            frame.axes.get_xaxis().set_ticks([])
            frame.axes.get_yaxis().set_ticks([])
            plt.imshow(image)
        plt.savefig(name, bbox_inches='tight')


    def save_images_closeby(self, images, name):
        plt.figure(figsize=(20, 10))
        columns = 20

        gs = gridspec.GridSpec(len(images) / columns + 1, columns, top=1., bottom=0., right=1., left=0., hspace=0.,
                               wspace=0.)
        for i, g in enumerate(gs):
            ax = plt.subplot(g)
            ax.imshow(images[0])
            ax.set_xticks([])
            ax.set_yticks([])
        # ax.set_aspect('auto')

        plt.savefig(name, bbox_inches='tight')

    def display_image(self, image):
        # plt.axis('off')
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        plt.imshow(image)
        plt.show()

    def vis_square(self, data):
        """Take an array of shape (n, height, width) or (n, height, width, 3)
           and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

        # normalize data for display
        # data = (data - data.min()) / (data.max() - data.min())

        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = (((0, n ** 2 - data.shape[0]),
                    (0, 5), (0, 5))  # add some space between filters
                   + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
        data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

        print data.shape

        plt.imshow(data);
        plt.axis('off')
        plt.show()



if __name__ == '__main__':
    print('Inside Main.')

    hm = HeatMap()

    image_path = os.path.join('/home/joseph/Dataset/voc_2012/VOCdevkit/VOC2012/JPEGImages/2007_004856.jpg')
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hMap = hm.get_map(img, verbose=False)

    # hm.display_image(cv2.cvtColor(cv2.applyColorMap(np.uint8(hMap), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB))
    # hm.display_image(hMap)

    # for i in range(1, 4):
    #     hMap = hm.get_map(img, verbose=True, layer_name='res3b'+str(i))
    #     hm.display_image(hMap)

    # for index in range(0, 23):
    #     image_path = os.path.join(
    #         '/home/cs17mtech01001/workspace/SDD-RFCN-python/data/detections/bookstore_video0_9500_hr_bc_pedestrian_'+str(index)+'.png')
    #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     hMap = hm.get_map(img, verbose=False)
    #     print 'Shape of the map', hMap.shape
    #     # hm.display_image(hMap)