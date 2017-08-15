import cv2
import matplotlib.pyplot as plt
import numpy as np

import caffe

def get_map(image):
    print 'Inside get_map.'


def display_image(image):
    plt.axis('off')
    plt.imshow(image)
    plt.draw()
    plt.show()


if __name__ == '__main__':
    print('Inside Main.')
    image_path = '/home/cs17mtech01001/workspace/SDD-RFCN-python/caffe/examples/images/cat.jpg'
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    display_image(img)
    get_map(img)

