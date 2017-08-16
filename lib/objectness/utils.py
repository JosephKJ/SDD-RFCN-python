import scipy
import os
import cv2
import numpy as np
from map import HeatMap

def generate_objectness_map(heatMapObj, image, hr_method='interpolation'):
    """
    Given an image, the objecness map will be returned.
    Steps:
        1. Create a Higher Resolution Image
        2. Get objectness

    :param image:
    :return:
    """
    # 1. Create a Higher Resolution Image
    img = scipy.misc.imresize(image, 8.0, interp='bicubic')
    if hr_method == 'super_resolution':
        # TODO: Super Resolution
        pass

    # 2. Get objectness
    heat_map = heatMapObj.get_map(img)
    min_pixel_intensity = heat_map.min()
    binary_map = np.where(heat_map > min_pixel_intensity, 1, 0)

    # Trim off any extra rows in the map
    map_h, map_w = binary_map.shape
    img_h, img_w, _ = image.shape
    if map_h > img_h:
        diff = map_h - img_h
        binary_map = np.delete(binary_map, 1, axis=0)  # remove a row

    if map_w > img_w:
        diff = map_w - img_w
        binary_map = np.delete(binary_map, 1, axis=1)  # remove a column

    # Expand the map to three channels
    three_channel_map = np.stack((binary_map, binary_map, binary_map), axis=2)

    print type(image)

    # Applying map on the image
    image = image * three_channel_map

    print three_channel_map.shape
    print binary_map.shape
    print image.shape
    print heat_map.shape

    print type(image)

    # print heat_map
    # print 'min_pixel_intensity', min_pixel_intensity
    # print binary_map
    # heatMapObj.display_image(image)
    # heatMapObj.display_image(heat_map)
    # heatMapObj.display_image(binary_map)
    # print image.shape
    # print heat_map.shape


    return img

if __name__ == '__main__':
    print('Inside Main.')

    hm = HeatMap()

    image_path = os.path.join(
        '/home/cs17mtech01001/workspace/SDD-RFCN-python/data/detections/bookstore_video0_9500_pedestrian_7.png')
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    generate_objectness_map(hm, img)
