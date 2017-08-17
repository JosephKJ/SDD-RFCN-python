import scipy
import os
import cv2
import numpy as np
from map import HeatMap

def generate_objectness_map(heatMapObj, image, hr_method='interpolation'):
    """
    Generates the objectness confidence score, for a given image.
    :param heatMapObj: An object of the heatmap Class
    :param image: The image which should be processed
    :param hr_method: optional, to so SR or not.
    :return: binary_map: which contains the objectness info; filtered_image: which is the map applied to the image.
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
    negative_binary_map = np.where(heat_map > min_pixel_intensity, 0, 1)

    # Trim off any extra rows in the map
    map_h, map_w = binary_map.shape
    img_h, img_w, _ = image.shape
    if map_h > img_h:
        diff = map_h - img_h
        binary_map = np.delete(binary_map, diff, axis=0)  # remove 'diff' rows
        negative_binary_map = np.delete(negative_binary_map, diff, axis=0)  # remove 'diff' rows

    if map_w > img_w:
        diff = map_w - img_w
        binary_map = np.delete(binary_map, diff, axis=1)  # remove 'diff' columns
        negative_binary_map = np.delete(negative_binary_map, diff, axis=1)  # remove 'diff' columns

    # Expand the map to three channels
    three_channel_map = np.stack((binary_map, binary_map, binary_map), axis=2)

    # Applying map on the image
    filtered_image = image * three_channel_map
    filtered_image = filtered_image.astype(np.uint8)

    return binary_map, negative_binary_map, filtered_image


def semantic_segment_image(heatMapObj, image, color='red'):
    # Getting the objectness
    binary_map, negative_binary_map, filtered_image = generate_objectness_map(heatMapObj, image)

    # Calculating the background
    three_channel_map = np.stack((negative_binary_map, negative_binary_map, negative_binary_map), axis=2)
    background = (image * three_channel_map).astype(np.uint8)

    # Segmentation Foreground
    r,g,b = get_rgb_from_color(color)
    foreground = np.stack((binary_map*r, binary_map*g, binary_map*b), axis=2).astype(np.uint8)

    # Combined Image
    full_image = background + foreground
    return full_image


def get_rgb_from_color(color):
    colors = {'red': (255, 83, 26), 'green': (26, 255, 83), 'blue': (26, 140, 255),
              'black': (77, 0, 77), 'white': (230, 230, 230), 'violet': (255, 26, 255)}
    return colors[color];


if __name__ == '__main__':
    print('Inside Main.')

    hm = HeatMap()

    image_path = os.path.join(
        '/home/cs17mtech01001/workspace/SDD-RFCN-python/data/detections/bookstore_video0_9500_pedestrian_2.png')
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    semantic_segment_image(hm, img, 'red')