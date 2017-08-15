import scipy

def generate_objectness_map(image, hr_method='interpolation'):
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

    return img