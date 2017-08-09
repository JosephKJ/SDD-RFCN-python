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
    if hr_method == "interpolation":
        img = scipy.misc.imresize(image, 8.0, interp='bicubic')
    else:
        # TODO: Super Resolution
        pass

    # 2. Get objectness

    return img