from keras import backend as K
import inception_v4
import numpy as np
import cv2
import os
import skimage.measure
from tqdm import tqdm

# This function comes from Google's ImageNet Preprocessing Script
def central_crop(image, central_fraction):
    """Crop the central region of the image.
    Remove the outer parts of an image but retain the central region of the image
    along each dimension. If we specify central_fraction = 0.5, this function
    returns the region marked with "X" in the below diagram.
       --------
      |        |
      |  XXXX  |
      |  XXXX  |
      |        |   where "X" is the central 50% of the image.
       --------
    Args:
    image: 3-D array of shape [height, width, depth]
    central_fraction: float (0, 1], fraction of size to crop
    Raises:
    ValueError: if central_crop_fraction is not within (0, 1].
    Returns:
    3-D array
    """
    if central_fraction <= 0.0 or central_fraction > 1.0:
        raise ValueError('central_fraction must be within (0, 1]')
    if central_fraction == 1.0:
        return image

    img_shape = image.shape
    depth = img_shape[2]
    fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
    bbox_h_start = int(np.divide(img_shape[0], fraction_offset))
    bbox_w_start = int(np.divide(img_shape[1], fraction_offset))

    bbox_h_size = int(img_shape[0] - bbox_h_start * 2)
    bbox_w_size = int(img_shape[1] - bbox_w_start * 2)

    image = image[bbox_h_start:bbox_h_start + bbox_h_size, bbox_w_start:bbox_w_start + bbox_w_size]
    return image


def get_processed_image(img_path):
    # Load image and convert from BGR to RGB
    im = np.asarray(cv2.imread(img_path))[:, :, ::-1]
    im = central_crop(im, 0.875)
    im = cv2.resize(im, (299, 299))
    im = inception_v4.preprocess_input(im)
    if K.image_data_format() == "channels_first":
        im = np.transpose(im, (2, 0, 1))
        im = im.reshape(-1, 3, 299, 299)
    else:
        im = im.reshape(-1, 299, 299, 3)
    return im


def poolingPreds(preds):
    preds = preds.reshape(8, 8, -1)
    res = skimage.measure.block_reduce(preds, (8, 8, 1), np.max)
    return res.reshape(-1)


def predict(model, path):
    img = get_processed_image(path)
    preds = model.predict(img)
    return preds


if __name__ == "__main__":
    model = inception_v4.create_model(weights='imagenet', include_top=False)

    for c in tqdm(range(1, 31)):
        basePath = '../../data/train/' + str(c)
        imageName = os.listdir(basePath)
        feature_list = np.array([[]])
        for i in imageName:
            if i.startswith('.'):
                continue
            f = poolingPreds(predict(model, os.path.join(basePath, i)))
            if feature_list.size == 0:
                feature_list = np.array([f])
            else:
                feature_list = np.concatenate((feature_list, np.array([f])), axis=0)
        np.save('../../feature/train/'+str(c)+'.npy', feature_list)
