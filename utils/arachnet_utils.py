import cv2
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
import random

from tensorflow.keras.preprocessing.image import Iterator, ImageDataGenerator
from tensorflow.keras.utils import Sequence
from utils.image_preprocessing import PreProcess

import cv2 as cv
import numpy as np


def segmentation_feature_extraction(data, task="None"):
    def min_max_norm(x):
        x = x.astype(np.float32)
        return (x - x.min()) / (x.max() - x.min())

    image_rep = (min_max_norm(data) * 255).astype(np.uint8)
    # cv2.imshow("test", cv2.resize(image_rep, (256, 256)))
    # cv2.waitKey(0)
    sobs = sobel_img(image=image_rep)
    gaus_im = gaussInvar_img(image=image_rep)
    hu_rep = min_max_norm(data)

    digit_rep = np.digitize(hu_rep, bins=np.arange(0, 1, 0.3))
    _digit_rep = [hu_rep * (digit_rep == (i+1)) for i in range(3)]

    sobs, gaus_im, hu_rep = min_max_norm(sobs), min_max_norm(gaus_im), min_max_norm(hu_rep)

    # for 2PFM model
    return (gaus_im, sobs)

    # for 5PFM model
    # return (gaus_im, sobs, *_digit_rep)


def gaussInvar_img(image):
    img = image.copy()
    if len(image.shape) == 3 and image.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # optional , small difference
    sigma = random.randrange(2, 11, 3)
    gaussian_im = nd.gaussian_filter(img, sigma=sigma)
    return gaussian_im


def sobel_img(image):
    img = image.copy()
    if (len(image.shape) == 3) and (image.shape[2] == 3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_im = sobel(img)

    return sobel_im


def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


class GenImageDataGenerator(Sequence):

    def __init__(self, x, y, batch_size=32, arch="epu", preprocess="basic", debug=False, size=(96, 96),
                 image_channels=3, output_channels=1, interpolation=cv.INTER_CUBIC, multiple_outputs=None, **kwargs):
        def saliency(_x):
            _x = cv.cvtColor(_x, cv.COLOR_BGR2GRAY)
            _x = cv.resize(_x, (96, 96), interpolation=cv.INTER_CUBIC)
            _x = _x.astype(np.float32) / 255
            height, width = _x.shape
            return _x.reshape(height, width, 1)

        def segmentation(_x):
            _x = cv.cvtColor(_x, cv.COLOR_BGR2GRAY)
            _x = cv.resize(_x, (96, 96), interpolation=cv.INTER_CUBIC)
            _x = _x.astype(np.float32) / 255
            height, width = _x.shape
            return _x.reshape(height, width, 1)

        def autoencoder_process(_x):
            # _x = cv.cvtColor(_x, cv.COLOR_BGR2LAB)
            _x = _x.astype(np.float32) / 255
            return _x

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.debug = debug
        self.multiple_outputs = multiple_outputs
        self.preprocess = PreProcess(size=size, preprocess=preprocess, interpolation=interpolation)
        self.preprocess_y = PreProcess(size=size, preprocess=segmentation, interpolation=interpolation)
        self.generator = ImageDataGenerator(**kwargs)
        self.shape = (size[0], size[1], image_channels)
        self.output_shape = (size[0], size[1], output_channels)
        self.arch = arch

    def __len__(self):
        l = int(len(self.x) / self.batch_size)
        if l * self.batch_size < len(self.x):
            l += 1
        return l

    def __getitem__(self, index):
        x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        y = self.y[index * self.batch_size:(index + 1) * self.batch_size]

        x_batch = []
        y_batch = []
        for data_point_x, data_point_y in zip(x, y):
            transform = self.generator.get_random_transform(self.shape)
            x_batch.append(self.preprocess.get_processed_image(data_point_x,
                                                               generator=self.generator,
                                                               transform=transform))
            y_batch.append(self.preprocess_y.get_processed_image(data_point_y,
                                                                 generator=self.generator,
                                                                 transform=transform))

        del x
        del y

        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)

        if self.multiple_outputs is not None:
            supplementary_y = self.multiple_outputs * [(y_batch * 2) - 1]
            y_batch = [y_batch] + supplementary_y

        if self.arch is "epu":

            epu_x_batch = []
            batch_size, features, height, width, channels = x_batch.shape

            for i in range(features):
                feat_batch = x_batch[:, i, :, :, :].reshape(batch_size, height, width, channels)
                epu_x_batch.append(feat_batch)
            del x_batch
            return tuple(epu_x_batch), y_batch
        else:
            return x_batch, y_batch
