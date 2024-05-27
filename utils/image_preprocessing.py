import cv2 as cv
import numpy as np
import pickle

class PreProcess(object):

    def __init__(self, size=(128, 128), preprocess="basic", interpolation=cv.INTER_CUBIC):
        """

        :param size:
        :param preprocess: needs to be function that pre-processes the image
        :param interpolation:OpenCV Flag
        """
        self._width, self._height = size
        self._interpolation = interpolation

        if preprocess is "basic":
            self._preprocess = PreProcess.basic_image_preprocess
        else:
            self._preprocess = preprocess

    def get_processed_image(self, image, generator=None, transform=None):

        cv_ext = ["png", "jpg"]
        pkl_ext = ["pkl", "pk"]

        if isinstance(image, str):
            ext = image.split(".")[-1]

            if ext in cv_ext:
                image = cv.imread(image, -1)
                image = self.resize(image)
            elif ext in pkl_ext:
                image = pickle.load(open(image, "rb"))
                image = self.resize(image)
            else:
                raise NameError

            if generator is not None and transform is not None:
                if len(image.shape) < 3:
                    image = np.expand_dims(image, axis=2)
                image = generator.apply_transform(image, transform)

            return self.__get_preprocess(image)

        elif isinstance(image, list):
            return self.get_processed_image_list(image, generator=generator, transform=transform)
        else:
            image = self.resize(image)

            if generator is not None and transform is not None:
                image = generator.apply_transform(image, transform)

            return self.__get_preprocess(image)

    def get_processed_image_list(self, images, generator=None, transform=None):
        processed_images = []
        for image in images:
            processed_images.append(self.get_processed_image(image, generator=generator, transform=transform))
        return processed_images

    def resize(self, image):
        return cv.resize(image, (self._width, self._width),
                         interpolation=self._interpolation)

    def __get_preprocess(self, image):
        if isinstance(self._preprocess, list):
            features = []
            for preprocess in self._preprocess:
                features.append(preprocess(image))
            return features
        else:
            return self._preprocess(image)

    @staticmethod
    def basic_image_preprocess(image):
        return PreProcess.min_max_norm(image)
        # return image.astype(np.float32) / 255.

    @staticmethod
    def min_max_norm(image):
        return (image.astype(np.float32) - image.min()) / ((image.max() - image.min()) + 1e-6)

    @staticmethod
    def basic_image_deprocess(image):
        return (PreProcess.min_max_norm(image) * 255).astype(np.uint8)

    @staticmethod
    def numerical_sorting(list_item):
        if type(list_item) is not list:
            raise Exception("needs list object")
        list_item.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        return list_item

    @staticmethod
    def center_crop(image, offset=96):
        height, width = image.shape[:2]
        x = width // 2 - (offset//2)
        y = height // 2 - (offset//2) + 50
        return image[y:y+offset, x:x+offset]