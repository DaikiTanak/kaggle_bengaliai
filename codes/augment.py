import torch
import numpy as np
import Image

class GaussianNoise:

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be added noise.

        Returns:
            PIL Image:
        """
        img_arr = np.array(img)
        img_arr.flags.writeable = True

        img_arr += np.random.normal(mean, std, img_arr.shape)

        added_pil = Image.fromarray(np.uint(img_arr))

        return added_pil
