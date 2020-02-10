import torch


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
        pass
