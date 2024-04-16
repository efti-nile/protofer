import cv2
import numpy as np
import random


class Fer2013:

    """
    A naive all-in-memory dataset class.
    """

    classes = [
        'angry',
        'disgust',
        'fear',
        'happy',
        'neutral',
        'sad',
        'surprise'
    ]

    def __init__(self, path):
        self.path = Path(path)

    def load_train(self):
        return self._load_data('train')

    def load_test(self):
        return self._load_data('test')

    def _load_data(self, stage):
        """
        Args:
            stage: can be 'train' or 'test'
        """
        pairs = []
        for i, cls in enumerate(self.classes):
            img_dir = self.path / stage / cls
            for img_pth in img_dir.glob('*.jpg'):
                pairs.append((cv2.imread(str(img_pth)), i))
        random.shuffle(pairs)
        x, y = list(zip(*pairs))
        return np.array(x), np.array(y)
