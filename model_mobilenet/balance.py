import os
import sys

import cv2
import numpy as np
from albumentations import (Blur, Compose, HorizontalFlip,
                            RandomBrightnessContrast, ShiftScaleRotate)


# Define augmentation pipeline
aug = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.7),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    Blur(blur_limit=3, p=0.5)
])


def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames


def save_images_to_folder(images, filenames, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for img, filename in zip(images, filenames):
        cv2.imwrite(os.path.join(folder, filename), img)


def augment_images(images, n_samples):
    augmented_images = []
    while len(augmented_images) < n_samples:
        for img in images:
            augmented = aug(image=img)['image']
            augmented_images.append(augmented)
            if len(augmented_images) >= n_samples:
                break
    return augmented_images[:n_samples]


def balance_dataset(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    classes = os.listdir(input_folder)
    max_count = max(len(os.listdir(os.path.join(input_folder, cls))) for cls in classes)
    
    for cls in classes:
        class_input_path = os.path.join(input_folder, cls)
        class_output_path = os.path.join(output_folder, cls)
        
        images, filenames = load_images_from_folder(class_input_path)
        n_samples = max_count
        
        if len(images) < n_samples:
            augmented_images = augment_images(images, n_samples - len(images))
            images.extend(augmented_images)
            filenames.extend([f"aug_{i}.jpg" for i in range(len(augmented_images))])
        
        save_images_to_folder(images, filenames, class_output_path)


if __name__ == '__main__':
    balance_dataset(sys.argv[1], sys.argv[2])
