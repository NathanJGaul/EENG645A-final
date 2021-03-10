import os
import re
import typing
import datetime
from random import randrange
import math
import cv2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, TensorBoard
from tensorflow.python.keras.models import load_model
from tensorflow.keras.losses import mae
from tensorflow.keras.backend import elu
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
import segmentation_models as sm

def generate_informational_plots(file_root: str):
    # load in damage labels
    damage_path = os.path.join(file_root, "train.csv")
    damage = pd.read_csv(damage_path)
    damage = damage.to_numpy()

    # load in data stats
    images_folder = os.path.join(file_root, "train_images")
    not_damaged_num = 0
    damaged_num = 0
    damage_class_counts = np.bincount(list(damage[:,1]))[1:]
    image_damage_counts = [0, 0, 0, 0]

    files = os.listdir(images_folder)
    damaged_num = np.sum(np.isin(files,damage[:,0]))
    not_damaged_num = len(files) - damaged_num

    for file in set(files):
        image_damage_counts[list(damage[:,0]).count(file)] += 1

    # some insightful plots
    plt.figure()
    plt.bar(["Damaged", "Not Damaged"], [damaged_num, not_damaged_num])
    plt.ylabel("Count")
    plt.title("Damaged vs Not Damaged Count")
    plt.show()

    plt.figure()
    plt.bar(['1', '2', '3', '4'], damage_class_counts)
    plt.ylabel("Count")
    plt.xlabel("Classification")
    plt.title("Damage Classification Counts")
    plt.show()

    plt.figure()
    plt.bar(['0', '1', '2', '3'], image_damage_counts)
    plt.ylabel("Number of Images")
    plt.xlabel("Number of Damage Sites")
    plt.title("Damage Site Counts")
    plt.show()


def pixel_to_xy(pixel, image_size):
    (image_width, image_height) = image_size
    y = math.floor(pixel / image_width)
    x = pixel % image_width
    return x, y


def runlength_to_xys(runlength: str, image_size):
    runs = runlength.split(' ')
    xys = []
    for idx, val in enumerate(runs):
        if idx % 2 == 0:
            start_pixel = int(val)
            run = int(runs[idx+1])
            for pixel in range(start_pixel, start_pixel+run):
                xys.append(pixel_to_xy(pixel, image_size))

    desired_length = sum([int(x) for idx, x in enumerate(runs) if idx % 2 == 1])
    assert desired_length == len(xys), "Checking the length of the mask array"

    return xys


def runlength_to_greyscalendarray(runlength: str, image_size):
    width, height = image_size
    mask = np.zeros(width*height, dtype=np.uint8)
    runlength_list = list(map( int, runlength.split(" ")))
    positions = runlength_list[0::2]
    lengths = runlength_list[1::2]
    for pos, le in zip(positions, lengths):
        mask[pos-1:pos+le-1] = 1
    return mask.reshape((height, width), order="F")


def buildUNetModel1():
    # based on https://www.youtube.com/watch?v=azM57JuQpQI
    # UNet's original paper https://arxiv.org/abs/1505.04597v1

    input = Input(shape=(256, 1600, 1))
    a1 = Conv2D(16, (3, 3), activation="relu", padding="same")(input)
    a2 = Dropout(0.1)(a1)
    a3 = Conv2D(16, (3, 3), activation="relu", padding="same")(a2)
    a4 = MaxPooling2D((2, 2))(a3)


# https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# https://www.kaggle.com/go1dfish/clear-mask-visualization-and-simple-eda
def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.uint8) # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    return fname, masks

# https://www.kaggle.com/go1dfish/clear-mask-visualization-and-simple-eda
def show_mask(img, imageId, mask, palet):
    fig, ax = plt.subplots(figsize=(15, 15))

    for ch in range(4):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)
    ax.set_title(imageId)
    ax.imshow(img)
    plt.show()

# https://albumentations.ai/docs/examples/example_kaggle_salt/
def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

def main():
    file_root = os.path.join("/opt", "data", "gaul_severstal_data")
    log_dir = os.path.join("./logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    images_folder = os.path.join(file_root, "train_images")

    # color pallet for defect classification
    pallet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]

    # load in damage labels
    df_path = os.path.join(file_root, "train.csv")
    df = pd.read_csv(df_path)

    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    # rearrange dataframe for easier access to all defects in an image
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    # train-validation split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=42)

    # show the first image in the dataframe and its mask
    imageId, mask = make_mask(0, train_df)
    imageFile = os.path.join(images_folder, imageId)
    img = cv2.imread(imageFile)
    show_mask(img, imageId, mask, pallet)



if __name__ == "__main__":
    main()
