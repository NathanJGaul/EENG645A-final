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
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow_probability as tfp
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tqdm.keras import TqdmCallback

import albumentations as A
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
def show_mask(img, mask, palette):
    fig, ax = plt.subplots(figsize=(15, 15))

    for ch in range(4):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palette[ch], 2)
    ax.imshow(img)
    plt.show()

# https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    plt.tight_layout()
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(n, 1, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

# class for managing the dataset
# used example https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb
class Dataset:
    def __init__(
            self,
            dataframe,
            images_dir,
            augmentation=None,
            preprocessing=None,
            greyscale=False,
    ):
        self.df = dataframe
        self.ids = list(self.df.index)
        self.images_dir = images_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.greyscale = greyscale

    def __getitem__(self, i):
        imageId = self.ids[i]
        image = cv2.imread(os.path.join(self.images_dir, imageId))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if self.greyscale else cv2.COLOR_BGR2RGB)
        _, mask = make_mask(i, self.df)

        # apply augmentation
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.greyscale:
            image = np.expand_dims(image,axis=-1)

        return image, mask

    def __len__(self):
        return len(self.ids)

class Dataloader(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:

    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return batch[0],batch[1].astype('float32')

    def __len__(self):
        """Denotes number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

def get_training_augmentation(resize_size):
    train_transform = [
        A.Resize(height=resize_size[0], width=resize_size[1], always_apply=True, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation(resize_size):
    train_transform = [
        A.Resize(height=resize_size[0], width=resize_size[1], always_apply=True, p=1),
    ]
    return A.Compose(train_transform)

def get_testing_augmentation(resize_size):
    train_transform = [
        A.Resize(height=resize_size[0], width=resize_size[1], always_apply=True, p=1),
    ]
    return A.Compose(train_transform)

def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

def load_dataframe_split(dataframe_path, val_size=0.2, test_size=0.2, random_state=42):
    df = pd.read_csv(dataframe_path)

    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    # rearrange dataframe for easier access to all defects in an image
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    # break out testing data
    df, train_df = train_test_split(df, test_size=test_size, stratify=df["defects"], random_state=random_state)

    # train-validation split
    train_df, val_df = train_test_split(df, test_size=val_size, stratify=df["defects"], random_state=random_state)

    return train_df, val_df, train_df

# https://github.com/zhixuhao/unet/blob/master/model.py
def unet(input_size, n_classes, pretrained_weights = None):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(n_classes, 1, activation = 'softmax')(conv9)

    model = Model(inputs, conv10)
    return model

def main():
    #sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

    # paths
    file_root = os.path.join("/opt", "data", "gaul_severstal_data")
    log_dir = os.path.join("./logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    images_dir = os.path.join(file_root, "train_images")

    # color pallet for defect classification
    palette= [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]

    # load in damage labels
    df_path = os.path.join(file_root, "train.csv")
    train_df, val_df, test_df = load_dataframe_split(df_path)

    # test the dataset
    train_dataset = Dataset(dataframe=train_df, images_dir=images_dir)
    image, mask = train_dataset[5]
    #show_mask(image, mask, palette)

    # basic network hyperparameters
    backbone = 'vgg16'
    batch_size = 8
    lr = 0.0001
    epochs = 5
    activation = 'softmax'
    n_classes = 4

    # convert images to greyscale
    use_greyscale = False
    # image resizing
    image_scale_down = 4
    height = int(np.floor(256 / image_scale_down / 32) * 32)
    width = int(np.floor(1600 / image_scale_down / 32) * 32)
    resize_size = (height, width) # original is (256, 1600), needs to be divisible by 32

    image_shape = (resize_size[0], resize_size[1], 1 if use_greyscale else 3)

    # create model
    #model = sm.Unet(backbone, classes=n_classes, activation=activation)
    model = unet(input_size=image_shape, n_classes=n_classes)

    # preprocessing
    preprocessing_input = sm.get_preprocessing(backbone)

    # optimizer
    optimizer = keras.optimizers.Adam(lr)

    # losses
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + focal_loss

    # metrics
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile model
    model.compile(optimizer, total_loss, metrics)

    # Dataset for training
    train_dataset = Dataset(
        dataframe=train_df,
        images_dir=images_dir,
        augmentation=get_training_augmentation(resize_size=resize_size),
        preprocessing=get_preprocessing(preprocessing_input),
        greyscale=use_greyscale,
    )

    # Dataset for validation
    val_dataset = Dataset(
        dataframe=val_df,
        images_dir=images_dir,
        augmentation=get_validation_augmentation(resize_size=resize_size),
        preprocessing=get_preprocessing(preprocessing_input),
        greyscale=use_greyscale,
    )

    # Dataset for validation
    test_dataset = Dataset(
        dataframe=test_df,
        images_dir=images_dir,
        augmentation=get_training_augmentation(resize_size=resize_size),
        preprocessing=get_preprocessing(preprocessing_input),
        greyscale=use_greyscale,
    )

    # Dataloaders
    train_dataloader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = Dataloader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)

    # check shapes for errors
    assert train_dataloader[0][0].shape == (batch_size, resize_size[0], resize_size[1], 1 if use_greyscale else 3)
    assert val_dataloader[0][0].shape == (1, resize_size[0], resize_size[1], 1 if use_greyscale else 3)
    assert test_dataloader[0][0].shape == (1, resize_size[0], resize_size[1], 1 if use_greyscale else 3)
    assert train_dataloader[0][1].shape == (batch_size, resize_size[0], resize_size[1], n_classes)
    assert val_dataloader[0][1].shape == (1, resize_size[0], resize_size[1], n_classes)
    assert test_dataloader[0][1].shape == (1, resize_size[0], resize_size[1], n_classes)

    # define callbacks
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=log_dir),
        keras.callbacks.ModelCheckpoint('./src/best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
        TqdmCallback(verbose=2),
    ]

    # train model
    history = model.fit(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=epochs,
        verbose=0,
        callbacks=callbacks,
        validation_data=val_dataloader,
        validation_steps=len(val_dataloader),
    )

if __name__ == "__main__":
    main()
