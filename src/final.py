import os
import re
import typing
import datetime
from random import randrange
import math
import cv2

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
from ray import tune
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.schedulers import HyperBandScheduler


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

    print(f'Damaged: {damaged_num}, Undamaged: {not_damaged_num}, Total: {damaged_num+not_damaged_num}')
    # some insightful plots
    plt.figure()
    plt.bar(["Damaged", "Undamaged"], [damaged_num, not_damaged_num])
    plt.ylabel("Count")
    plt.title("Images Damaged vs Undamaged")
    plt.show()

    print(f'Class 1: {damage_class_counts[0]}, Class 2: {damage_class_counts[1]}, Class 3: {damage_class_counts[2]}, Class 4: {damage_class_counts[3]}')
    plt.figure()
    plt.bar(['1', '2', '3', '4'], damage_class_counts)
    plt.ylabel("Count")
    plt.xlabel("Classification")
    plt.title("Damage Classification Counts")
    plt.show()

    print(f'0 Damage Images: {image_damage_counts[0]}, 1 Damage Images: {image_damage_counts[1]}, 2 Damage Images: {image_damage_counts[2]}, 3 Damage Images: {image_damage_counts[3]}')
    plt.figure()
    plt.bar(['0', '1', '2', '3'], image_damage_counts)
    plt.ylabel("Number of Images")
    plt.xlabel("Number of Damage Sites")
    plt.title("Per Image Damage Classification Counts")
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

    num_classes = df.shape[1] - 1
    labels = df.iloc[row_id][:num_classes]
    masks = np.zeros((256, 1600, num_classes), dtype=np.uint8) # float32 is V.Imp
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
def mask_image(image, mask):
    # color pallet for defect classifications
    palette= [(255, 174, 188), (229, 231, 160), (200, 248, 180), (198, 231, 251)]

    image = np.copy(image)
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    num_classes = mask.shape[2]

    for ch in range(num_classes):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for j in range(0, len(contours)):
            cv2.polylines(image, contours[j], True, palette[ch], 2)

    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb
def visualize(image, mask, pred_mask=None, name=None, save_dir=None):
    """PLot images in one row."""
    num_classes = mask.shape[2]
    if pred_mask is not None:
        fig, axs = plt.subplots(2 + num_classes, 2)
        gs = GridSpec(6, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])

        # plot configs
        plt.setp([axs, ax1], xticks=[], yticks=[])
        for ax0 in axs:
            for ax in ax0:
                ax.axis('off')
        ax1.axis('off')
        fig.suptitle(name)

        ax1.imshow(image)
        axs[1][0].title.set_text("True Mask")
        axs[1][0].imshow(mask_image(image, mask))
        for ch in range(num_classes):
            axs[ch + 2][0].imshow(mask[..., ch])

        axs[1][1].title.set_text("Predicted Mask")
        axs[1][1].imshow(mask_image(image, pred_mask))
        for ch in range(num_classes):
            axs[ch + 2][1].imshow(pred_mask[..., ch])
    else:
        fig, axs = plt.subplots(2 + num_classes, 1)

        # plot configs
        plt.setp(axs, xticks=[], yticks=[])
        for ax in axs:
            ax.axis('off')
        fig.suptitle(name)

        axs[0].imshow(image)
        axs[1].imshow(mask_image(image, mask))
        for ch in range(num_classes):
            axs[ch + 2].imshow(mask[..., ch])

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, name), dpi=300)
    else:
        plt.show()

# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

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

def get_training_augmentation(resize_shape=None, interpolation=cv2.INTER_LINEAR):
    transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
    if resize_shape is not None:
        transform.append(
            A.Resize(height=resize_shape[0], width=resize_shape[1], interpolation=interpolation, always_apply=True, p=1),
        )
    return A.Compose(transform)

def get_validation_augmentation(resize_shape=None,):
    if resize_shape is not None:
        transform = [
            A.Resize(height=resize_shape[0], width=resize_shape[1], always_apply=True, p=1),
        ]
    return A.Compose(transform)

def get_testing_augmentation(resize_shape=None):
    if resize_shape is not None:
        transform = [
            A.Resize(height=resize_shape[0], width=resize_shape[1], always_apply=True, p=1),
        ]
    return A.Compose(transform)

def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

def load_dataframe_split(dataframe_path, classes=[1, 2, 3, 4], val_size=0.2, test_size=0.2, random_state=42):
    df = pd.read_csv(dataframe_path)

    # filter data by desired classes
    df = df[df['ClassId'].isin(classes)]

    # train/val/test splits
    val_size = int(np.floor(val_size * len(df)))
    test_size = int(np.floor(test_size * len(df)))

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

def get_model(backbone,
              encoder_freeze,
              n_classes,
              activation,
              dropout):

    base_model = sm.Unet(backend=backbone,
                         encoder_freeze=encoder_freeze,
                         classes=n_classes,
                         activation=activation,
                         encoder_weights='imagenet')

    # base unet
    input = base_model.input
    base_model_output = base_model.get_layer('final_conv').output

    # add drpoout
    base_model_output = keras.layers.Dropout(dropout)(base_model_output)

    # add activation
    output = keras.layers.Activation(activation, name=activation)(base_model_output)
    model = keras.models.Model(input, output)

    return model

def get_datasets(train_df,
                 val_df,
                 test_df,
                 images_dir,
                 resize_shape,
                 preprocessing_input,
                 use_greyscale):

    train_dataset = Dataset(
        dataframe=train_df,
        images_dir=images_dir,
        augmentation=get_training_augmentation(resize_shape=resize_shape),
        preprocessing=get_preprocessing(preprocessing_input),
        greyscale=use_greyscale,
    )

    val_dataset = Dataset(
        dataframe=val_df,
        images_dir=images_dir,
        augmentation=get_validation_augmentation(resize_shape=resize_shape),
        preprocessing=get_preprocessing(preprocessing_input),
        greyscale=use_greyscale,
    )

    test_dataset = Dataset(
        dataframe=test_df,
        images_dir=images_dir,
        augmentation=get_testing_augmentation(resize_shape=resize_shape),
        preprocessing=get_preprocessing(preprocessing_input),
        greyscale=use_greyscale,
    )

    return train_dataset, val_dataset, test_dataset

def get_dataloaders(train_dataset,
                    val_dataset,
                    test_dataset,
                    train_batch_size):

    train_dataloader = Dataloader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataloader = Dataloader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def dataset_test(train_df,
                 images_dir,
                 use_greyscale,
                 resize_shape):
    # test the dataset
    train_dataset = Dataset(dataframe=train_df, images_dir=images_dir, greyscale=use_greyscale)
    image, mask = train_dataset[5]
    augmented = get_training_augmentation(resize_shape=resize_shape)(image=image, mask=mask)
    aug_image=augmented['image']
    aug_mask=augmented['mask']

    # test visualization
    test_visualize = False
    if test_visualize:
        visualize(image, mask)
        visualize(aug_image, aug_mask)
        visualize(aug_image, aug_mask, aug_mask)

def train_unet(config, train_dataloader=None, val_dataloader=None, loss=None, metrics=None, checkpoint_dir=None):

    epochs = 15
    batch_size = 1

    model = get_model(backbone='vvg16', encoder_freeze=config["encoder_freeze"], n_classes=1,
                      activation='sigmoid', dropout=config["dropout"])

    model.compile(optimizer=config["optimizer"](config["learning_rate"]),
                  loss=loss,
                  metrics=metrics)

    history = model.fit(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=epochs,
        verbose=0,
        batch_size=batch_size,
        validation_data=val_dataloader,
        validation_steps=len(val_dataloader),
        callbacks=[
            TuneReportCallback({
                "loss": "loss",
                "iou_score": "iou_score",
                "val_loss": "val_loss",
                "val_iou_score": "val_iou_score",
            }, on="epoch_end"),
            TqdmCallback(verbose=2),
        ]
    )

    # save best model of the trial
    with tune.checkpoint_dir(step=1) as checkpoint_dir:
        checkpoint_dir = os.path.dirname(os.path.dirname(checkpoint_dir)) # go up two directories
        score_file_path = os.path.join(checkpoint_dir, 'score')
        score_file_exists = os.path.isfile(score_file_path)
        new_val_iou_score = history.history['val_iou_score'][0]
        best_model_file_path = os.path.join(checkpoint_dir, 'best_model.h5')

        if score_file_exists:
            old_val_iou_score = 0
            with open(score_file_path) as f:
                old_val_iou_score = float(f.read())
            if new_val_iou_score > old_val_iou_score:
                # we have a new best model
                with open(score_file_path, 'w') as f:
                    f.write(str(new_val_iou_score))
                model.save(best_model_file_path)
        else:
            # first model of the trial
            with open(score_file_path, 'w') as f:
                f.write(str(new_val_iou_score))
            model.save(best_model_file_path)

    print(history.history.keys())

def main():
    #sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

    # paths
    file_root = os.path.join("/opt", "data", "gaul_severstal_data")
    project_dir = os.path.join("/opt", "project")
    log_dir = os.path.join(project_dir, "src", "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    images_dir = os.path.join(file_root, "train_images")
    df_path = os.path.join(file_root, "train.csv")
    local_tune_dir = os.path.join(project_dir, "src", "tune")
    plot_dir = os.path.join(project_dir, "src", "plots")

    # dataset split
    training_split = 0.6
    validation_split = 0.2
    testing_split = 1.0 - (training_split + validation_split)

    # unet encoder backbone
    backbone = 'vgg16'

    # data options
    classes = [3]
    n_classes = len(classes)
    use_greyscale = False
    batch_size = 1

    # load in damage labels
    train_df, val_df, test_df = load_dataframe_split(df_path, classes=classes, val_size=validation_split, test_size=testing_split)

    # image resizing parameters
    image_scale_down = 5
    height = int(np.floor(256 / image_scale_down / 32) * 32)
    width = int(np.floor(1600 / image_scale_down / 32) * 32)
    image_channels = 1 if use_greyscale else 3
    resize_shape = (height, width, image_channels) # original is (256, 1600, 3), needs to be divisible by 32
    mask_shape = (height, width, n_classes)

    # image preprocessing
    preprocessing_input = sm.get_preprocessing(backbone)

    # loss
    dice_loss = sm.losses.DiceLoss()

    # metrics
    iou_score = sm.metrics.IOUScore(threshold=0.5)
    metrics = [iou_score]

    # datasets and dataloaders
    train_dataset, val_dataset, test_dataset = get_datasets(images_dir=images_dir,
                                                            preprocessing_input=preprocessing_input,
                                                            resize_shape=resize_shape,
                                                            train_df=train_df,
                                                            val_df=val_df,
                                                            test_df=test_df,
                                                            use_greyscale=use_greyscale)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_dataset=train_dataset,
                                                                        val_dataset=val_dataset,
                                                                        test_dataset=test_dataset,
                                                                        train_batch_size=batch_size)

    # tuner config
    config = {
        "dropout": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5]),
        "learning_rate": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
        "optimizer": tune.grid_search([keras.optimizers.Adam, keras.optimizers.RMSprop]),
        "encoder_freeze": tune.grid_search([True, False]),
    }

    # best model config
    best_config = {
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "optimizer": keras.optimizers.Adam,
        "encoder_freeze": False,
    }

    use_best_config = True
    config = best_config if use_best_config else config

    # search algorithm
    # hyperopt = HyperOptSearch(metric="val_iou_score", mode="max")

    # trial scheduler
    hyperband = HyperBandScheduler(metric="val_iou_score", mode="max")

    # tune model or load best_model.h5
    tune_model = True
    model_name = "best_model.h5"

    if tune_model:
        analysis = tune.run(
            tune.with_parameters(train_unet,
                                 train_dataloader=train_dataloader,
                                 val_dataloader=val_dataloader,
                                 loss=dice_loss,
                                 metrics=metrics),
            resources_per_trial={"gpu": 1},
            config=config,
            # search_alg=hyperopt,
            scheduler=hyperband,
            local_dir=local_tune_dir)
    else:
        model = keras.models.load_model(model_name,
                                        custom_objects={"dice_loss": dice_loss,
                                                        "iou_score": iou_score})

        use_test = True
        evaluate_dataset = test_dataset if use_test else val_dataset
        evaluate_dataloader = test_dataloader if use_test else val_dataloader

        make_plots = False
        if make_plots is True:
            for i in range(len(evaluate_dataset)):
                imageId = evaluate_dataset.ids[i]
                image, true_mask = evaluate_dataset[i]
                image = np.expand_dims(image, axis=0)
                pr_mask = model.predict(image)

                image = denormalize(image[0])
                pr_mask = denormalize(pr_mask[0]).astype('uint8')

                visualize(image, true_mask, pr_mask, name=imageId, save_dir=plot_dir)

        # model evaluation and baseline comparison
        evaluate_results = model.evaluate(evaluate_dataloader, batch_size=1)
        for i, val in enumerate(evaluate_results):
            print(f'{model.metrics_names[i]}: {val}')

        # baseline is a mask covering the entire left half of the image
        baseline_mask = np.zeros(mask_shape)
        baseline_mask[:,:np.int(width/2),:] = 1
        baseline_iou_scores = []
        for i in range(len(evaluate_dataset)):
            image, true_mask = evaluate_dataset[i]
            image = denormalize(image)
            iou = iou_score(true_mask.astype('float32'), baseline_mask)
            #visualize(image, true_mask, baseline_mask.astype('uint8'))
            baseline_iou_scores.append(iou)
        average_baseline_iou = np.average(baseline_iou_scores)
        print(f'baseline iou_score: {average_baseline_iou}')

if __name__ == "__main__":
    main()
