#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:15:50 2020

@author: meyerct6
"""


#https://stackoverflow.com/questions/55213599/u-net-with-pixel-wise-weighted-cross-entropy-input-dimension-errors
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    # Unet algorithm
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    # Output of the model
    conv10 = Conv2D(1, 1, activation='sigmoid', name='true_output')(conv9)

    # Define input of mask weights
    mask_weights = Input(input_size)

    # Define custom loss function which uses dice and binary cross entropy loss functions
    def dice_xent_loss(y_pred, y_true):
        """Adaptation of https://arxiv.org/pdf/1809.10486.pdf for multilabel
        classification with overlapping pixels between classes. Dec 2018.
        """
        # loss_dice = 0# weighted_dice(y_true, y_pred, mask_weights)
        loss_dice = tversky_loss(y_true, y_pred)
        loss_xent = weighted_binary_crossentropy(y_true, y_pred, mask_weights)
        return loss_dice + 10 ** 4 * loss_xent

    def wbce(y_pred, y_true):
        return weighted_binary_crossentropy(y_true, y_pred, mask_weights)

    # Model has 2 inputs (image and mask_weights) as well as one output the predicted image
    model = Model(inputs=[inputs, mask_weights], outputs=conv10)
    print("conv10", conv10)
    print(type(conv10))
    print("shape", conv10.shape)
    model.compile(optimizer=Adam(lr=3e-5), loss=dice_xent_loss, metrics=['accuracy'])
    print("type", type(model))
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


# Functions for calculating the weighted binary cross entropy
def weighted_binary_crossentropy(y_true, y_pred, weight_map):
    return tf.reduce_mean((K.binary_crossentropy(y_true,
                                                 y_pred) * weight_map)) / (tf.reduce_sum(weight_map) + K.epsilon())


# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
def tversky_loss(y_true, y_pred):
    alpha = .3
    beta = .7

    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2)) + beta * K.sum(p1 * g0, (0, 1, 2))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T


# Calculate dice loss
def weighted_dice(y_true, y_pred, weight_map):
    dice_numerator = 2.0 * K.sum(y_pred * y_true * weight_map, axis=[1, 2, 3])
    dice_denominator = K.sum(weight_map * y_true, axis=[1, 2, 3]) + \
                       K.sum(y_pred * weight_map, axis=[1, 2, 3])
    loss_dice = (dice_numerator) / (dice_denominator + K.epsilon())
    h1 = tf.square(tf.minimum(0.1, loss_dice) * 10 - 1)
    h2 = tf.square(tf.minimum(0.01, loss_dice) * 100 - 1)
    return 1.0 - tf.reduce_mean(loss_dice) + \
           tf.reduce_mean(h1) * 10 + \
           tf.reduce_mean(h2) * 10


#Generator for training
def trainGenerator(batch_size,train_path,image_folder,mask_folder,weights_folder,aug_dict_image,aug_dict_mask,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",weights_save_prefix = "weights",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict_image)
    weights_datagen = ImageDataGenerator(**aug_dict_mask)
    mask_datagen = ImageDataGenerator(**aug_dict_mask)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    weights_generator = weights_datagen.flow_from_directory(
        train_path,
        classes = [weights_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = weights_save_prefix,
        seed = seed)

    while True:
        X1i = image_generator.next() / 255.
        #X2i = np.ones(weights_generator.next().shape)
        X2i = weights_generator.next() / 255. + 1.
        X3i = np.round(mask_generator.next() / 255.).astype(int) + 1
        yield [X1i,X2i],X3i
#Generator for testing
def testGenerator(batch_size,train_path,image_folder,mask_folder,weights_folder,aug_dict_image,aug_dict_mask,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",weights_save_prefix = "weights",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict_image)
    mask_datagen = ImageDataGenerator(**aug_dict_mask)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)


    while True:
        X1i = image_generator.next() /255.
        X2i = np.ones(X1i.shape)
        X3i = np.round(mask_generator.next() / 255.).astype(int) + 1
        yield [X1i,X2i],X3i

#Generator for predicting
def predictGenerator(batch_size,train_path,image_folder,mask_folder,weights_folder,aug_dict_image,aug_dict_mask,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",weights_save_prefix = "weights",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict_image)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)


    while True:
        X1i = image_generator.next() / 255.
        X2i = np.ones(X1i.shape)
        X3i = np.ones(X1i.shape)
        yield [X1i,X2i],X3i
