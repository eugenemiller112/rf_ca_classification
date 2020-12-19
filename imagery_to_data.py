# Greyscale image diff code
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageStat
from statistics import median
import ntpath
import time
import tempfile
import shutil
import math
import glob
import random
import os, sys

print(cv2.__version__)

# This code is modified from image_preprocessing.py @https://github.com/eugenemiller112/kralj_lab_2020
# Changes were made in accordance to necessity in the Aminoglycoside Sensitivity Detection project 2020-21.



def frame_extraction(video_dir, save_dir, final_frame, greyscale = False):
    directory = os.fsencode(video_dir)    # video directory
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        fileloc = os.path.join(video_dir, filename)  # file location as string
        # filenoext = os.path.splitext(filename)[0]
        if filename.endswith(".avi"):  # all .avi videos
            vidcap = cv2.VideoCapture(fileloc)  # read video
            success, image = vidcap.read()
            count = 0
            success = True
            if not os.path.exists(os.path.join(save_dir, filename)):
                os.makedirs(os.path.join(save_dir, filename))
            while success and (count <= final_frame):  # every time a new image is detected
                framename = "frame%d.jpg" % (count)
                save = os.path.join(os.path.join(save_dir, filename), framename)
                print(save)

                if greyscale:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                cv2.imwrite(save, image)  # save frame as JPEG file
                success, image = vidcap.read()
                count += 1
                continue
        else:
            continue

#  Removes borders of an image (note: overwrites original file)
def trim(path):
    im = Image.open(path)
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))  # looks at top left pixel to determine the border color
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -1)
    bbox = diff.getbbox()  # creates a mask
    if bbox:
        cropped = im.crop(bbox)
        cropped.save(path)  # saves image

# New function!
def resize_squishy(image_path, sz, interpolate = False):
    im = cv2.imread(image_path)
    if interpolate:
        return cv2.resize(im, (sz, sz), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(im, (sz,sz))


# Greyscale image diff code. Looks at images in read_dir for images matching names generated by frame_extraction and
# matching the frames specified (first_frame, last_frame). Then calculates the difference in greyscale pixel values
# and generates a new image to be saved in save_dir.
def diff_imager(read_dir, save_dir, first_frame, last_frame, saved_frame = None):
    name = os.path.basename(read_dir)
            # read in the desired first and last frames

    img2 = cv2.imread(os.path.join(read_dir, 'frame%d.jpg' % last_frame))

    # to boost efficiency, if the first frame was already read it can be handed to diff_imager to save time from cv2.imread()
    if saved_frame is None:
        img1 = cv2.imread(os.path.join(read_dir, 'frame%d.jpg' % first_frame))
    else:
        img1 = saved_frame
            # try converting these to doubles -- to increase resolution.
            # diff has the required difference data
    try:
        diff = np.abs(img1.astype(np.uint) - img2.astype(np.uint)).astype(np.uint8)
    except ValueError:      # in case for some reason the images were trimmed improperly, skips the iteration
        print("ValueError encountered")

            # Convert from array and save as image
    img = Image.fromarray(diff)
    save = os.path.join(save_dir, (name + "diff(%d - %d).png" % (first_frame, last_frame)))
    print(save)
    img.save(save)
    return img2