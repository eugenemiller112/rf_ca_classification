# Greyscale image diff code
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageStat
from statistics import median
from datetime import datetime
import ntpath
import time
import tempfile
import shutil
import math
import glob
import random

import os, sys

#from laplace_of_gaussian import LoGFilter
from sobel import sobelFilter
from scipy import ndimage

from matplotlib import pyplot as plt

print(cv2.__version__)

# This code is modified from image_preprocessing.py @https://github.com/eugenemiller112/kralj_lab_2020
# Changes were made in accordance to necessity in the Aminoglycoside Sensitivity Detection project 2020-21.

def data_gen_sobel_diff_figs(data_path, small_delta, diff_upper_bound):    #data path assumed to have videos of AVIS separated into folders based on class

    save1 = r"D:\Fig_Data"
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y, %H-%M-%S")
    newdir = os.path.join(save1, date_time)
    os.mkdir(newdir)

    for cl in os.listdir(data_path):

        print("cl", cl)

        os.mkdir(os.path.join(newdir, cl))
        save2 = os.path.join(newdir, cl)
        temp_dir = tempfile.mkdtemp()
        temp_frames_dir = os.path.join(temp_dir, "framestemp"+str(cl))
        os.mkdir(temp_frames_dir)

        print("temp frames dir", temp_frames_dir)

        viddir = os.path.join(data_path, cl)

        for subfolder in os.listdir(viddir):

            print("subfolder", subfolder)

            save3 = os.path.join(save2, subfolder)
            os.mkdir(save3)
            print("save3", save3)
            subdir = os.path.join(viddir, subfolder)

            for fl in os.listdir(subdir):

                print("fl", fl)

                movpath = os.path.join(subdir, fl)

                if fl == "movie_GreenColor.avi":

                    print("movpath", movpath)

                    filename = os.fsdecode(fl)
                    fileloc = movpath
                    save_dir = temp_frames_dir
                    vidcap = cv2.VideoCapture(fileloc)  # read video
                    success, image = vidcap.read()
                    count = 0
                    success = True
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    while success :  # every time a new image is detected
                        framename = "frame%d.jpg" % (count)
                        if not os.path.exists(os.path.join(save_dir, filename.replace(".avi", "\\"))):
                            os.mkdir(os.path.join(save_dir, filename.replace(".avi", "\\")))
                        save = os.path.join(save_dir, filename.replace(".avi", "\\") + framename)
                        print(save)

                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                        cv2.imwrite(save, image)  # save frame as JPEG file
                        trim(save)
                        success, image = vidcap.read()
                        count += 1
                        continue
                    break

            p = os.path.join(temp_frames_dir, "movie_GreenColor")
            for frame in os.listdir(p):

                print("frame", frame)
                im = Image.fromarray(resize_squishy(os.path.join(p,frame), 256))
                im = sobelFilter(im)
                #plt.imshow(im)
                im = Image.fromarray(im)
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                im.save(os.path.join(p,frame)) # Resize them
                print(str(os.path.join(p,frame)))
                #quit()

            savedframes = np.empty(diff_upper_bound,
                                   dtype=object)  # array of proper size is declared


            for i in range(0, diff_upper_bound - small_delta):
                if i + 1 >= small_delta:
                    savedframes[i] = diff_imager(p, save3, i, i + small_delta,
                                                 saved_frame=savedframes[i - diff_upper_bound])
                    continue
                savedframes[i] = diff_imager(p, save3, i, i + small_delta)
            # Generate diff images



        shutil.rmtree(temp_frames_dir)
        print("Done")
        print(newdir)
    return newdir

def frame_extraction(video_dir, save_dir, final_frame, greyscale = False):
    directory = os.fsencode(video_dir)    # video directory
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        fileloc = os.path.join(video_dir, filename)  # file location as string
        if filename.endswith(".avi"):  # all .avi videos
            vidcap = cv2.VideoCapture(fileloc)  # read video
            success, image = vidcap.read()
            count = 0
            success = True
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            while success and (count <= final_frame):  # every time a new image is detected
                framename = "frame%d.jpg" % (count)
                if not os.path.exists(os.path.join(save_dir, filename.replace(".avi", "\\"))):
                    os.mkdir(os.path.join(save_dir, filename.replace(".avi", "\\")))
                save = os.path.join(save_dir, filename.replace(".avi", "\\") + framename)
                print(save)

                if greyscale:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                cv2.imwrite(save, image)  # save frame as JPEG file
                trim(save)
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
    diff = ImageChops.add(diff, diff, 2.0, -200)
    bbox = diff.getbbox()  # creates a mask
    if bbox:
        cropped = im.crop(bbox)
        cropped.save(path)  # saves image

# New function!
def resize_squishy(image_path, sz, interpolate = False):
    im = cv2.imread(image_path)
    if interpolate:
        return cv2.resize(im, (sz, sz), interpolation=cv2.INTER_NEAREST)
    print(len(cv2.resize(im, (sz,sz))))
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
        print(os.path.join(read_dir, 'frame%d.jpg' % first_frame))
    else:
        img1 = saved_frame
            # diff has the required difference data
    try:
        diff = np.abs(img1.astype(np.uint) - img2.astype(np.uint)).astype(np.uint8)
        img = Image.fromarray(diff)
        save = os.path.join(save_dir, (name + "diff(%d - %d).png" % (first_frame, last_frame)))
        print(save)
        img.save(save)
        return img2
    except ValueError:      # in case for some reason the images were trimmed improperly, skips the iteration
        print("ValueError encountered")


            # Convert from array and save as image


def data_gen_diff(data_path, small_delta, diff_upper_bound):    #data path assumed to have videos of AVIS separated into folders based on class
    save = r"D:\ASD"
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y, %H-%M-%S")
    newdir = os.path.join(save, date_time)
    os.mkdir(newdir)


    for cl in os.listdir(data_path):
        os.mkdir(os.path.join(newdir, cl))
        save = os.path.join(newdir, cl)
        temp_dir = tempfile.mkdtemp()
        temp_frames_dir = os.path.join(temp_dir, "framestemp"+str(cl))
        viddir = os.path.join(data_path,cl)
        frame_extraction(viddir, temp_frames_dir, (diff_upper_bound + small_delta), greyscale=True)   # Read videos

        for vid in os.listdir(temp_frames_dir):
            p = os.path.join(temp_frames_dir, vid)
            for fl in os.listdir(p):
                print("fl", fl)
                trim(os.path.join(p,fl)) # Trim frames
                im = Image.fromarray(resize_squishy(os.path.join(p,fl), 256))
                im.save(os.path.join(p,fl)) # Resize them

        # Generate diff images

        for fl in os.listdir(temp_frames_dir):

            path = os.path.join(temp_frames_dir, fl)

            print("path",path)

            savedframes = np.empty(diff_upper_bound,
                                            dtype=object)  # array of proper size is declared

            for i in range(0, diff_upper_bound):
                if i + 1 >= small_delta:
                    savedframes[i] = diff_imager(path, save, i, i+small_delta, saved_frame = savedframes[i-diff_upper_bound])
                    continue
                savedframes[i] = diff_imager(path, save, i, i + small_delta)




        print(temp_frames_dir)
        print(os.listdir(temp_frames_dir))
        shutil.rmtree(temp_frames_dir)
        print("Done")
        print(save)
    return newdir

def data_gen_sobel_diff(data_path, small_delta, diff_upper_bound):    #data path assumed to have videos of AVIS separated into folders based on class
    save = r"D:\ASD"
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y, %H-%M-%S")
    newdir = os.path.join(save, date_time)
    os.mkdir(newdir)


    for cl in os.listdir(data_path):
        os.mkdir(os.path.join(newdir, cl))
        save = os.path.join(newdir, cl)
        temp_dir = tempfile.mkdtemp()
        temp_frames_dir = os.path.join(temp_dir, "framestemp"+str(cl))
        viddir = os.path.join(data_path,cl)
        frame_extraction(viddir, temp_frames_dir, diff_upper_bound , greyscale=True)   # Read videos

        for vid in os.listdir(temp_frames_dir):
            p = os.path.join(temp_frames_dir, vid)
            for fl in os.listdir(p):
                print("fl", fl)
                trim(os.path.join(p,fl)) # Trim frames
                print(os.path.join(p,fl))
                quit()
                im = Image.fromarray(resize_squishy(os.path.join(p,fl), 256))
                im = sobelFilter(im)
                im = Image.fromarray(im, mode="RGB")
                im.save(os.path.join(p,fl)) # Resize them

        # Generate diff images

        for fl in os.listdir(temp_frames_dir):

            path = os.path.join(temp_frames_dir, fl)

            print("path",path)

            savedframes = np.empty(diff_upper_bound,
                                            dtype=object)  # array of proper size is declared

            for i in range(0, diff_upper_bound - small_delta):
                if i + 1 >= small_delta:
                    savedframes[i] = diff_imager(path, save, i, i+small_delta, saved_frame = savedframes[i-diff_upper_bound])
                    continue
                savedframes[i] = diff_imager(path, save, i, i + small_delta)




        print(temp_frames_dir)
        print(os.listdir(temp_frames_dir))
        shutil.rmtree(temp_frames_dir)
        print("Done")
        print(save)
    return newdir

def data_gen_LoG_diff(data_path, small_delta, diff_upper_bound):    #data path assumed to have videos of AVIS separated into folders based on class
    save = r"D:\ASD"
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y, %H-%M-%S")
    newdir = os.path.join(save, date_time)
    os.mkdir(newdir)


    for cl in os.listdir(data_path):
        os.mkdir(os.path.join(newdir, cl))
        save = os.path.join(newdir, cl)
        temp_dir = tempfile.mkdtemp()
        temp_frames_dir = os.path.join(temp_dir, "framestemp"+str(cl))
        viddir = os.path.join(data_path,cl)
        frame_extraction(viddir, temp_frames_dir, (diff_upper_bound + small_delta), greyscale=True)   # Read videos

        for vid in os.listdir(temp_frames_dir):
            p = os.path.join(temp_frames_dir, vid)
            for fl in os.listdir(p):
                print("fl", fl)
                trim(os.path.join(p,fl)) # Trim frames
                im = Image.fromarray(resize_squishy(os.path.join(p,fl), 256))
                im = LoGFilter(im)
                im = Image.fromarray(im, mode="RGB")
                im.save(os.path.join(p,fl)) # Resize them

        # Generate diff images

        for fl in os.listdir(temp_frames_dir):

            path = os.path.join(temp_frames_dir, fl)

            print("path",path)

            savedframes = np.empty(diff_upper_bound,
                                            dtype=object)  # array of proper size is declared

            for i in range(0, diff_upper_bound):
                if i + 1 >= small_delta:
                    savedframes[i] = diff_imager(path, save, i, i+small_delta, saved_frame = savedframes[i-diff_upper_bound])
                    continue
                savedframes[i] = diff_imager(path, save, i, i + small_delta)




        print(temp_frames_dir)
        print(os.listdir(temp_frames_dir))
        shutil.rmtree(temp_frames_dir)
        print("Done")
        print(save)
    return newdir



#data_gen_sobel_diff_figs(r'C:\Users\eugmille\Desktop\Fig Dat - Kan Only', 5, 240)