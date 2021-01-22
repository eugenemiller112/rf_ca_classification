import os
from PIL import Image, ImageStat

# Returns the mean brightness of an image after converting to black/white
def brightness(im_file):
   im = Image.open(im_file).convert('L')
   stat = ImageStat.Stat(im)
   return stat.mean[0]

# Removes all images below a certain brightness threshold in a folder.
def low_activity_elim(folder, threshold):
    for f in os.listdir(folder):
        if f.endswith("png") | f.endswith("jpg"):
            p = os.path.join(folder,f)
            if brightness(p) < threshold:
                os.remove(p)

# Gives statistics on pixel brightnesses of images in a folder.
def image_statistics(path):
    ls12 = []
    for f in os.listdir(path):
        if f.endswith(".png") | f.endswith(".jpg"):
            b = brightness(os.path.join(path, f))
            ls12.append(b)
    ls12.sort(reverse=True)
    print(path)
    print("Average", sum(ls12) / len(ls12))
    print("Sorted Descending", ls12)
    print("Min", min(ls12))
    print("Median", median(ls12))