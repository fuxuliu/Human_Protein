'''
Given folders of the full images from the human protein atlas, extract single cell crops.
To obtain single cell crops, an otsu filter is used to segment the nuclei channel, and then large enough connected
components in the nuclei channel are used as the centers for these crops.
Author: Alex Lu
Email: alexlu@cs.toronto.edu
Copyright (C) 2018 Alex Lu
'''

from PIL import Image
import numpy as np
import os
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from skimage.measure import label
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.ndimage.measurements import center_of_mass

import warnings
warnings.filterwarnings("ignore")

'''
Given an image, extract single cell crops:
ARGUMENTS:
imagepath: full path of the image
foldername: name of the folder image is in
imagename: name of the jpeg file
savepath: directory to save single cell crops to (will save in a subdirectory named after the folder)
outfile: file to write coordinates of cell centers to (for debugging purposes)
scale: scale to downsize original images by
cropsize: size of square crops (in pixels on the rescaled image) to extract
'''


def find_centers_and_crop(imagepath, foldername, imagename, savepath, outfile, scale=4, cropsize=128):
    # Get the image and resize
    color_image = np.array(Image.open(imagepath))
    print("Working on", imagename)
    image_shape = color_image.shape[:2]
    image_shape = tuple(ti//scale for ti in image_shape)
    color_image = resize(color_image, image_shape)

    # Split the image into channels
    microtubules = color_image[:, :, 0]
    antibody = color_image[:, :, 1]
    nuclei = color_image[:, :, 2]

    # Segment the nuclear channel and get the nuclei
    min_nuc_size = 100.0

    val = threshold_otsu(nuclei)
    smoothed_nuclei = gaussian(nuclei, sigma=5.0)
    binary_nuclei = smoothed_nuclei > val
    binary_nuclei = remove_small_holes(binary_nuclei, min_size=300)
    labeled_nuclei = label(binary_nuclei)
    labeled_nuclei = clear_border(labeled_nuclei)
    labeled_nuclei = remove_small_objects(
        labeled_nuclei, min_size=min_nuc_size)

    # Iterate through each nuclei and get their centers (if the object is valid), and save to directory
    for i in range(1, np.max(labeled_nuclei)):
        current_nuc = labeled_nuclei == i
        if np.sum(current_nuc) > min_nuc_size:
            y, x = center_of_mass(current_nuc)
            x = np.int(x)
            y = np.int(y)

            c1 = y - cropsize // 2
            c2 = y + cropsize // 2
            c3 = x - cropsize // 2
            c4 = x + cropsize // 2

            if c1 < 0 or c3 < 0 or c2 > image_shape[0] or c4 > image_shape[1]:
                pass
            else:
                nuclei_crop = nuclei[c1:c2, c3:c4]
                antibody_crop = antibody[c1:c2, c3:c4]
                microtubule_crop = microtubules[c1:c2, c3:c4]

                folder_suffix = imagename.rsplit("_", 4)[0]
                outfolder = savepath + foldername + "_" + folder_suffix
                outimagename = imagename.rsplit("_", 3)[0] + "_" + str(i)

                if not os.path.exists(outfolder):
                    os.mkdir(outfolder)

                Image.fromarray(nuclei_crop).save(
                    outfolder + "//" + outimagename + "_blue.tif")
                Image.fromarray(antibody_crop).save(
                    outfolder + "//" + outimagename + "_green.tif")
                Image.fromarray(microtubule_crop).save(
                    outfolder + "//" + outimagename + "_red.tif")

                output = open(outfile, "a")
                output.write(foldername + "_" +
                             folder_suffix + "/" + outimagename)
                output.write("\t")
                output.write(str(x))
                output.write("\t")
                output.write(str(y))
                output.write("\n")
                output.close()


if __name__ == "__main__":
    '''Loop to call the cell crop segmentation on all folders in a directory. If you used the download_hpa.py file
    to obtain the HPA images, they will already be in the format required by this script.'''

    # Path of the downloaded HPA images
    filepath = "../human_protein_atlas/"
    # Path to save the crops to
    outpath = "../human_protein_atlas_single_cell/"
    # Path to save file of cell center coordinates to
    outfile = "../human_protein_atlas_single_cell_centers.txt"

    # Creates output folder if necessary
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # Loop over all folders in the input folder and extract single cell crops for all images
    # Writes single cell crops to sub-directories in the outpath folder,
    # named identical to the sub-directories in the input folder
    for directory in os.listdir(filepath):
        for image in os.listdir(filepath + directory):
            imagepath = filepath + directory + "//" + image
            find_centers_and_crop(imagepath, directory,
                                  image, outpath, outfile)
