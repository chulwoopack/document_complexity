"""Load Modules"""
import os
os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'
import cv2
from PIL import Image
from glob import glob
import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm

from imageio import imread, imsave
import tensorflow.contrib.slim as slim
from dh_segment.io import PAGE
from dh_segment.inference import LoadedModel
from dh_segment.post_processing import boxes_detection, binarization

import argparse
import csv

############
# argparse #
############
parser = argparse.ArgumentParser(description='Analyze document image complexity based on the segmentation result.')
parser.add_argument('--image_list', type=str, required=True)
parser.add_argument('--output_dir', type=str, default='./outputs/')
parser.add_argument('--output_prefix', type=str, required=True)
parser.add_argument('--index_start', type=int, default=-1)
parser.add_argument('--index_end', type=int, default=-1)
args = parser.parse_args()

IMAGE_LIST_FILE = args.image_list
OUTPUT_PREFIX   = args.output_prefix
INDEX_START     = args.index_start
INDEX_END       = args.index_end
OUTPUT_DIR      = args.output_dir
OUTPUT_CSV_NAME = OUTPUT_PREFIX + '_' + str(INDEX_START) + '_' + str(INDEX_END) + '.csv'

BG_ID     = 0
TEXT_ID   = 1
FIGURE_ID = 2
LINE_ID   = 3
TABLE_ID  = 4
CONNECTIVITY  = 4
SMALL_ZONE_THRESHOLD = 0.0001 # 0.1% of the input size

with open(IMAGE_LIST_FILE, 'rb') as f:
    image_lists = np.load(f)
    image_lists = image_lists.tolist()
print("Total {} images are found.".format(len(image_lists)))
if INDEX_START==-1: INDEX_START=0
if INDEX_END==-1: INDEX_END=len(image_lists)
    
print("Processing image(s)from-to: [{}-{}].".format(INDEX_START,INDEX_END))
image_lists = image_lists[INDEX_START:INDEX_END]


"""Start session"""
sess = tf.InteractiveSession()

"""Load model"""
#model_dir = './pretrained_model/model_oliveira/'
model_dir = './pretrained_model/model_ENP/export/1564895133'
#m = LoadedModel(model_dir, predict_mode='filename')
m = LoadedModel(model_dir, predict_mode='image')

"""Collect deep representation"""
deep_reps = []
for image_list in tqdm(image_lists):
    numpy_image = None
    filename, file_extension = os.path.splitext(image_list)
    # Read an image: make sure 
    # case 1. tif
    if(file_extension == "tif"):
        tif_image   = Image.open(image_list)
        numpy_image = np.asarray(tif_image)
        if(len(numpy_image.shape)==2):
            numpy_image = np.stack((numpy_image,)*3, axis=-1)
    # case 2. jp2, jpg, png, etc.
    else:
        numpy_image = cv2.imread(image_list)

    # Produce a binary image for additional analysis
    numpy_image_gray = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)
    (T, threshInv) = cv2.threshold(numpy_image_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    background_pixel = np.sum(threshInv//255)
    # (1) image original size
    ori_h, ori_w = threshInv.shape
    total_pixel = ori_h * ori_w
    # (2) image pixel-density
    pixel_density = (total_pixel-background_pixel)/total_pixel
    
    # Run segmentation
    prediction_outputs = m.predict(numpy_image)
    pred_labels = np.copy(prediction_outputs['labels'][0]).astype(np.uint8)
    
    # Consider textual region only
    mask_texts   = np.copy(pred_labels)
    mask_texts[mask_texts != TEXT_ID] = 0
    
    # Collect attributes
    txt_num_labels, txt_labels, txt_stats, txt_centroids = cv2.connectedComponentsWithStats(mask_texts, CONNECTIVITY, cv2.CV_32S)
    # (3) number of total zones
    num_tot_zones = txt_num_labels
    # txt_stats contain the following: left, top, width, height, and area of connected components
    zone_areas = txt_stats[:,4]
    # (4) number of small zones
    num_small_zones = 0
    for zone in zone_areas:
        if zone < ori_h*ori_w*SMALL_ZONE_THRESHOLD:
            num_small_zones += 1 
    # (5) number of final zones
    num_fin_zones = num_tot_zones - num_small_zones
    
    res = [image_list, num_tot_zones, num_small_zones, num_fin_zones, ori_h, ori_w, pixel_density]
    
    # Save results
    # (a) .csv file will contain metadata
    with open(os.path.join(OUTPUT_DIR,OUTPUT_CSV_NAME), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(res)
    # (b) .npy file will contain actual coordinate info
    OUTPUT_NPY_NAME = os.path.basename(filename) + '.npy'
    with open(os.path.join(OUTPUT_DIR,OUTPUT_NPY_NAME), 'wb') as f:
        np.save(f, txt_stats)
    
print('Done.')