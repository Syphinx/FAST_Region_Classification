import cv2
import glob
import logging
import matplotlib.pyplot as plt
from medpy.io import load
import numpy
import os
from PIL import Image
import random
import re
from scipy import ndimage
import sys


# Split a dataset into train, val, test sets
SKILL_LABELS = ["Experts", "Intermediates", "Novices", "ReturnedNovices"]
SCAN_LABELS = ["Scan01", "Scan02", "Scan03", "Scan04"]
IMAGE_EXTENSION = "mha" # Name of the image extension
OUTPUT_IMAGE_SIZE = [224, 224]


def write_slices(image_filename, write_directory):
  image_data, image_header = load(image_filename)
  image_data = image_data[170:788, 150:624, :] # There relevant area of the ultrasound screen capture
  zoom_factor = [OUTPUT_IMAGE_SIZE[0] / image_data.shape[0], OUTPUT_IMAGE_SIZE[1] / image_data.shape[1], 1]
  image_data = ndimage.zoom(image_data, zoom_factor, order=0)
  scan = get_scan(image_filename)
  image_write_subdirectory = get_skill(image_filename) + "_" +  os.path.basename(image_filename)
  
  for i in range(image_data.shape[2]): # z-axis
    os.makedirs(os.path.join(write_directory, scan, image_write_subdirectory), exist_ok=True)
    if (image_data == image_data[0]).all():
      continue # don't do anything if it is a slice with no intensity

    jpg_image_data = create_jpg_image_data(image_data[:, :, i])
    jpg_image = Image.fromarray(jpg_image_data, mode="RGB")
    jpg_image_filename = os.path.join(write_directory, scan, image_write_subdirectory, "img_" + str(i).zfill(5) + ".jpg")
    jpg_image.save(jpg_image_filename)

    # Optical flow
    curr_umat_data = cv2.UMat(image_data[:, :, i].astype(numpy.uint8))
    if i == 0:
      prev_umat_data = cv2.UMat(image_data[:, :, i].astype(numpy.uint8))
    else:
      prev_umat_data = cv2.UMat(image_data[:, :, i - 1].astype(numpy.uint8))

    flow_computer = cv2.optflow.DualTVL1OpticalFlow_create()
    flow_data = flow_computer.calc(prev_umat_data, curr_umat_data, None).get()

    jpg_flowx_data = create_jpg_image_data(flow_data[:, :, 0])
    jpg_flowx = Image.fromarray(jpg_flowx_data, mode="RGB")
    jpg_flowx_filename = os.path.join(write_directory, scan, image_write_subdirectory, "flow_x_" + str(i).zfill(5) + ".jpg")
    jpg_flowx.save(jpg_flowx_filename)

    jpg_flowy_data = create_jpg_image_data(flow_data[:, :, 1])
    jpg_flowy = Image.fromarray(jpg_flowy_data, mode="RGB")
    jpg_flowy_filename = os.path.join(write_directory, scan, image_write_subdirectory, "flow_y_" + str(i).zfill(5) + ".jpg")
    jpg_flowy.save(jpg_flowy_filename)




def create_jpg_image_data(image_data):
  # Normalize onto 255
  jpg_image_data = image_data
  jpg_image_data = jpg_image_data - numpy.min(jpg_image_data)
  jpg_image_data = (255 / numpy.max(jpg_image_data)) * jpg_image_data 
  jpg_image_data = jpg_image_data.astype(numpy.uint8)

  # Force greyscale to three channels
  jpg_image_data = numpy.tile(numpy.expand_dims(jpg_image_data, axis=-1), (1, 1, 3))

  return jpg_image_data


def get_participant_names(data_directory):
  participant_names = dict()
  for skill in SKILL_LABELS:
    participant_names[skill] = []
    image_filenames = glob.glob(os.path.join(data_directory, skill, "**", "*." + IMAGE_EXTENSION), recursive=True)
    for curr_image_filename in image_filenames:
      curr_participant_name = get_participant_name(curr_image_filename)
      participant_names[skill].append(curr_participant_name)
    participant_names[skill] = list(set(participant_names[skill])) # Remove any duplicates
    print(participant_names[skill])

  return participant_names


def get_participant_name(image_filename):
  participant_name = os.path.basename(image_filename)
  
  participant_name = participant_name.replace("." + IMAGE_EXTENSION, "")
  for scan in SCAN_LABELS:
    participant_name, __, __ = participant_name.partition(scan)

  return participant_name


def get_scan(image_filename):
  for scan in SCAN_LABELS:
    if scan in image_filename:
      return scan
  return ""


def get_skill(image_filename):
  for skill in SKILL_LABELS:
    if skill in image_filename:
      return skill
  return ""


def get_image_filenames(data_directory, participant_name, participant_skill):
  image_filenames = glob.glob(os.path.join(data_directory, participant_skill, "**", "*" + participant_name + "*." + IMAGE_EXTENSION), recursive=True)
  return image_filenames


def write_slice_sets(data_directory, write_directory, test_ratio):
  # Get the participant names and allocate them into train/val/test sets
  participant_names = get_participant_names(data_directory)

  # Shuffle everything
  for skill in participant_names.keys():
    random.shuffle(participant_names[skill])

  # Want to try to get an equal proportion of each skill level in each set
  train_set = {key: [] for key in participant_names.keys()}
  val_set = {key: [] for key in participant_names.keys()}
  test_set = {key: [] for key in participant_names.keys()}
  for skill in participant_names.keys():
    for curr_participant_name in participant_names[skill]:
      if len(test_set[skill]) <= test_ratio * (len(val_set[skill]) + len(train_set[skill])):
        test_set[skill].append(curr_participant_name)
      elif len(val_set[skill]) <= test_ratio * (len(test_set[skill]) + len(train_set[skill])):
        val_set[skill].append(curr_participant_name)
      else:
        train_set[skill].append(curr_participant_name)

  # Write the slices to file
  print("Train set:")
  for participant_skill in train_set:
    for participant_name in train_set[participant_skill]:
      image_filenames = get_image_filenames(data_directory, participant_name, participant_skill)
      for curr_image_filename in image_filenames:
        print(curr_image_filename)
        write_slices(curr_image_filename, os.path.join(write_directory, "train"))

  print("Val set:")
  for participant_skill in val_set:
    for participant_name in val_set[participant_skill]:
      image_filenames = get_image_filenames(data_directory, participant_name, participant_skill)
      for curr_image_filename in image_filenames:
        print(curr_image_filename)
        write_slices(curr_image_filename, os.path.join(write_directory, "val"))

  print("Test set:")
  for participant_skill in test_set:
    for participant_name in test_set[participant_skill]:
      image_filenames = get_image_filenames(data_directory, participant_name, participant_skill)
      for curr_image_filename in image_filenames:
        print(curr_image_filename)
        write_slices(curr_image_filename, os.path.join(write_directory, "test"))


if __name__ == "__main__":
  data_directory = str(sys.argv[1])
  write_directory = str(sys.argv[2])
  test_ratio = float(sys.argv[3])

  write_slice_sets(data_directory, write_directory, test_ratio)
