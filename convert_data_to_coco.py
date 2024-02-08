#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: @rmnoa
"""

import os
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import imageio
import nrrd
import sys

# Change these
data_dir = sys.argv[1] # Directory containing mix of images and jsons
output_dir = sys.argv[2] # Should not exist before running
output_json_file = sys.argv[2] + '.json'
classes = ["RA1", "RA2", "RA3", "RB4", "RC5", "RC6"] # Labels
boxes_dir_name = 'BOXES' # Subdirectory of BOXES
width, height = 514, 660 # Dimensions of input images, after rotation to fix

# Check output_dir exist or not
if os.path.isdir(output_dir):
    output_dir_exists = True
else:
    output_dir_exists = False
    os.mkdir(output_dir)

boxes_jsons = {}
image_ids = []
# Extract filenames from directory
for pathname, dirnames, filenames in os.walk(data_dir):
    if boxes_dir_name in pathname:
        image_id = Path(pathname).parents[0].name
        image_ids.append(image_id)
        boxes_jsons[image_id] = filenames

output_json = {}
# Prepare the JSON
# Categories
output_json['categories'] = [ { '1': 'RA1' }, { '2': 'RA2' }, { '3': 'RA3' },
                             { '1': 'RA4' }, { '1': 'RA5' }, { '1': 'RA6' } ]

# Images and annotations list in JSON
images = []
annotations = []
for image_id in image_ids:
    for boxes_json in boxes_jsons[image_id]:
        # Get label from filename
        class_id = boxes_json.split('_')[0].split('.')[0]
        category_id = classes.index(class_id) + 1 # Models accept ids > 0
        # Parse JSON
        with open(os.path.join(data_dir, image_id, boxes_dir_name, boxes_json)) as f:
            boxes = json.load(f)
            
            # Calculate bounding box
            # 0.06 is for mm to pixel conversion
            center = [i/0.06 for i in boxes['markups'][0]['center']][0:2]
            sizes = [i/0.06 for i in boxes['markups'][0]['size']][0:2]
            
            x1, x2, y1, y2 = [ (center[0] - sizes[0] / 2),
                              (center[0] + sizes[0] / 2),
                              (center[1] - sizes[1] / 2),
                              (center[1] + sizes[1] / 2 ) ]
            bbox = x1, y1, x2, y2
            
            annotation = { 'image_id': image_id,
                          'category_id': category_id,
                          'bbox': bbox
                          }
            
            annotations.append(annotation)
    
    # Convert NRRD to JPG and move to output dir
    image_name = os.path.join(data_dir, image_id, image_id + '.nrrd')
    image_name_new = os.path.join(output_dir, image_id + '.jpg')
    
    # To prevent writing images on every run
    if not output_dir_exists:
        pixels = nrrd.read(image_name)[0]
        # Remove extra dim
        pixels = np.squeeze(pixels, axis=2)
        # Normalize to 0 - 255 range
        pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min()) * 255
        # Prepare for writing to JPEG (unit8 data type required)
        # Swapaxes is due to the images being rotated originally
        pixels = pixels.astype(np.uint8).swapaxes(-1, -2)
        # Output to file
        imageio.imwrite(image_name_new, pixels)
    
    date_captured = str(datetime.fromtimestamp(os.stat(image_name).st_mtime))
    
    # images JSON list
    images.append({ 'id': image_id,
                   'width': width,
                   'height': height,
                   'file_name': image_name_new,
                   'date_captured': date_captured
                   })

# Add to JSON
output_json['annotation'] = annotations
output_json['images'] = images

# Write to JSON file
with open(output_json_file, 'w') as f:
    json.dump(output_json, f)