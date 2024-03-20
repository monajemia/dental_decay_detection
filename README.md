# Dental Decay Object Detection
This project trains a Faster R-CNN model to detect dental decay from bitewing images used in dentistry.  
  
## Scripts Purpose  
-- main.py  
---- Contains dataset class, dataloader, model, training, etc.  
-- convert_data_to_coco.py  
---- Converts out custom data from .nrrd (3dslicer output) to coco style JPEG and JSON. It is probably best that you create this part on your own. In the end, the main script expects COCO style data.  

## Notes
* This is my first real project in object detection using deep learning
* I have manually calculated TP, FP, FN, TN, Precision, Recall, ..., this you may find interesting.
