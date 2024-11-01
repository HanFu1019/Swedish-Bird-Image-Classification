import ultralytics
from ultralytics import YOLO
from PIL import Image
import cv2
import os

ultralytics.checks()

model = YOLO('yolov8n.pt')

'''
**********************************************
Change the dataset path
'''
parent_folder = '../Dataset/SWEDEN/'

subfolders = [f.name for f in os.scandir(parent_folder) if f.is_dir()]

#for folder in subfolders:
#    print(folder)

print("Total species: ", len(subfolders))


'''
**********************************************
Change the output_folder path which saves the bounding box .txt file
'''

for folder in subfolders:
    output_folder = './runs/detect'
    output_folder = os.path.join(output_folder, folder)
    if os.path.exists(output_folder):
        print(output_folder, " : Done!")
        continue
    results = model.predict(source=parent_folder+folder, save_txt=True, max_det=1, classes=14, name=folder)
  
