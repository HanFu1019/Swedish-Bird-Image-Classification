import os
from PIL import Image

'''
**********************************************************
Change
runs_folder: results of bounding box .txt files from yolo
output_folder: where to save cropped images
image_folder: path to original images
'''

runs_folder = 'C:/Users/43477/Desktop/Yolo_Crop/runs/'
detect_folder = os.path.join(runs_folder, 'detect')
output_folder = 'C:/Users/43477/Desktop/Yolo_Crop/crops/'
image_folder = 'C:/Users/43477/Desktop/SWEDEN/'


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


for folder_name in os.listdir(detect_folder):
    folder_path = os.path.join(detect_folder, folder_name)
    current_image_folder = os.path.join(image_folder, folder_name)
    current_output_folder = os.path.join(output_folder, folder_name)

    if not os.path.exists(current_output_folder):
      os.makedirs(current_output_folder)


    labels_folder = os.path.join(folder_path, 'labels')
    #if not os.path.exists(labels_folder):
        #continue


    for txt_file in os.listdir(labels_folder):
        txt_file_path = os.path.join(labels_folder, txt_file)
        current_image_path = os.path.join(current_image_folder, txt_file)


        with open(txt_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # [class] [x_center] [y_center] [width] [height]
                data = line.strip().split(' ')

                image_path = current_image_path[:-4] + '.jpg'
                image = Image.open(image_path)
                image = image.convert('RGB')

                #class_label = data[0]
                x_center = float(data[1])* image.width
                y_center = float(data[2])* image.height
                width = float(data[3])* image.width
                height = float(data[4])* image.height




                sideLength = max(width, height)

                left = max(0, int(x_center - sideLength / 2))
                top = max(0, int(y_center - sideLength / 2))
                right = min(image.width, int(x_center + sideLength / 2))
                bottom = min(image.height, int(y_center + sideLength / 2))



                cropped_image = image.crop((left, top, right, bottom))


                output_file_path = os.path.join(current_output_folder, f'{txt_file[:-4]}.jpg')
                cropped_image.save(output_file_path)
    print(folder_name," : Done!")