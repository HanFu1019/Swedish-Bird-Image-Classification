import os
from PIL import Image

def check_images_exist(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # 可能的图片扩展名
    missing_images = []  # 存储丢失的图片文件名

    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            _, file_extension = os.path.splitext(file_name)
            if file_extension.lower() in image_extensions:
                try:
                    Image.open(os.path.join(folder_path, file_name))
                except (IOError, OSError):
                    missing_images.append(file_name)

    if missing_images:
        print("以下图片文件不存在：")
        for missing_image in missing_images:
            print(missing_image)
    else:
        print("所有图片文件都存在！")

# 要检查的文件夹路径
folder_path = "C:/Users/43477/Desktop/Antarctica_Dataset/Ardenna tenuirostris"
check_images_exist(folder_path)
