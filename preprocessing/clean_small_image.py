import os
from PIL import Image

main_folder = './crops/'

#count = 0

for folder_name in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder_name)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            with Image.open(file_path) as img:
                width, height = img.size
        except:
            print(f"Error reading image: {file_path}")
            continue
        
        if min(width, height) < 50:
            os.remove(file_path)
            #count += 1
            print(f"Deleted {file_path}")
    
#print("Delete ", count, " imagaes!")
