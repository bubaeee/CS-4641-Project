from pandas import *
from PIL import Image
import os
from pathlib import Path
import wget
import io
import numpy as np
import cv2

directory = "Images"

path = os.path.join(Path.home(), "/CS-4641-Project/Images")
# print(path)
# getting image column from training dataset
data = read_csv("faceexp-comparison-data-test-public.csv", on_bad_lines='skip')
image_column = data[data.columns[0]].to_numpy()
# remove duplicates
image_column = set(image_column.tolist())
# print(len(image_column))

for i in range(len(image_column)):
    # setting filename and image URL
    filename = f'train_image{i}.jpg'
    image_url = list(image_column)[i]
    # calling urlretrieve function to get resource
    try:
        image = wget.download(image_url)
    except:
        pass


    #print(image)
    #image = Image.open(io.StringIO(image.content))
    #save image in filename
    #image = image.save(f"{image}/{filename}")
    # image = cv2.imread(image)
    # cv2.imwrite(os.path.join(path,filename), image)
    # print("complete")

#resize images
images = [file for file in os.listdir() if file.endswith(('jpeg', 'png', 'jpg'))]
for image in images:
    try:
        img = Image.open(image)
        #need to convert to RGB format for resize
        img = img.convert('RGB')
        img.thumbnail((200,200))
        img.save("resized_"+image, optimize=True, quality=40)
    except:
        pass