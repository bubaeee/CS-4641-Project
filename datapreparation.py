from pandas import *
from PIL import Image
import os
from pathlib import Path
import wget
import cv2
import io

directory = "Images"

path = os.path.join(Path.home(), "/CS-4641-Project/Images")
# print(path)
# getting image column from training dataset
data = read_csv("faceexp-comparison-data-train-public.csv", on_bad_lines='skip')
image_column = data[data.columns[0]].to_numpy()
# remove duplicates
image_column = set(image_column.tolist())
# print(len(image_column))
for i in range(len(image_column)):
    # setting filename and image URL
    filename = f'train_image{i}.jpg'
    image_url = image_column[i]
    # calling urlretrieve function to get resource
    try:
        image = wget.download(image_url)
    except:
        pass
    
    # print(image)
    # image = Image.open(io.StringIO(image.content))
    # save image in filename
    # image = image.save(f"{image}/{filename}")
    # image = cv2.imread(image)
    # cv2.imwrite(os.path.join(path,filename), image)
    # print("complete")
