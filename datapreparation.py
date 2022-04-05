from pandas import *
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path
import wget
import io
import numpy as np
import cv2

download_directory = "Download-images"
grayscale_directory = "Grayscale-images"
faces_directory = "Faces"

path_to_images = os.path.join(os.getcwd(), download_directory)
path_to_grayscale = os.path.join(os.getcwd(), grayscale_directory)
path_to_faces = os.path.join(os.getcwd(), faces_directory)

def downloadImages(num_to_try, full=False):
    # getting image column from training dataset
    data = read_csv("faceexp-comparison-data-test-public.csv", on_bad_lines='skip')
    #print(data.head())
    #data = read_csv("faceexp-first20.csv", on_bad_lines='skip')
    image_column = data[data.columns[0]].to_numpy()
    # remove duplicates
    image_column = set(image_column.tolist())

    num = len(image_column) if full else num_to_try
    for i in range(num):
        # setting filename and image URL
        filename = f'train_image{i}.jpg'
        image_url = list(image_column)[i]
        # calling urlretrieve function to get resource
        try:
            image = wget.download(image_url, out=path_to_images)
        except:
            print(f'wget failed to get image {i}')

def resize(scale_factor, resize_threshold=350):

    images = [file for file in os.listdir(path_to_images) if file.endswith(('jpeg', 'png', 'jpg'))]

    for image in images:
        try:
            image_path = os.path.join(path_to_images, image)

            img = cv2.imread(image_path)

            resized = img
            if (img.shape[1]>resize_threshold or img.shape[0] > resize_threshold):
                width = int(img.shape[1] * scale_factor / 100)
                height = int(img.shape[0] * scale_factor / 100)
                resized = cv2.resize(img, (width,height))
            #convert to grayscale
            gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            output_path = os.path.join(path_to_grayscale, "resized_graycsale_"+image)
            cv2.imwrite(output_path, gray_img)
        except:
            print('error resizing image: ', image)

def facialDetection():
    count = 0
    images = [file for file in os.listdir(path_to_grayscale) if file.endswith(('jpeg', 'png', 'jpg'))]
    for image in images:
        try:
            image_path = os.path.join(path_to_grayscale, image)
            count += detectFace(image_path)
        except:
            print('error detecting face: ', image)

    print(f'detection rate: {count/len(images)}')

def detectFace(image_path):
    #load image and convert to grayscale

    image = cv2.imread(image_path)

    image_name = image_path.split('/')[-1]
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #detect face in image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    #draw rectangle around face
    if len(faces)==0:
        print('no face detected')
        return 0
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    #save image with rectangle drawn to file
    cv2.imwrite(os.path.join(path_to_faces, image_name), image)
    return 1

if __name__ == '__main__':
    #downloadImages(10)
    #resize(50)
    facialDetection()
    