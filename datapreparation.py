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
resize_directory = "Resized-images"
faces_directory = "Faces"

path_to_images = os.path.join(os.getcwd(), download_directory)
path_to_resize = os.path.join(os.getcwd(), resize_directory)
path_to_faces = os.path.join(os.getcwd(), faces_directory)

def downloadImages(num_to_try, full=False):
    '''
    Downloads the images in the training dataset.
    Parameters: 
        num_to_try (int): number of images to download
        full (boolean): boolean flag to download all the images training dataset
    '''
    # getting image column from training dataset
    data = read_csv("faceexp-comparison-data-test-public.csv", on_bad_lines='skip')
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
            raise Exception(f'wget failed to get image {i}')

def resize(scale_factor, resize_threshold=350):
    '''
    Resizes all images in the image directory as part of the preprocessing steps.
    Parameters:
        scale_factor (int): the scale factor for resizing
        resize_threshold (int): the threshold value which determines which images are resized 
    '''
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
            output_path = os.path.join(path_to_resize, "resized_"+image)
            cv2.imwrite(output_path, resized)
        except:
            raise Exception('error resizing image: ', image)

def facialDetection():
    '''
    Prints the detection rate of the images in our directory.
    '''
    count = 0
    images = [file for file in os.listdir(path_to_resize) if file.endswith(('jpeg', 'png', 'jpg'))]
    for image in images:
        try:
            image_path = os.path.join(path_to_resize, image)
            count += detectFace(image_path)
        except:
            raise Exception('error detecting face: ', image)

    print(f'detection rate: {count/len(images)}')

def detectFace(image_path):
    ''' 
    Utility function that performs the facial detection using OpenCV
    Parameters:
        image_path (str): path to image which has faces to be detected.
    Returns:
        detected (int): flag that lets us know how many faces are successfully detected
    '''
    #load image and convert to grayscale

    image = cv2.imread(image_path)

    image_name = image_path.split('/')[-1]
    detected = 0
    #convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #detect face in image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)
    #draw rectangle around face
    if len(faces)==0:
        print('no face detected')
        return detected
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    #save image with rectangle drawn to file
    cv2.imwrite(os.path.join(path_to_faces, image_name), image)
    detected = 1
    return detected

if __name__ == '__main__':
    # only call download images when you want to download more training images
    # downloadImages(10) 
    resize(50)
    facialDetection()
    
    