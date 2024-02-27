import tensorflow.keras
import cv2
import numpy as np
import time

np.set_printoptions(suppres=True)

model = tensorflow.keras.models.load_model("tm/keras_model.h5")

with open('tm/labels.txt','r') as f:
    class_names = f.read().split('\n')

data = np.ndarray(shape=(1,244,244,3),dtype=np.float32)
size = (224,224)

cap = cam.capture_array()

while cap.isOpened():
    start = time.time()
    ret,img = cap
    
    height,width,channels = img.shape
    
    scale_value = width/height
    
    img_resized = cv2.resize(img,size,fx=scale_value,fy=1,inter
