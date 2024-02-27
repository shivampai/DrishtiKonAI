import cv2
from picamera2 import Picamera2
import os
import pyttsx3

cam = Picamera2()

cam.preview_configuration.main.size=(1280,720)
cam.preview_configuration.main.format = "RGB888"
cam.preview_configuration.align()
cam.configure("preview")
cam.start()

thres = 0.45 # Threshold to detect object


#Edit in Coco to arr(USB Drive File)
classNames= ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","street sign","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","hat","backpack","umbrella","shoe","eye glasses","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","plate","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","mirror","dining table","window","desk","toilet","door","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","blender","book","clock","vase","scissors","teddy bear","hair drier","toothbrush","hair brush"]
#

configPath = '/home/Shivam/Desktop/DrishtiKonAI/obf/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = '/home/Shivam/Desktop/DrishtiKonAI/obf/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def tts(txt):
    tts = pyttsx3.init()
    tts.setProperty('rate',300)
    tts.setProperty('volume',0.9)
    tts.say(txt)
    tts.runAndWait()
while True:
    img = cam.capture_array()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            tts(classNames[classId-1] + " detected")

    cv2.imshow('Output',img)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
