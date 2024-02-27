import cv2
from picamera2 import Picamera2
import os
import pyttsx3
import RPi.GPIO as GPIO

cam = Picamera2()

cam.preview_configuration.main.size=(1280,720)
cam.preview_configuration.main.format = "RGB888"
cam.preview_configuration.align()
cam.configure("preview")
cam.start()


#IR SETUP
GPIO.setmode(GPIO.BOARD)
GPIO.setup(3,GPIO.IN)
#START OF MAIN CODE

thres = 0.45 # Threshold to detect object


#Edit in Coco to arr(USB Drive File)
classNames = ["person","","","","","","","","","","","","stop sign","","","bird","cat","","","","","","","","","","backpack","","shoe","eye glasses","","","","","","","","","","","","","","bottle","","","","","","","","banana","apple","","","","","","","","","","","potted plant","","","","","","","","tv","laptop","mouse","remote","keyboard","cell phone","","","","","","","book","","vase","","teddy bear","","",""]
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
    tts.setProperty('rate',200)
    tts.setProperty('volume',0.9)
    tts.say(txt)
    tts.runAndWait()
tts("Device on")
while True:
    irStatus = GPIO.input(3)
    if irStatus == 1:
        img = cam.capture_array()
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
        print(classIds)
        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                if classNames[classId-1] == "":
                    print("")
                else:
                    if classId == 1:
                        cName = 'person'
                    else:
                        cName = classNames[classId-1]
                        
                    conf = round(confidence*100,2)
                    if conf > 50:
                        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                        cv2.putText(img,cName.upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        cv2.putText(img,str(conf),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        tts(cName + " detected")
        cv2.imshow('Output',img)
        if cv2.waitKey(100) == ord('q'):
            break
    else:
        tts("Stop")
cv2.destroyAllWindows()
