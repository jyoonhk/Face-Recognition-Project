import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import streamlit as st
from keras.models import model_from_json
from keras.preprocessing import image

#Load cascade files
face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('Cascades/haarcascade_smile.xml')
# profile_cascade = cv2.CascadeClassifier('Cascades/haarcascade_profileface.xml')

# initiating variables so streamlit doesn't give errors
name = None 
age = None
position = None

def make_new_ID(name, age, position):
    with open('Database/Student_database.csv','r+') as f:
        existing_info = f.readlines()
        nameList = []
        for line in existing_info:
            entry = line.lower().split(',')
            nameList.append(entry[0])
        if name not in nameList:           
            f.writelines(f'\n{name},{age},{position}')
    return None

def video_capture(n):
    cap = cv2.VideoCapture(0)
    count = 0
    while count<1:
        ret, img = cap.read()
        st.image(img[:,:,::-1])
        user_image = np.array(img)
        cap.release()
        count+=1
    return user_image

def read_dir(path): 
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    return images, classNames
 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
def markAttendance(name):
    with open('Database/Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.lower().split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
 
#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

###################################################### 

def check_ident(cap,encodeListKnown, classNames, run_attendee):
    while run_attendee:
        success, img = cap.read()
        #img = captureScreen()
        imgS = cv2.resize(img,(0,0),None,0.25,0.25) # one fourth of the size
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
        facesCurFrame = face_recognition.face_locations(imgS) # find location of all faces in small image
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame) 
    
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)  # return the position of element of lowest faceDistances

            if np.amin(faceDis)>0.6:
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
                cv2.putText(img,'UNAUTHORISED USER',(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

            elif matches[matchIndex]:
                name = classNames[matchIndex].upper()
                #print(name)
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                markAttendance(name)

        return img[:,:,::-1]
    cap.release()
    cv2.destroyAllWindows()
    
#Function to take and save picture
def take_picture(n):
    cap = cv2.VideoCapture(0)
    count = 0
    while count<2:
        ret, img = cap.read()
        st.image(img[:,:,::-1])
        user_image = np.array(img)
        cap.release()
        path = 'ImagesAttendance'
        cv2.imwrite(os.path.join(path , name +'.jpg'), user_image)
        return user_image


def emotion_recog(cap, model, run_emotion):
    while run_emotion:
        ret,test_img=cap.read()# captures frame and returns boolean value and captured image
        if not ret:
            continue
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x,y,w,h) in faces_detected:
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
            roi_gray=cv2.resize(roi_gray,(48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            #find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  
            predicted_emotion = emotions[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        return test_img[:,:,::-1]