import streamlit as st
import cv2 
import app
import app2
import os
import face_recognition
import numpy as np 
import pandas as pd 
from PIL import Image,ImageEnhance
import time
from keras.models import model_from_json
from keras.preprocessing import image

path = 'ImagesAttendance'

#Run to store cached data
@st.cache #method to get data once and store in cache.
def get_images_classNames():
    images, classNames = app.read_dir(path)
    return images, classNames
images, classNames = get_images_classNames()

@st.cache #method to get data once and store in cache.
def get_encodeListKnown(images):
    return app.findEncodings(images)
encodeListKnown = get_encodeListKnown(images)

#Creating Table of Content class
class Toc:

    def __init__(self):
        self._items = []
        self._placeholder = None
    
    def title(self, text):
        self._markdown(text, "h1")

    def header(self, text):
        self._markdown(text, "h2", " " * 2)

    def subheader(self, text):
        self._markdown(text, "h3", " " * 4)

    def placeholder(self, sidebar=True):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)
    
    def _markdown(self, text, level, space=""):
        key = "".join(filter(str.isalnum, text)).lower()

        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")


toc = Toc()

#replacing st with toc for relevant title/headers
toc.placeholder()

toc.title('Facial Recognition Project')
st.write('By Yoon, Edward, Daniel, and Danny')
st.image('Images/banner1.jpg', channels = 'BGR')
toc.header('Introduction')
st.write('''
Facial Recognition is a technology that can identify/verify elements of a human face by collecting a set of biometric data of each person associated with their face. 

The objective of facial recognition models is to identify a face from an image, and verify this face against a set of known faces.

The uses of facial recognition technology include:
- Security & Law Enforcement: criminal databases, validating identity at ATMs
- Mobile device security: Unlocking phones, authorising transactions
- Smart advertisement: Targeted adverts based on people's age and gender
- Tracking attendance: In person and online (Zoom)
''')

### BUSINESS CASE
toc.header('Business Case')
st.write('''
Our Facial Recognition products aim to provide benefits to businesses, by providing tools for:
- Enhanced Security: Protection against unauthorised visitors/users
- Employee Welfare: Detection of aggressive behaviour and physical threats against employees, and employee well-being detection.
- Sales: Smart Advertising techniques tailored to the user's age/gender

An example database with user images was created and facial recognition models were trained to recognise these authorised users. Features such as emotion detection and gender/age detection have also been implemented.
''')

### Model Overview
toc.header('Model Overview')
st.write('''
3 facial recognition models were tested and implemented using external data and our user data. The models are: User Detection, Emotion Detection, and Gender/Age Detection.

These are a mixture of pretrained models e.g. OpenCV and face_recognition libraries, and a Deep Learning CNN model we have trained.

Each model follows a similar process: 
- Detect a face from an image
- Process this face through a feature model (e.g. user name, emotions, gender/age)
- Overlay the model outputs on the original image

More details are given below.
''')
st.image('Images/process1.jpg')

## Add NEW USER
toc.header('User Database')
st.write('''
For Security Management, all users are required to have an entry in the User Database. Facial Recognition will be used to ensure that only authorised users can access the building premises.

To begin, please input a new user\'s information below
''')

if st.checkbox('Tick here if you are a new user.'):
    st.subheader('Please input your information:')
    name = st.text_input('Name')
    age = st.text_input('Age')
    position = st.text_input('Position')
    
    if st.button('Upload information'):  
        app.make_new_ID(name,age,position)
        st.text('Information saved')

    # Saving New image
    st.subheader('Upload your image')
    select = st.selectbox('Select how to load image:', ['Load Image', 'Webcam'])

    if select == 'Load Image':
        user_image = st.file_uploader("Please upload an image of your face",type=['jpg','png','jpeg'])
        if user_image is not None:
            img = Image.open(user_image)
            if st.button('Save Picture'):
                path = 'ImagesAttendance'
                with open(os.path.join(path , name +'.jpg'),"wb") as f:
                    f.write(user_image.getbuffer())
                st.text('Image saved!')

    if select == 'Webcam':
        if st.button('Take Picture'):
            user_image = app.video_capture(1)
            path = 'ImagesAttendance'
            cv2.imwrite(os.path.join(path , name+'.jpg'), user_image)
            st.text('Image saved!')

# Run functions to load known users / images in database  
# images, classNames = app.read_dir(path)
# encodeListKnown = app.findEncodings(images)

#Display user images stored in database
toc.subheader('Current users in database')
resized = [cv2.resize(image, (140,140)) for image in images]
st.image(resized, channels = 'BGR')

toc.title('Live examples')

#ATTENDEE DETECTOR
toc.header("1. User Detector")
st.write('Purpose: Security & Law Enforcement, Mobile device security, Tracking attendance.')
st.image('Images/detector1.jpg', channels = 'BGR')
st.write('''
A pretrained facial recognition model from the OpenCV face_recognition library was used to detect users from the images uploaded to our internal database. 

This model will identify unauthorised faces that differ significantly from users in our database. Users that are recognised by the model will be labelled in green and their movements logged for attendance tracking.
''')
toc.subheader('Test User Detector pretrained model')

run_attendee = st.checkbox('Start User Detector')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
while run_attendee:
    _, frame = camera.read()
    FRAME_WINDOW.image(app.check_ident(camera,encodeListKnown, classNames, run_attendee))
else:
    camera.release()
    cv2.destroyAllWindows()
    st.write('Stopped')

#EMOTION DETECTOR
toc.header("2. Emotion Detector")
st.write('Purpose: Security & Law Enforcement')
st.write('''
A Convolutional Neural Network was trained to detect facial emotions by using a Kaggle dataset with over 35,000 images of various facial emotions. 

The CNN model was trained to detect: Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral expressions.

Example images:
''')
st.image('Images/emotions2.jpg', channels = 'BGR')
st.write('Distribution of emotions in training dataset:')
st.image('Images/emotions1.jpg', channels = 'BGR')
st.write('Accuracy/Loss - Train v Test against Epoch')
st.image('Images/emotions3.jpg', channels = 'BGR')
toc.subheader('Test the Emotion Detector CNN model')

#load model and weights
model = model_from_json(open("EmotionRecCNNAnalysis/model.json", "r").read()) 
model.load_weights('EmotionRecCNNAnalysis/model.h5')

run_emotion = st.checkbox('Start Emotion Detector')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
while run_emotion:
    _, frame = camera.read()
    FRAME_WINDOW.image(app.emotion_recog(camera, model, run_emotion))
else:
    camera.release()
    cv2.destroyAllWindows()
    st.write('Stopped')

#GENDER AND AGE DETECTOR
toc.header("3. Gender and Age Detector")
st.write('Purpose: Smart Advertisement, Tracking attendance.')
st.write('''The model for age and gender was pre-trained from Adience data set with 26,580 photos, while the face detection model is from OpenCV's DNN module.
- The Gender has 2 classes: Male and Female
- Age is divided between 8 classes: 0 – 2, 4 – 6, 8 – 12, 15 – 20, 25 – 32,38 – 43, 48 – 53, 60 – 100
''')
st.image('Images/agegender1.jpg')
st.write('''The Distribution of Age/Gender''')
st.image('Images/agegender2.jpg')
toc.subheader('Test the Gender/Age Detector model')

run_genderage = st.checkbox('Start Age/Gender Detector')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
while run_genderage:
    hasframe,frame = camera.read()
    FRAME_WINDOW.image(app2.Age_Gender(camera, run_genderage))
else:
    camera.release()
    cv2.destroyAllWindows()
    st.write('Stopped')

toc.header('Conclusions')
st.write('''
This project has demonstrated some uses of facial recognition technology, within Security & Law Enforcement, attendance tracking and advertising. 

Further enhancements could include:

- Enriching the original datasets to better differentiate between similar users.
- Aggregating our 3 models into 1 model to simultaneously detect authorised users, emotions, and gender/age.
- Enhance model to differentiate between real people and pictures/images of people.

Concerns on Facial Recognition include:

- Privacy / Misuse of Facial Data: The technology could enable mass surveillance of all people; there is no widespread legislation of facial recognition technology.
- Security Breaches: Improper storage of user images could expose users to privacy breaches / security threats / stolen identity.
- Bias and Inaccuracies: Algorithms trained on racially biased datasets could lead to misidentifying and/or discrimination of people from minority ethnic backgrounds. 
''')

toc.generate()