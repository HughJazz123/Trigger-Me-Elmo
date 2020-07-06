import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import turtle as t
import glob
import random
import os
import pyaudio
import wave
import audioop
#-----------------------

color = (67,67,67) #gray

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
chunk = 1024
p=pyaudio.PyAudio()

def loadVggFaceModel():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    
    return vgg_face_descriptor

#------------------------

model = loadVggFaceModel()

base_model_output = Sequential()
base_model_output = Convolution2D(6, (1, 1), name='predictions')(model.layers[-4].output)
base_model_output = Flatten()(base_model_output)
base_model_output = Activation('softmax')(base_model_output)

race_model = Model(inputs=model.input, outputs=base_model_output)

#pre-trained race and ethnicity prediction model weights can be found here: https://drive.google.com/file/d/1nz-WDhghGQBC4biwShQ9kYjvQMpO6smj/view?usp=sharing
race_model.load_weights('weights/race_model_single_batch.h5')

#------------------------

races = ['Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Hispanic']

#------------------------

cap = cv2.VideoCapture(0) #webcam

#turtle stuff
t.hideturtle()
t.bgcolor('black')
t.setup(width=1.0,height=1.0)
t.bgpic('elmo2.png')
t.update()
#

asian_directory = glob.glob('Sound/Asian/*.wav')
black_directory = glob.glob('Sound/Black/*.wav')
hispanic_directory = glob.glob("Sound/Hispanic/*.wav")
white_directory = glob.glob('Sound/White/*.wav')
other = glob.glob("Sound/Other/*.wav")
speech_copy = ''
count = 0

while(True):
    ret, img = cap.read()
    img = cv2.resize(img, (int(640), int(360)))
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    for (x,y,w,h) in faces:
        if w > 130: 
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1) #draw rectangle to main image
            
            detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
            
            #add margin
            margin_rate = 30
            try:
                margin_x = int(w * margin_rate / 100)
                margin_y = int(h * margin_rate / 100)
                
                detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
                detected_face = cv2.resize(detected_face, (224, 224)) #resize to 224x224

                #display margin added face
                #cv2.rectangle(img,(x-margin_x,y-margin_y),(x+w+margin_x,y+h+margin_y),(67, 67, 67),1)
                
            except Exception as err:
                #print("margin cannot be added (",str(err),")")
                detected_face = img[int(y):int(y+h), int(x):int(x+w)]
                detected_face = cv2.resize(detected_face, (224, 224))
            
            #print("shape: ",detected_face.shape)
            
            if detected_face.shape[0] > 0 and detected_face.shape[1] > 0 and detected_face.shape[2] >0: #sometimes shape becomes (264, 0, 3)
                
                
                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255
                
                prediction_proba = race_model.predict(img_pixels)
                prediction = np.argmax(prediction_proba)
                
                if False: # activate to dump
                    for i in range(0, len(races)):
                        if np.argmax(prediction_proba) == i:
                            print("* ", end='')
                        print(races[i], ": ", prediction_proba[0][i])
                    print("----------------")
                
                race = races[prediction]
                #--------------------------
                #background
                overlay = img.copy()
                color = (255,255,255)
                proba = round(100*prediction_proba[0, prediction], 2)
                if proba >= 51:
                    label = str(race)
                    if count < 10:
                        count+=1
                    elif count == 10:
                        count+=1
                        cv2.putText(img, label, (x+w//2-10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    else:
                        count = 0
                        if label == "Asian" or label == "Indian":directory = asian_directory
                        elif label == "Black":directory = black_directory
                        elif label == "White" or label == "Middle Eastern":directory = white_directory
                        elif label == "Hispanic":directory = hispanic_directory
                        else:directory = other
                        speech = random.choice(directory)
                        if speech == speech_copy and len(directory)>=2:
                            continue
                        speech_copy = speech
                        wf = wave.open(speech,'rb')
                        stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                                        channels = wf.getnchannels(),
                                        rate = wf.getframerate(),
                                        output = True)
                        data = wf.readframes(chunk)
                        while data != b'':
                            stream.write(data)
                            data = wf.readframes(chunk)
                            rms= audioop.rms(data,2)
                            if rms>500:
                                t.bgpic('elmo.png')
                                t.update()
                            else:
                                t.bgpic('elmo2.png')
                                t.update()
    cv2.imshow('img',img)
    
    if cv2.waitKey(1) & 0xFF == 27: #press q to quit
        break

#kill open cv things        
cap.release()
cv2.destroyAllWindows()
t.bye()
stream.close()
p.terminate()
