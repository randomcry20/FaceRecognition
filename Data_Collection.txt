#!/usr/bin/env python
# coding: utf-8
# # Dataset Collection

#Collecting Training dataset for model!! 
import cv2
import os
# Loading HaarCascade Model to detect frontal face!! 
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
count=0
path="C:/Users/HP/Desktop/MLOps/Face/train/Naman" # Path Where You'll save images for training of your model.
def CheckFaces(sample):
    if sample is ():
        return None
    else:
        return sample

while (count != 200):
    status, photo = cap.read()
    faces = face_classifier.detectMultiScale(photo, 1.3, 5)
    input_face = CheckFaces(faces) # To check if there is any face which was not detected due to any hassel
    if input_face is not None:
        cv2.putText(photo, str(count), (30, 50),cv2.FONT_ITALIC, 1, (255,0,0), 2) # Counter While Detecting Face! 
        cv2.imshow("Win", photo)
        count+=1
        cv2.imwrite(os.path.join(path , str(count) + ".png"), photo) # Saving The Images! 
        if cv2.waitKey(1)==13:
            print("Process Interrupted by user")
            break
    else:
        print("Train Face Not Found")
cv2.destroyAllWindows()
cap.release()
print("Training Data is now collected")



#Collecting Testing dataset for model!!  
cap2 = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count2=0
path_train="C:/Users/HP/Desktop/MLOps/Face/Validation/Naman/" # Path Where You'll save images for testing of your model.
def CheckFaces(sample_train):
    if sample_train is ():
        return None
    else:
        return sample_train
while (count2 != 40):
    status, test = cap2.read()
    test_face = face_classifier.detectMultiScale(test, 1.3, 5)
    input_test = CheckFaces(test_face) # To check if there is any face which was not detected due to any hassel
    if input_test is not None:
        cv2.putText(test, str(count2), (30, 50),cv2.FONT_ITALIC, 1, (255,0,0), 2) # Counter While Detecting Face! 
        cv2.imshow("Win", test)
        count2+=1
        cv2.imwrite(os.path.join(path_train , str(count2) + ".png"), test) # Saving The Images! 
        if cv2.waitKey(1)==13:
            print("Process Interrupted by user")
            break
    else:
        print("Test Face Not Found")
cv2.destroyAllWindows()
cap2.release()
print("Test Data is now collected")







