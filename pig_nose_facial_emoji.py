import cv2
import numpy as numpy
import dlib
from math import hypot
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
nose_img = cv2.imread("images/pig_nose.png")

while True:

  _,frame = cap.read()

  # Getting a gray frame for landmark prediction - Less Computation
  gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

  # Detecting faces in frame
  faces = detector(frame)

  # Faces can be multiple- so processing on every face
  for face in faces:

    # Getting landmarks of current face from gray frame
    landmarks = predictor(gray_frame,face)

    top_nose = (landmarks.part(29).x,landmarks.part(29).y)
    left_nose = (landmarks.part(31).x,landmarks.part(31).y)
    right_nose = (landmarks.part(35).x,landmarks.part(35).y)
    center_nose = (landmarks.part(30).x,landmarks.part(30).y)
    #cv2.circle(frame,top_nose,3,(255,0,0),-1)

    # Calculating width of nose- distn between 2 points - Eucledian formula
    nose_width = int(hypot(left_nose[0]-right_nose[0],left_nose[1]-right_nose[1])*1.7)
    nose_height = int(nose_width*0.7)

    # Resizing nose according to size of nose being detected
    nose_pig = cv2.resize(nose_img,(nose_width,nose_height))
    # Getting gray frame of pig nose emoji
    nose_pig_gray = cv2.cvtColor(nose_pig,cv2.COLOR_BGR2GRAY)

    # Creating a frame from gray frame of pig nose. Wherever nose pixel value is > 200, set is to 255(white) else 0(black){0-black,1-white}
    _,nose_mask = cv2.threshold(nose_pig_gray,200,255,cv2.THRESH_BINARY_INV) # Black-white nose mask, black area = emoji area
    mask_inv = cv2.bitwise_not(nose_mask)
    top_left = (int(center_nose[0]-nose_width/2),int(center_nose[1]-nose_height/2))
    bottom_right = (int(center_nose[0]+nose_width/2),int(center_nose[1]+nose_height/2))

    #cv2.rectangle(frame,top_left,bottom_right,(0,255,0),2) drawing rectangle on nose where emoji will be placed

    # displaying emoji area from actual frame - first y then x (y:y+h,x:x+w), starting pt. - Top Left
    nose_area = frame[top_left[1]:top_left[1]+nose_height, top_left[0]:top_left[0]+nose_width]

    #1. Putting the B/W mask over nose.  - emoji area px val = 0(black)
    nose_area_no_nose = cv2.bitwise_and(nose_area,nose_area,mask=mask_inv)
    #2. Taking only emoji part from pig nose img- px value outside nose emoji= 0(black) 
    crop_pig_nose = cv2.bitwise_and(nose_pig,nose_pig,mask=nose_mask)

    # Just adding 1 and 2
    final_nose = cv2.add(nose_area_no_nose,crop_pig_nose)

    # Putting finnal nose rectangle on original Frame
    frame[top_left[1]:top_left[1]+nose_height, top_left[0]:top_left[0]+nose_width] = final_nose

    


    cv2.imshow("Frame",frame)
    #cv2.imshow("Crop Pig Nose",crop_pig_nose)
    #cv2.imshow("Mask Inv",mask_inv)
    #cv2.imshow("Pig Nose",nose_pig)
    #cv2.imshow("Nose Mask",nose_mask)
    #cv2.imshow("Gray Pig Nose",nose_pig_gray)
    #cv2.imshow("Nose Area",nose_area)
    #cv2.imshow("Nose ANN",nose_area_no_nose)
    #cv2.imshow("Final Nose",final_nose)

  key = cv2.waitKey(1)

  if key == 27:
    break