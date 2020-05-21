import cv2
import numpy as numpy
import dlib
from math import hypot
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
glasses_img = cv2.imread("images/glasses.png")
mustache_img = cv2.imread("images/mustache.png")

def put_glasses(frame,landmarks):

    top_eyebrow = (landmarks.part(21).x,landmarks.part(21).y)
    left_eye = (landmarks.part(36).x,landmarks.part(36).y)
    right_eye = (landmarks.part(45).x,landmarks.part(45).y)
    center_eye = (landmarks.part(27).x,landmarks.part(27).y)

    glasses_width = int(hypot(left_eye[0]-right_eye[0],left_eye[1]-right_eye[1])*1.5)
    glasses_height = int(glasses_width*0.8)

    glasses = cv2.resize(glasses_img,(glasses_width,glasses_height))
    glasses_gray = cv2.cvtColor(glasses,cv2.COLOR_BGR2GRAY)

    _,glasses_mask = cv2.threshold(glasses_gray,200,255,cv2.THRESH_BINARY_INV) # Black-white nose mask, black area = emoji area
    mask_inv = cv2.bitwise_not(glasses_mask)
    top_left = (int(center_eye[0]-glasses_width/2),int(center_eye[1]-glasses_height/2))
    bottom_right = (int(center_eye[0]+glasses_width/2),int(center_eye[1]+glasses_height/2))

    glasses_area = frame[top_left[1]:top_left[1]+glasses_height, top_left[0]:top_left[0]+glasses_width]

    glasses_area_no_glasses = cv2.bitwise_and(glasses_area,glasses_area,mask=mask_inv)
    crop_glasses = cv2.bitwise_and(glasses,glasses,mask=glasses_mask)

    final_eyes = cv2.add(glasses_area_no_glasses,crop_glasses)

    frame[top_left[1]:top_left[1]+glasses_height, top_left[0]:top_left[0]+glasses_width] = final_eyes

    return frame


def put_mustache(frame,landmarks):

    left_bottom = (landmarks.part(50).x,landmarks.part(50).y)
    right_bottom = (landmarks.part(52).x,landmarks.part(52).y)
    left_top = (landmarks.part(31).x,landmarks.part(31).y)
    right_top = (landmarks.part(35).x,landmarks.part(35).y)
    center = (landmarks.part(51).x,landmarks.part(51).y)
    mustache_width = int(hypot(left_top[0]-right_top[0],left_top[1]-right_top[1])*3)
    mustache_height = int(mustache_width*0.5)

    mustache = cv2.resize(mustache_img,(mustache_width,mustache_height))
    mustache_gray = cv2.cvtColor(mustache,cv2.COLOR_BGR2GRAY)

    _,mustache_mask = cv2.threshold(mustache_gray,200,255,cv2.THRESH_BINARY_INV) # Black-white nose mask, black area = emoji area
    mask_inv = cv2.bitwise_not(mustache_mask)
    top_left = (int(center[0]-mustache_width/2),int(center[1]-mustache_height/2))
    bottom_right = (int(center[0]+mustache_width/2),int(center[1]+mustache_height/2))

    mustache_area = frame[top_left[1]:top_left[1]+mustache_height, top_left[0]:top_left[0]+mustache_width]

    mustache_area_no_mustache = cv2.bitwise_and(mustache_area,mustache_area,mask=mask_inv)
    crop_mustache = cv2.bitwise_and(mustache,mustache,mask=mustache_mask)

    final_mustache = cv2.add(mustache_area_no_mustache,crop_mustache)

    frame[top_left[1]:top_left[1]+mustache_height, top_left[0]:top_left[0]+mustache_width] = final_mustache

    return frame

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

    frame = put_glasses(frame,landmarks)

    frame = put_mustache(frame,landmarks)

    

    


    cv2.imshow("Frame",frame)
    

  key = cv2.waitKey(1)

  if key == 27:
    break