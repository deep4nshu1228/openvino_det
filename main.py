# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-08-03 18:42:33
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-08-03 19:24:27

import time
import click

import cv2

from src.face_detector import FaceDetector
from src import utils
from AgeGenderRecognitionModel import AgeGenderRecognitionModel

@click.command()
@click.option('-v','--video_source', default=0)
@click.option('-c','--confidence', type=float, default=0.5)

def crop_image_with_bbox(image, bbox):
  """
  Crops an image within a given bounding box.

  Args:
    image: A NumPy array representing the image.
    bbox: A list of four integers representing the bounding box coordinates (xmin, ymin, xmax, ymax).

  Returns:
    A NumPy array representing the cropped image.
  """

  # Ensure the bounding box coordinates are within image bounds.
  try:
        xmin = max(0, bbox[0])
        ymin = max(0, bbox[1])
        xmax = min(image.shape[1], bbox[2])
        ymax = min(image.shape[0], bbox[3])
  except Exception as e:
      print(e)

  # Crop the image using the adjusted bounding box.
  cropped_image = image[ymin:ymax, xmin:xmax]

  return cropped_image



def main(video_source=0, confidence=0.5):

    detector = FaceDetector(model='./ultra-lightweight-face-detection-rfb-320/FP16/ultra-lightweight-face-detection-rfb-320.xml',         confidence_thr=0.5, overlap_thr=0.7)


    #---------------------------------------
    # agd = AgeGenderRecognitionModel("./age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml")
    # agd.load_model()

    ageGenderNet = cv2.dnn.readNet("./age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml",
                               "./age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin")
    genders = ["female", "male"]
    video = cv2.VideoCapture(0)

    n_frames = 0
    fps_cum = 0.0
    fps_avg = 0.0
    cur_req_id = {'age':0}
    next_req_id = {'age':1}
    while True:
        ret, frame = video.read()
        if ret == False:
            print("End of the file or error to read the next frame.")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frame = clahe.apply(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        start_time = time.perf_counter()
        bboxes, scores = detector.inference(frame)
        end_time = time.perf_counter()

        # facedetection = FaceDetector("./face-detection-retail-0005/FP32/face-detection-retail-0005.xml")

        for bbox in bboxes:
            print("bbox: {}".format(bbox))
            x_min, y_min, x_max, y_max = bbox
            x = x_min
            y = y_min
            w = x_max - x_min
            h = y_max - y_min

            if w > 0:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(128,128,128),1) #draw rectangle to main image

                detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
    
                # age, gender = agd.predict(frame[int(y):int(y+h), int(x):int(x+w)], cur_req_id['age'], next_req_id['age'])

                # if gender==0:
                #     gender='F'
                # else:
                #     gender='M'
                # age = int(age)
                # cv2.putText(frame, "age:"+str(int(age))+" gender:"+str(gender), (bbox[0]-10,bbox[1]+5), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)


                faceBlob = cv2.dnn.blobFromImage(detected_face, 1.0, (227, 227),
                                 (78.4263377603, 87.7689143744, 114.895847746),
                                 swapRB=False)
                
                ageGenderNet.setInput(faceBlob)
                aPreds = ageGenderNet.forward()
                predict_gender = aPreds[0][1][0][0]
                gConfidence = aPreds[0][1][1][0]
                gd = genders[int(predict_gender + 0.5)]

                predict_age = aPreds[0][0][0][0] * 100
                aConfidence = aPreds[0][0][1][0]

                cv2.putText(frame, "age:"+str(int(predict_age))+" gender:"+str(gd), (bbox[0]-10,bbox[1]+5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)



        n_frames += 1
        fps = 1.0 / (end_time - start_time)
        fps_cum += fps
        fps_avg = fps_cum / n_frames

        frame = utils.draw_boxes_with_scores(frame, bboxes, scores)
        frame = utils.put_text_on_image(frame, text='FPS: {:.2f}'.format( fps_avg ))

        cv2.imshow('video', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break

if __name__ == '__main__':
    main()