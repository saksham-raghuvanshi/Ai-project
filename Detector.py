import os
from time import sleep

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def main_app(name):
    face_cascade = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")
    model = load_model("mask_recog.h5")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print(f".\\data\\classifiers\\{name}_classifier.xml")
    recognizer.read(f".\\data\\classifiers\\{name}_classifier.xml")
    cap = cv2.VideoCapture(0)
    pred = 0
    while True:
        ret, frame = cap.read()
        # default_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        faces_list = []
        # preds = []

        for x, y, w, h in faces:
            roi_gray = gray[y : y + h, x : x + w]

            # face mask detection
            face_frame = frame[y : y + h, x : x + w]
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_frame = cv2.resize(face_frame, (224, 224))
            face_frame = img_to_array(face_frame)

            # Expand the shape of an array. Insert a new axis that will appear at the axis position in the expanded array shape.
            face_frame = np.expand_dims(face_frame, axis=0)
            face_frame = preprocess_input(face_frame)
            faces_list.append(face_frame)

            id, confidence = recognizer.predict(roi_gray)

            if len(faces_list) > 0:
                """Generates output predictions for the input samples.
                Computation is done in batches. This method is designed for batch processing of large numbers of inputs. It is not intended for use inside of loops that iterate over your data and process small numbers of inputs at a time.
                """
                print(faces_list)
                pred_mask = model.predict(faces_list[-1])  # Return Numpy array(s) of predictions.

            print(f"preds : {pred_mask}")
            for pred in pred_mask:
                (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(
                frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
            )

            # cv2.rectangle(frame, (x, y), (x + w, y + h),color, 3)

            ####### end of face mask detection | Result is in label variable
            confidence = 100 - int(confidence)
            pred = 0
            if confidence > 50:
                # if u want to print confidence level
                # confidence = 100 - int(confidence)
                pred += +1
                text = name.upper()
                font = cv2.FONT_HERSHEY_PLAIN
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = cv2.putText(
                    frame,
                    text,
                    (x, y - 4),
                    font,
                    1,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            else:
                pred += -1
                text = "UnknownFace"
                font = cv2.FONT_HERSHEY_PLAIN
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                frame = cv2.putText(
                    frame,
                    text,
                    (x, y - 4),
                    font,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imshow("image", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(pred)
            if pred > 0:
                dim = (124, 124)
                img = cv2.imread(
                    f".\\data\\{name}\\{pred}{name}.jpg", cv2.IMREAD_UNCHANGED
                )
                resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(f".\\data\\{name}\\50{name}.jpg", resized)
                Image1 = Image.open(f".\\2.png")

                # make a copy the image so that the
                # original image does not get affected
                Image1copy = Image1.copy()
                Image2 = Image.open(f".\\data\\{name}\\50{name}.jpg")
                Image2copy = Image2.copy()

                # paste image giving dimensions
                Image1copy.paste(Image2copy, (195, 114))

                # save the image
                Image1copy.save("end.png")
                frame = cv2.imread("end.png", 1)

                cv2.imshow("Result", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()


# face mask detection code

# faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
# faceCascade = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")

# def face_mask_detector(frame):
#   # frame = cv2.imr ead(fileName)

#   # Converts an image from one color space to another. The function converts an input image from one color space to another
#   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # color space conversion code

#   # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
#   faces = faceCascade.detectMultiScale(gray,
#                                         scaleFactor=1.1,
#                                         minNeighbors=5,
#                                         minSize=(60, 60),
#                                         flags=cv2.CASCADE_SCALE_IMAGE)
#   faces_list=[]
#   preds=[]
#   for (x, y, w, h) in faces:
#         # frame[y:y+h, x:x+w] specifies a slice of the frame array, where y and x are the starting indices of the row and column ranges, respectively, and h and w are the heights and widths of the sub-array to be extracted.-
#       face_frame = frame[y:y+h,x:x+w]
#       face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
#       face_frame = cv2.resize(face_frame, (224, 224))
#       face_frame = img_to_array(face_frame)

#       # Expand the shape of an array. Insert a new axis that will appear at the axis position in the expanded array shape.
#       face_frame = np.expand_dims(face_frame, axis=0)
#       face_frame =  preprocess_input(face_frame)
#       faces_list.append(face_frame
#       )
#       if len(faces_list)>0:
#         """Generates output predictions for the input samples.

#         Computation is done in batches. This method is designed for batch processing of large numbers of inputs. It is not intended for use inside of loops that iterate over your data and process small numbers of inputs at a time.
#         """
#         preds = model.predict(faces_list) # Return Numpy array(s) of predictions.

#       for pred in preds:
#           (mask, withoutMask) = pred

#       label = "Mask" if mask > withoutMask else "No Mask"
#       color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
#       label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
#       cv2.putText(frame, label, (x, y- 10),
#                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#       cv2.rectangle(frame, (x, y), (x + w, y + h),color, 3)
#   # cv2_imshow(frame)
#   return frame


# """To capture a video, you need to create a VideoCapture object. Its argument can be either the device index or the name of a video file. A device index is just the number to specify which camera. Normally one camera will be connected (as in my case). So I simply pass 0 (or -1). You can select the second camera by passing 1 and so on. After that, you can capture frame-by-frame
#     """
# cap = cv2.VideoCapture(0)

# # cap.read() returns a bool (True/False). If the frame is read correctly, it will be True. So you can check for the end of the video by checking this returned value.
# # ret, frame = cap.read()
# # frame_height, frame_width, _ = frame.shape
# out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
# print("Processing Video...")
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

# while cap.isOpened(): # Returns true if video writer has been successfully initialized.
#   ret, frame = cap.read()
#   if not ret:
#     out.release() # Closes the video writer.
#     break
#   output = face_mask_detector(frame)
#   out.write(output) # Writes the next video frame.

# # Closes the video writer.
# out.release()
# print("Done processing video")
