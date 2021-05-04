import numpy as np
import cv2 as cv 

# img = cv.imread('imagen_input.jpg')
# face_cascade = cv.CascadeClassifier('opencv/data/haarcascades_cuda/haarcascade_smile.xml')
face_cascade = cv.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalcatface_extended.xml')
# face_cascade = cv.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_default.xml')

webcam = cv.VideoCapture(0)

while(1):
    valido, img = webcam.read()

    if valido:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30), maxSize=(150,150), flags=cv.CASCADE_SCALE_IMAGE)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            img = cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

        cv.imshow('meow meow dijo el gato', img)


        k = cv.waitKey(5) & 0xFF
        if k == 27:
            cv.destroyAllWindows()
            break

webcam.release()


