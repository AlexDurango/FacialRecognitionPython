import numpy as np
import cv2 as cv 

# img = cv.imread('imagen_input.jpg')
# face_cascade = cv.CascadeClassifier('opencv/data/haarcascades_cuda/haarcascade_smile.xml')
# face_cascade = cv.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalcatface_extended.xml')

# Haarcascade para la AI
face_cascade = cv.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_default.xml')

# Iniciar la cámara
webcam = cv.VideoCapture(0)

while(1):

    # Lee los datos de la cam
    valido, img = webcam.read()

    if valido:

        # Convierte la imagen en color a gris
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Detecta los rostros using gray image
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(5,5), maxSize=(150,150), flags=cv.CASCADE_SCALE_IMAGE)

        # faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Por cada cara, esa retorna el alto, ancho, X y Y.
        for (x,y,w,h) in faces:
            img = cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

        # Abrir la ventana
        cv.imshow('facial detection', img)


        # Espera a que se presione ESC para cerrar la ventana
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            cv.destroyAllWindows()
            break

webcam.release()


