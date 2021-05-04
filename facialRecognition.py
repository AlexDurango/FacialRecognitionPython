import cv2
import face_recognition


# Cargar imagenes
img_alex = face_recognition.load_image_file("Alex.jpeg")
img_rick = face_recognition.load_image_file("Ricky.jpeg")
img_hitler = face_recognition.load_image_file("hitler.jpeg")
img_ismael = face_recognition.load_image_file("IsmaelPemberty.jpeg")
img_luna = face_recognition.load_image_file("Lunita.jpeg")
img_cam = face_recognition.load_image_file("Cam.jpeg")

# encodings de las imagenes

alex_encodings = face_recognition.face_encodings(img_alex)[0]
rick_encodings = face_recognition.face_encodings(img_rick)[0]
hitler_encodings = face_recognition.face_encodings(img_hitler)[0]
ismael_encodings = face_recognition.face_encodings(img_ismael)[0]
luna_encodings = face_recognition.face_encodings(img_luna)[0]
cam_encodings = face_recognition.face_encodings(img_cam)[0]

encodings_conocidos = [
    alex_encodings,
    rick_encodings,
    hitler_encodings,
    ismael_encodings,
    luna_encodings,
    cam_encodings
]
nombres_conocidos = [
    "Alex Durango",
    "Rick Astley",
    "Adolf Hitler",
    "Ismael Pemberty",
    "Luna Ortiz",
    "Cam"
]

#Inicializa la cámara
webcam = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX

# Reduce la imagen a 1/5 del tamaño original
reduccion = 5

while 1:
    loc_rostros = []
    encodings_rostros = []
    nombres_rostros = []
    nombre = ""

    valid, img = webcam.read()

    if valid:
        img_rgb = img[:,:,::-1]
        img_rgb = cv2.resize(img_rgb, (0,0), fx=1.0/reduccion, fy=1.0/reduccion)

        loc_rostros = face_recognition.face_locations(img_rgb)
        encodings_rostros = face_recognition.face_encodings(img_rgb, loc_rostros)

        for encoding in encodings_rostros:

            coincidencias = face_recognition.compare_faces(encodings_conocidos, encoding)
            if True in coincidencias:
                nombre = nombres_conocidos[coincidencias.index(True)]
            else:
                nombre = "???"

            nombres_rostros.append(nombre)
        
        for (top, right, bottom, left), nombre in zip(loc_rostros, nombres_rostros):
            top = top*reduccion
            right = right*reduccion
            bottom = bottom*reduccion
            left = left*reduccion

            if nombre != "???":
                color = (0,255,0)
            else:
                color = (0,0,255)
            
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            cv2.rectangle(img, (left, bottom - 20), (right, bottom), color, -1)
            cv2.putText(img, nombre, (left, bottom - 6), font, 0.6, (0,0,0), 1)
        
        cv2.imshow('computer goes brrrr', img)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break

webcam.release()






