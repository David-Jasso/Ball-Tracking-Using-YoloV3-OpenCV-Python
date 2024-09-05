import cv2 as cv
import numpy as np
from random import randint
from yolo import yolo_predict, yolo_postprocess
import time

# Inicializar parametros para YOLOV3.
objectnessThreshold = 0.5 # Umbral de objetualidad (0-1).
confThreshold = 0.5       # Umbral de confianza (0-1).
nmsThreshold = 0.4        # Umbral de supresión no máxima NMS.
inpWidth = 416            # Anchura de la imagen de entrada.
inpHeight = 416           # Altura de la imagen de entrada.

# Cargar las clase que contiene todos los objetos para entrenar el modelo.
classesFile = "DeteccionySeguimiento\models\coco.names"
classes = []

# Abrir el archivo en modo lectura y almacenar las classes en una lista.
with open(classesFile, 'rt') as f: 
    classes = f.read().rstrip('\n').split('\n')

# Cargar configuración y pesos de la red neuronal.
modelConfig = "DeteccionySeguimiento\models\yolov3.cfg"
modelWeights = "DeteccionySeguimiento\models\yolov3.weights"

# Cargar el modelo a partir de los pesos y la configuración dada.
yolo = cv.dnn.readNetFromDarknet(modelConfig, modelWeights)

# Iniciar Tracker TLD para seguimiento de objetos.
trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
#Tracking, learning, and detection.
tracker = cv.legacy.TrackerTLD_create() 

# Crear bandera para habilitar el seguimiento.
tracking_enabled = False

# Iniciar captura de video.
video = "DeteccionySeguimiento\soccer-ball.mp4"
cap = cv.VideoCapture(video)

# Definir el códec y crear el objeto VideoWriter
video = cv.VideoWriter("DeteccionySeguimiento/BallTracking.avi", cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (int(cap.get(3)), int(cap.get(4))))

if not cap.isOpened():
    print("Error al conectar la cámara")
    exit()

# Bucle infinito.
while True:
    ret, frame = cap.read()
    key = cv.waitKey(1)

    if not ret: 
        print("Error reading frame")
        break

    if key == ord('q'): break
    elif key == ord('d') or not tracking_enabled:
        # Procesamiento de la red neuronal.
        outs = yolo_predict(frame, inpHeight, inpWidth, yolo)
        print(outs[0])
        
        # Eliminar los recuadros delimitadores de baja confianza.
        ind, box, conf, class_id = yolo_postprocess(frame, outs, objectnessThreshold, confThreshold, nmsThreshold, classes)
        print(class_id)  # Impresion de las clases de los objetos detectados.
        print(ind)       # Impresion de los indices de los objetos detectados.
        print(box)       # Impresion de los recuadros de los objetos detectados.

        ball = False
        bbox = []
        #Ciclo para obtener la clase "sport ball" de la deteccion de objetos.
        for i in range(len(class_id)):
            if class_id[i] == 32:
                print('sport ball: ', class_id[i])
                bbox = box[i]
                ball = True
                break
                  
        #Asignar a bbox la caja delimitadora de la clase 32 o clase 'sport ball'.
        print('bbox: ', bbox)
        if ball:
            print('bbox: ', bbox)
            bbox = tuple(bbox)
            # Habilitar bandera para realizar el seguimiento.
            tracking_enabled = True
            tracker.init(frame, bbox)

        # Mostrar Deteccion de Objetos.
        cv.imshow("Video", frame)
        cv.waitKey(0)
    
    elif tracking_enabled:
        #Inicializar tracker.
        #tracker.init(frame, bbox)
        # Update tracker
        ok, bbox = tracker.update(frame)

        #Dibujar el cuadro delimitador obtenido de la detección.
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            cv.putText(frame, "Tracking detected", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        else:
            # Tracking failure
            cv.putText(frame, "Tracking failure detected", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            bbox = []
            tracking_enabled = False
        
    # Escribir el video.
    video.write(frame)
    cv.imshow("Video", frame)
    
    
#Destruir las ventanas creadas para liberar espacio
cap.release()
video.release()
cv.destroyAllWindows()