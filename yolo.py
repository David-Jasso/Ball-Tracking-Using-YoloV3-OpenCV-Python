import cv2 
import numpy as np

# Obtener los nombres de la capa de salida de la red neuronal.
def getOutputsNames(yolo):
    # Obtener todos los nombres de la red neuronal.
    layersNames = yolo.getLayerNames() 

    # Obtener los nombres de la ultima capa de la red neuronal.
    return [layersNames[i - 1] for i in yolo.getUnconnectedOutLayers()]

# Procesamiento de las entradas de la red neuronal.
def yolo_predict(image, height, width, yolo):
    # Crear una mancha 4D a partir de un fotograma
    blob = cv2.dnn.blobFromImage(image, 1 /255, (width, height), [0,0,0], 1, crop=False)

    # Establece la entrada a la red neuronal.
    yolo.setInput(blob)

    # Retornar el forward pass para obtener la salida de la red neuronal.
    return yolo.forward(getOutputsNames(yolo))

# Dibujar el cuadro delimitador despues de encontrar los objetos detectados
def draw_pred(box, confidence, classId, image, classes):
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]

    # Mostrar el cuadro delimitador para cada objeto detectado.
    cv2.rectangle(image, (left, top), (left + width, top + height), (255, 0, 0), 1)
    # Mostrar la etiqueta en la parte superior del cuadro delimitador.
    cv2.putText(image, classes[classId], (left, top), cv2.FONT_HERSHEY_COMPLEX, 1.4, (0,0, 255), 2)

# Post Procesamiento de la salida de la red para eliminar cuadros delimitadores de baja confianza.
def yolo_postprocess(image, outs, objectnessThreshold, confThreshold, nmsThreshold, classes):
    image_h, image_w = image.shape[:2]

    classIds = []
    confidences = []
    boxes = []
    newboxes = []
    newclassIds = []

    #Escanear cuadros delimitadores de la red y mantener solo los de alta confianza.
    for out in outs:
        for detection in out:
            #if detection[4] > objectnessThreshold:
            scores = detection[5:]  #Aqui son todas las clases y su probabilidad de pertenecer a esa clase
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:  
                center_x = int(detection[0] * image_w)
                center_y = int(detection[1] * image_h)
                width = int(detection[2] * image_w)
                height = int(detection[3] * image_h)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                classIds.append(classId)
                confidences.append(confidence)
                boxes.append([left, top, width, height])

    # Realizar una supresión no máxima para eliminar las cajas delimitadoras de baja confianza.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in indices:
        box = boxes[i]
        cId = classIds[i]
        newboxes.append(box)
        newclassIds.append(cId)
        draw_pred(box, confidences[i], classIds[i], image, classes)

    return indices, newboxes, confidence, newclassIds

