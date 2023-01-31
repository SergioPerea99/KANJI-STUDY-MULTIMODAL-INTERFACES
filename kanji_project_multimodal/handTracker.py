import mediapipe as mp
import numpy as np
import cv2


class HandTracker():

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):

        #Modo de imagen estática
        self.mode = mode

        #Numero maximo de manos
        self.maxHands = maxHands

        #Minimo de confianza en la detección
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.modelComplex = 1

        #Detección de manos
        self.mpHands = mp.solutions.hands

        #Modo en el que se detectan las manos
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)

        #Dibujar los puntos de las manos
        self.mpDraw = mp.solutions.drawing_utils

    
    #Función que detecta las manos
    def findHands(self, img, draw=True):

        #Se pasa de BGR a RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Se obtienen las manos
        self.results = self.hands.process(imgRGB)

        #Si hemos obtenido marcas de la mano    
        if self.results.multi_hand_landmarks:
            #Comprobamos esas marcas
            for handLm in self.results.multi_hand_landmarks:

                #Si queremos que se pinte
                if draw:
                    #las pintamos
                    self.mpDraw.draw_landmarks(img, handLm, self.mpHands.HAND_CONNECTIONS)

        #Devolvemos la imagen
        return img

    def getPostion(self, img, handNo = 0, draw=True):
        #Creamos una lista de landMark
        lmList =[]

        #Si tenemos marcas de la mano
        if self.results.multi_hand_landmarks:
            #Cojemos las marcas de la mano "handNo"
            myHand = self.results.multi_hand_landmarks[handNo]

            #Por cada marca de la mano
            for lm in myHand.landmark:

                #Obtenemos el tamaño de la imagen
                h, w, c = img.shape

                # Multiplicamos el  tamaño de la imgen por las landmarks
                # Ya que esto nos devuelve un valor entre 0 y 1 de como de lejos esta de la esquina inferior izquierda. Por ejemplo:
                # x: 0.3747435
                # y: 0.6749343
                cx, cy = int(lm.x*w), int(lm.y*h)

                #Las metemos en una lista
                lmList.append((cx, cy))

                #Si queremos dibujar un circulo en cada una de estas landmarks
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)
        
        #devolvemos la lista de landmark
        return lmList
    
    #Obtener los dedos que estan levantados
    def getUpFingers(self, img):

        #Obtenemos la posicion de los dedos
        pos = self.getPostion(img, draw=False)

        #Creamos una lista donde vamos a almacenar los dedos que estan levantados
        self.upfingers = []

        #Si pos tiene valores
        if pos:
            
            # IMPORTANTE PARA ENTENDERLO -> https://www.researchgate.net/publication/354115187/figure/fig1/AS:1060551832662018@1629866669595/Hand-landmarks-of-MediaPipe-Hands.ppm
            #pulgar
            # pos[4][1] < pos[3][1] -> Si la y del punto mas alto del pulgar esta por debajo del siguiente punto mas alto
            # Y
            # pos[5][0]-pos[4][0]> 10 -> si la x del punto 5 menos la x del punto mas alto del pulgar es mayor que -10
            self.upfingers.append((pos[4][1] < pos[3][1] and (pos[5][0]-pos[4][0]> 10)))

            #índice
            # pos[8][1] < pos[7][1]-> Si la y del punto mas alto del indice esta por debajo del siguiente punto mas alto
            # Y
            # pos[7][1] < pos[6][1] -> Si la y del segundo punto mas alto del indice esta por debajo del siguiente punto mas alto
            self.upfingers.append((pos[8][1] < pos[7][1] and pos[7][1] < pos[6][1]))

            #corazón
            # pos[12][1]-> Si la y del punto mas alto del indice esta por debajo del siguiente punto mas alto
            # Y
            # pos[11][1] < pos[10][1] -> Si la y del segundo punto mas alto del indice esta por debajo del siguiente punto mas alto
            self.upfingers.append((pos[12][1] < pos[11][1] and pos[11][1] < pos[10][1]))

            #anular
            # pos[16][1] < pos[15][1]-> Si la y del punto mas alto del indice esta por debajo del siguiente punto mas alto
            # Y
            # pos[15][1] < pos[14][1] -> Si la y del segundo punto mas alto del indice esta por debajo del siguiente punto mas alto
            self.upfingers.append((pos[16][1] < pos[15][1] and pos[15][1] < pos[14][1]))

            #meñique
            # pos[20][1] < pos[19][1]-> Si la y del punto mas alto del indice esta por debajo del siguiente punto mas alto
            # Y
            # pos[19][1] < pos[18][1] -> Si la y del segundo punto mas alto del indice esta por debajo del siguiente punto mas alto
            self.upfingers.append((pos[20][1] < pos[19][1] and pos[19][1] < pos[18][1]))

        #Devolvemos los dedos que estan levantados
        return self.upfingers

