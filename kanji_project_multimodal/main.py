from handTracker import *
import cv2
import mediapipe as mp
import numpy as np
import random
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from queue import Queue
import os
from PyQt5.QtGui import QPixmap
import threading


#Clase para pintar un rectangulo con un texto
class ColorRect():
    def __init__(self, x, y, w, h, color, text='', alpha = 0.8):

        #Posiciones del rectangulo
        self.x = x
        #y con respecto a la esquina superior izquierda
        self.y = y
        self.w = w
        self.h = h

        #Color del rectangulo
        self.color = color

        #Texto del rectangulo
        self.text=text

        #Transparencia del rectangulo
        self.alpha = alpha
        
    
    def drawRect(self, img, text_color=(255,255,255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):

        #Dibujamos el rectangulo
        #Transparencia
        alpha = self.alpha

        #tamaño del rectangulo
        bg_rec = img[self.y : self.y + self.h, self.x : self.x + self.w]

        #Color del rectangulo(Blanco por eso es ones)
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8)

        #Aqui cambiamos el color si es otro
        white_rect[:] = self.color

        #Mezclamos el fondo de la imagen el rectangulo
        #bg_rec = imgagen del fondo
        #alpha = transparencia de dicha imagen
        #white_rect = rectangulo blanco 
        #1-alpha = transparencia del rectangulo
        #1.0 -> constante
        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1-alpha, 1.0)
        
        # Ponemos la imagen de vuelta en su lugar
        img[self.y : self.y + self.h, self.x : self.x + self.w] = res

        #Ponemos el texto
        tetx_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)
        text_pos = (int(self.x + self.w/2 - tetx_size[0][0]/2), int(self.y + self.h/2 + tetx_size[0][1]/2))
        cv2.putText(img, self.text,text_pos , fontFace, fontScale,text_color, thickness)


    def drawRect_img(self, frame_actual_camera, img, alpha = 0.8):
        # Dibujamos el rectangulo
        # Transparencia
        alpha = self.alpha

        # Tamaño del rectangulo
        bg_rec = frame_actual_camera[self.y : self.y + self.h, self.x : self.x + self.w]

        # Color del rectangulo (Blanco por eso es ones)
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8)

        # Aqui cambiamos el color si es otro
        white_rect[:] = self.color

        # Mezclamos el fondo de la imagen el rectangulo
        # bg_rec = imagen del fondo
        # alpha = transparencia de dicha imagen
        # white_rect = rectangulo blanco 
        # 1-alpha = transparencia del rectangulo
        # 1.0 -> constante
        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1-alpha, 1.0)

        # Ponemos la imagen de vuelta en su lugar
        frame_actual_camera[self.y : self.y + self.h, self.x : self.x + self.w] = res

        # Ajustamos el tamaño de la imagen a mostrar en el rectángulo
        img = cv2.resize(img, (self.w, self.h))

        # Copiamos la imagen sobre el rectángulo
        frame_actual_camera[self.y:self.y+self.h, self.x:self.x+self.w] = cv2.addWeighted(frame_actual_camera[self.y:self.y+self.h, self.x:self.x+self.w], 1, img, 1, 0)


    #Para saber si se sale de la pantalla
    def isOver(self,x,y):
        if (self.x + self.w > x > self.x) and (self.y + self.h> y >self.y):
            return True
        return False



#METODO PARA COMPARAR LOS KANJIS
def compare_kanji(drawing, kanji_path):
    # Load the Kanji image
    kanji = cv2.imread(kanji_path, cv2.IMREAD_GRAYSCALE)

    # Convert the drawing to grayscale
    drawing_gray = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)

    # Use matchTemplate to compare the drawing and the Kanji image
    result = cv2.matchTemplate(drawing_gray, kanji, cv2.TM_CCOEFF_NORMED)

    # Get the maximum value from the result (which represents the best match)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Return the score
    return max_val


#METODO PARA MOSTRAR EL KANJI RANDOMIZADO
def loadReferenceKanji(folder):
    # Select a random Kanji image from the folder
    kanjiFilename = random.choice(os.listdir(folder))
    kanjiPath = os.path.join(folder, kanjiFilename)
    kanji = cv2.imread(kanjiPath)
    
    return kanji, kanjiPath

#Inicializamos el detector de manos
detector = HandTracker(detectionCon=0.8)

#Inicializamos la camara
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Creamos la pizarra para poder trabajar en ella
canvas = np.zeros((720,1280,3), np.uint8)
referenceKanji = np.zeros((720,1280,3), np.uint8)


# definir un punto anterior para utilizarlo con el dibujo de una línea
px,py = 0,0
#color inicial del pincel
color = (0,0,0)
#Parámetros del pincel
#####
brushSize = 20
eraserSize = 20
####

########### creación de colores ########

# Colores de los botones
colorsBtn = ColorRect(200, 0, 100, 100, (120,255,0), 'Colors')

colors = []

#color random
b = int(random.random()*255)-1
g = int(random.random()*255)
r = int(random.random()*255)
print(b,g,r)
colors.append(ColorRect(300,0,100,100, (b,g,r)))
#rojo
colors.append(ColorRect(400,0,100,100, (45,0,255)))
#azul
colors.append(ColorRect(500,0,100,100, (255,0,171)))
#verde
colors.append(ColorRect(600,0,100,100, (0,210,70)))
#amarillo
colors.append(ColorRect(700,0,100,100, (0,255,255)))
#borrar (negro)
colors.append(ColorRect(800,0,100,100, (0,0,0), "Eraser"))

#boton de clear
clear = ColorRect(900,0,100,100, (100,100,100), "Clear")

#------ BOTON PARA COMPARACIÓN DEL KANJI ------#
# Botón de puntuación
scoreBtn = ColorRect(1100, 100, 150, 100, (0,0,255), 'SCORE')

#Botón de mostrar un kanji a dibujar
kanjiRandomBtn = ColorRect(1100, 300, 150, 100, (0,0,0), 'KANJI')

#Botón de mostrar un kanji a dibujar
# Assume that the whiteboard has dimensions (width, height)
width, height = 1280, 720
button_width = int(width * 0.5)
button_height = int(height * 0.5)
button_x = int((width - button_width) / 2)
button_y = int((height - button_height) / 1.5)
kanjiRandom = ColorRect(button_x, button_y, button_width, button_height, (255,255,255), alpha = 0.6)
#kanjiRandom= ColorRect(50, 120, 1020, 580, (255,255,255),alpha = 1)

#Botón de mostrar un kanji a dibujar
finishBtn = ColorRect(1100, 500, 150, 100, (0,0,0), 'EXIT')

#Botón que muestra la puntuación
scoreDisplay = ColorRect(1100, 0, 150, 100, (255,255,255), 'Score: 0')

########## tamaño del pincel #######
pens = []
for i, penSize in enumerate(range(5,25,5)):
    pens.append(ColorRect(1100,50+100*i,100,100, (50,50,50), str(penSize)))

penBtn = ColorRect(1100, 0, 100, 50, color, 'Pen')

# Boton para obtener la pizarra
boardBtn = ColorRect(50, 0, 100, 100, (255,255,0), 'Board')

#define a white board to draw on
whiteBoard = ColorRect(50, 120, 1020, 580, (255,255,255),alpha = 0.2)

coolingCounter = 20
hideBoard = False
hideColors = True
hidePenSizes = True
AlreadyShowed = False
#Bucle de la aplicación
while True:

    if coolingCounter:
        coolingCounter -=1
        #print(coolingCounter)

    #LEEMOS LA CAPTURA DE VIDEO
    #RET = SI HA OBTENIDO LA IMAGEN O NO
    #FRAME = EL FRAME QUE HA OBTENIDO
    ret, frame = cap.read()

    #SI NO OBTENEMOS IMAGEN SE ACABA EL PROGRAMA
    if not ret:
        break
        
    #Si obtenemos la imagen le hacemos un resize
    frame = cv2.resize(frame, (1280, 720))

    #Le quitamos a la imagen el modo espejo, para que sea mas intuitivo
    frame = cv2.flip(frame, 1)

    #Detectamos las manos
    detector.findHands(frame)

    #Obtenemos la posición
    positions = detector.getPostion(frame, draw=False)

    #Obtenemos los dedos que estan levantados
    upFingers = detector.getUpFingers(frame)

    #Si hay dedos levantados
    if upFingers:

        #Posicion del punto mas alto del dedo indice
        x, y = positions[8][0], positions[8][1]

        #Si el dedo indice esta levantado
        #Y
        #No esta sobre la pizarra
        if upFingers[1] and not whiteBoard.isOver(x, y):
            px, py = 0, 0


            ##### Tamaño del lapiz ######
            if not hidePenSizes:
                for pen in pens:
                    if pen.isOver(x, y):
                        brushSize = int(pen.text)
                        pen.alpha = 0
                    else:
                        pen.alpha = 0.5

            ####### Cambiar de color del lapiz #######
            if not hideColors:
                for cb in colors:
                    if cb.isOver(x, y):
                        color = cb.color
                        cb.alpha = 0
                    else:
                        cb.alpha = 0.5

                #Se resetea la pizarra
                if clear.isOver(x, y):
                    clear.alpha = 0
                    canvas = np.zeros((720,1280,3), np.uint8)
                else:
                    clear.alpha = 0.5
            
            # BOTONES DE COLORS
            if colorsBtn.isOver(x, y) and not coolingCounter:
                #He cambiado este valor por 15 para ver si da mejor resultado
                coolingCounter = 15
                colorsBtn.alpha = 0
                hideColors = False if hideColors else True
                colorsBtn.text = 'Colors' if hideColors else 'Hide'
            else:
                colorsBtn.alpha = 0.5
            
            # Boton del tamaño del lapiz
            if penBtn.isOver(x, y) and not coolingCounter:
                #He cambiado este valor por 15 para ver si da mejor resultado
                coolingCounter = 15
                penBtn.alpha = 0
                hidePenSizes = False if hidePenSizes else True
                penBtn.text = 'Pen' if hidePenSizes else 'Hide'
            else:
                penBtn.alpha = 0.5

            
            #Boton del tamaño de la pizarra
            if boardBtn.isOver(x, y) and not coolingCounter:
                #He cambiado este valor por 15 para ver si da mejor resultado
                coolingCounter = 15
                boardBtn.alpha = 0
                hideBoard = False if hideBoard else True
                boardBtn.text = 'Board' if hideBoard else 'Hide'

            else:
                boardBtn.alpha = 0.5

            #Boton de mostrar kanji random a hacer
            if kanjiRandomBtn.isOver(x, y) and not AlreadyShowed:
                # Load a new reference Kanji and store it in the global variable
                
                referenceKanji, ruta_kanji_random = loadReferenceKanji('imagenes')
                
                # Draw the image on top of the whiteboard rectangle
                kanjiRandom.drawRect_img(frame, referenceKanji)
                
                AlreadyShowed = True
                
            #Boton de comparación de kanjis
            if scoreBtn.isOver(x, y) and not hideBoard:
                if AlreadyShowed:
                    # Capture the drawing and compare it with the Kanji
                    score = compare_kanji(canvas, ruta_kanji_random)
                    # Update the score display with the current score
                    scoreDisplay.text = f"Score: {score:.2f}"
            
            #Boton de EXIT
            if finishBtn.isOver(x,y):
                break
            
            
            
        # si solo tenemos levantado el anular
        elif upFingers[1] and not upFingers[2]:
            
            #si esta en la pizarra y no esta cerrada
            if whiteBoard.isOver(x, y) and not hideBoard:
                #print('index finger is up')
                #escribimos un circulo en su punta
                cv2.circle(frame, positions[8], brushSize, color,-1)
                #dibujamos en la pizarra
                if px == 0 and py == 0:
                    px, py = positions[8]

                #Si el color es negro de borra
                if color == (0,0,0):
                    cv2.line(canvas, (px,py), positions[8], color, eraserSize)         
                else:
                    cv2.line(canvas, (px,py), positions[8], color,brushSize)
                
                #puntos que se pintan son los del dedo anular
                px, py = positions[8]

        else:
            px, py = 0, 0
        
    # dibujamos el boton de los colores
    colorsBtn.drawRect(frame)
    cv2.rectangle(frame, (colorsBtn.x, colorsBtn.y), (colorsBtn.x +colorsBtn.w, colorsBtn.y+colorsBtn.h), (255,255,255), 2)

    # dibujamos el boton de la pizarra
    boardBtn.drawRect(frame)
    cv2.rectangle(frame, (boardBtn.x, boardBtn.y), (boardBtn.x +boardBtn.w, boardBtn.y+boardBtn.h), (255,255,255), 2)

    # dibujamos el boton para capturar la imagen dibujada
    if not hideBoard:
        scoreBtn.drawRect(frame)
        cv2.rectangle(frame, (scoreBtn.x, scoreBtn.y), (scoreBtn.x +scoreBtn.w, scoreBtn.y+scoreBtn.h), (255,255,255), 2)

    #dibujamos el boton de mostrar un kanji a dibujar
    kanjiRandomBtn.drawRect(frame)
    cv2.rectangle(frame, (kanjiRandomBtn.x, kanjiRandomBtn.y), (kanjiRandomBtn.x +kanjiRandomBtn.w, kanjiRandomBtn.y+kanjiRandomBtn.h), (255,255,255), 2)

    #dibujamos el "boton" del kanji
    if AlreadyShowed:
        kanjiRandom.drawRect_img(frame,referenceKanji)
        cv2.rectangle(frame, (kanjiRandom.x, kanjiRandom.y), (kanjiRandom.x +kanjiRandom.w, kanjiRandom.y+kanjiRandom.h), (255,255,255), 2)

    #dibujamos el boton para finalizar el programa
    finishBtn.drawRect(frame)
    cv2.rectangle(frame, (finishBtn.x, finishBtn.y), (finishBtn.x +finishBtn.w, finishBtn.y+finishBtn.h), (255,255,255), 2)

    #dibujamos el boton de la puntuación del usuario
    scoreDisplay.drawRect(frame)
    cv2.rectangle(frame, (scoreDisplay.x, scoreDisplay.y), (scoreDisplay.x +scoreDisplay.w, scoreDisplay.y+scoreDisplay.h), (255,255,255), 2)


    #dibujamos la pizarra
    if not hideBoard:       
        whiteBoard.drawRect(frame)
        ########### moviendo el dibujo a la imagen principal #########
        canvasGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(canvasGray, 20, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, imgInv)
        frame = cv2.bitwise_or(frame, canvas)


    ########## botones del color del pincel #########
    if not hideColors:
        for c in colors:
            c.drawRect(frame)
            cv2.rectangle(frame, (c.x, c.y), (c.x +c.w, c.y+c.h), (255,255,255), 2)

        clear.drawRect(frame)
        cv2.rectangle(frame, (clear.x, clear.y), (clear.x +clear.w, clear.y+clear.h), (255,255,255), 2)


    ########## boton del color del pincel ######
    #penBtn.color = color
    #penBtn.drawRect(frame)
    #cv2.rectangle(frame, (penBtn.x, penBtn.y), (penBtn.x +penBtn.w, penBtn.y+penBtn.h), (255,255,255), 2)
    if not hidePenSizes:
        for pen in pens:
            pen.drawRect(frame)
            cv2.rectangle(frame, (pen.x, pen.y), (pen.x +pen.w, pen.y+pen.h), (255,255,255), 2)


    cv2.imshow('video', frame)
    #cv2.imshow('canvas', canvas)
    k= cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

