from handTracker import *
import cv2
import mediapipe as mp
import numpy as np
import random
import os
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


def compare_kanji_v2(drawing, kanji_path):
    # Cargar la imagen del kanji
    kanji = cv2.imread(kanji_path, cv2.IMREAD_GRAYSCALE)

    # Convertir lo pintado a una escala de grises
    drawing_gray = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    height_, width_ = drawing_gray.shape[:2]
    scale_factor_x = 640 / width_
    scale_factor_y = 640 / height_
    drawing_gray = cv2.resize(drawing_gray,None,fx=scale_factor_x, fy=scale_factor_y)
    
    # Usar el algoritmo ORB (Oriented FAST and Rotated BRIEF) 
    orb = cv2.ORB_create()
    keypoints1, descriptor1 = orb.detectAndCompute(kanji, None)
    keypoints2, descriptor2 = orb.detectAndCompute(drawing_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if descriptor1 is not None and descriptor2 is not None:
        matches = bf.match(descriptor1, descriptor2)
    else:
        return 0.00

    # Ordenar los matches por distancias
    matches = sorted(matches, key=lambda x: x.distance)

    # Calcular el numero de matches (es decir, los de menor distancia a los puntos que necesitamos)
    num_good_matches = int(len(matches))
    good_matches = matches[:num_good_matches]

    # Calcular la media
    if num_good_matches:
        avg_distance = sum(m.distance for m in good_matches) / num_good_matches
    else:
        return 0.00

    # Normalizarlo entre 0 y 100
    score = 100 - avg_distance * 100 / (2 ** 8 - 1)
    return score

#METODO PARA MOSTRAR EL KANJI RANDOMIZADO
def loadReferenceKanji(folder):
    # Select a random Kanji image from the folder
    kanjiFilename = random.choice(os.listdir(folder))
    kanjiPath = os.path.join(folder, kanjiFilename)
    kanji = cv2.imread(kanjiPath)
    
    return kanji, kanjiPath

#Inicializamos el detector de manos
#detector = HandTracker(detectionCon=0.8)
detector = HandTracker()

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
scoreBtn = ColorRect(1075, 100, 175, 100, (0,0,255), 'SCORE')

#Botón de mostrar un kanji a dibujar
kanjiRandomBtn = ColorRect(1075, 300, 175, 100, (0,0,0), 'KANJI')

#Botón de mostrar un kanji a dibujar
width, height = 1280, 720
button_width = int(width * 0.5)
button_height = int(height * 0.5)
button_x = int((width - button_width) / 2)
button_y = int((height - button_height) / 1.5)
kanjiRandom = ColorRect(button_x, button_y, button_width, button_height, (255,255,255), alpha = 0.6)

#Botón de mostrar un kanji a dibujar
finishBtn = ColorRect(1075, 500, 175, 100, (0,0,0), 'EXIT')

#Botón que muestra la puntuación
scoreDisplay = ColorRect(1075, 0, 175, 100, (255,255,255), 'Score: 0')

# Boton para obtener la pizarra
boardBtn = ColorRect(50, 0, 100, 100, (255,255,0), 'Board')

# Pizarra
whiteBoard = ColorRect(50, 120, 1020, 580, (255,255,255),alpha = 0.2)

# Tamaños del pincel
pens = []
for i, penSize in enumerate(range(5,25,5)):
    pens.append(ColorRect(1100,50+100*i,100,100, (50,50,50), str(penSize)))

coolingCounter = 20
hideBoard = False
hideColors = True
hidePenSizes = True
AlreadyShowed = False





# ----------------------------- RECONOCIMIENTO DE VOZ ----------------------------------------------
from gtts import gTTS
import speech_recognition as sr

# Inicializar el reconocedor de voz
r = sr.Recognizer()

# Inicializar el micrófono
mic = sr.Microphone()

#Inicializar valores de la aplicación principal
brushSize_lock = threading.Lock()
color_lock = threading.Lock()
nuevoKanji_lock = threading.Lock()
Puntuación_lock = threading.Lock()


llamar_asistente = {"hola asistente", "quiero llamar al asistente", "hablar con el asistente", "asistente", " asistente"}
fin_programa = {"finalizar programa","finalizar el programa", "me he cansado", "me he cansao", "fin de la partida", "acabar el programa", ""}
respuestas_si = {"sí", "si", "dale", "claro","sí, estoy seguro", "sí estoy seguro", "si estoy seguro"}
respuestas_no = {"no", "qué va", "no no", "para nada", "quiero continuar"}
saludar_asistente = {"hola asistente", "¿sigues ahí?", "¿me escuchas?", "sigues ahí", "me escuchas"}

import sounddevice as sd
from pydub import AudioSegment

def speak(text):
    # Crear el archivo de audio
    tts = gTTS(text, lang='es')
    tts.save("output.mp3")
    # Abrir el archivo mp3 con pydub
    audio = AudioSegment.from_file("output.mp3", format="mp3")
    # Convertir el archivo mp3 a array de audio
    audio_array = np.array(audio.get_array_of_samples())
    # Reproducir el audio
    sd.play(audio_array)
    # Esperar a que termine la reproducción
    sd.wait()
    # Eliminar el archivo mp3
    os.remove("output.mp3")




def voice_command_thread():

    global brushSize, color, referenceKanji, ruta_kanji_random, AlreadyShowed, canvas, score, kanjiRandomBtn, scoreDisplay, ruta_kanji_random, referenceKanji
    while True:
        # Escuchar al usuario
        with mic as source:
            audio = r.listen(source)

        try:
            command = r.recognize_google(audio, language = "es-ES").lower()
            print("Comando reconocido: " + command)

            if command in llamar_asistente:
                speak("¿Qué desea?")
                no_escuchado_avisado = True
                while True:
                    with mic as source:
                        audio = r.listen(source, timeout = None, phrase_time_limit = 100)
                    try:
                        command = r.recognize_google(audio, language = "es-ES").lower()
                        print("Comando reconocido: " + command)

                        if "pincel más grande" in command:
                            brushSize_lock.acquire()
                            brushSize += 10
                            brushSize_lock.release()
                            speak("Tamaño del pincel aumentado a:" + str(brushSize))
                            break

                        elif "pincel más pequeño" in command:
                            brushSize_lock.acquire()
                            brushSize -= 10
                            brushSize_lock.release()
                            speak("Tamaño del pincel disminuido a:" + str(brushSize))
                            break
                        
                        if "cambiar color pincel" in command:
                            speak("¿Qué color desea poner?")
                            with mic as source:
                                audio = r.listen(source)

                            try:
                                color_ = r.recognize_google(audio).lower()
                                print("Comando reconocido: " + color)

                                # Si el color reconocido es uno de los permitidos, cambia el color del pincel
                                if color_ in ["rojo", "azul", "verde", "amarillo", "borrador"]:
                                    if color_ == "rojo":
                                        color_aux = (45,0,255)
                                    elif color_ == "azul":
                                        color_aux = (255,0,171)
                                    elif color_ == "verde":
                                        color_aux = (0,210,70)
                                    elif color_ == "amarillo":
                                        color_aux = (0,255,255)
                                    elif color_ == "borrador":
                                        color_aux = (0,0,0)

                                    color_lock.acquire()
                                    color = color_aux
                                    color_lock.release()

                                    speak("Color del pincel cambiado a " + color)

                                else:
                                    speak("Color no permitido")
                                break

                            except sr.UnknownValueError:
                                speak("No se pudo reconocer el comando de voz")
                                break

                        if command == "muéstrame un carácter chino":
                            speak("Obteniendo un Kanji aleatorio")
                            nuevoKanji_lock.acquire()
                            
                            # Load a new reference Kanji and store it in the global variable
                            referenceKanji, ruta_kanji_random = loadReferenceKanji('imagenes')
                        
                            # Pintar el kanji superpuesto en la pizarra
                            kanjiRandom.drawRect_img(frame, referenceKanji)
                            AlreadyShowed = True
                            nuevoKanji_lock.release()
                            break

                        if command == "ya he terminado":
                            speak("Obteniendo la puntuación")
                            Puntuación_lock.acquire()
                            # Capturar lo pintado y realizar un sistema de puntuación
                            score = compare_kanji_v2(canvas, ruta_kanji_random)
                            # Actualizar la puntuación
                            scoreDisplay.text = f"Score: {score:.2f}"

                            #Reiniciar la pizarra
                            clear.alpha = 0
                            canvas = np.zeros((720,1280,3), np.uint8)


                            kanjiRandomBtn = ColorRect(1075, 300, 175, 100, (0,0,0), 'NEXT KANJI')
                            AlreadyShowed = False
                            Puntuación_lock.release()

                            break
                        
                        if command in fin_programa:
                            speak("¿Está seguro?")
                            no_escuchado_avisado = True
                            while True:
                                with mic as source:
                                    audio = r.listen(source, timeout = None, phrase_time_limit = 100)
                                command = r.recognize_google(audio, language = "es-ES").lower()
                                print("Comando reconocido: " + command)
                                if command in respuestas_si:
                                    speak("Perfecto, ¡Muchas gracias por jugar! Hasta la próxima")
                                    os._exit(0)
                                elif command in respuestas_no:
                                    speak("Vale, si me necesitas vuelve a llamarme")
                                    break
                                else:
                                    speak("No he podido escuchar nada, llámame de nuevo")

                            break
                        
                        if command in saludar_asistente:
                            speak("Sigo esperando una orden, ¿Qué desea?")

                    except sr.UnknownValueError:
                        if no_escuchado_avisado == True:
                            speak("No he podido escuchar ningún comando, hable de nuevo")
                            no_escuchado_avisado = False
                        elif no_escuchado_avisado == False:
                            print("No he podido escuchar nada")


        except sr.UnknownValueError:
            print("No se pudo reconocer el comando de voz. Inténtalo de nuevo")

voice_thread = threading.Thread(target=voice_command_thread)
voice_thread.start()

#-------------------- BUCLE DE LA APLICACIÓN -------------------------------

#Bucle de la aplicación
while True:

    if coolingCounter:
        coolingCounter -=1

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
                
                # Pintar el kanji superpuesto en la pizarra
                kanjiRandom.drawRect_img(frame, referenceKanji)
                
                AlreadyShowed = True
                
            #Boton de comparación de kanjis
            if scoreBtn.isOver(x, y) and not hideBoard and AlreadyShowed:
                Puntuación_lock.acquire()
                # Capturar lo pintado y realizar un sistema de puntuación
                score = compare_kanji_v2(canvas, ruta_kanji_random)
                # Actualizar la puntuación
                scoreDisplay.text = f"Score: {score:.2f}"

                #Reiniciar la pizarra
                clear.alpha = 0
                canvas = np.zeros((720,1280,3), np.uint8)


                kanjiRandomBtn = kanjiRandomBtn = ColorRect(1075, 300, 175, 100, (0,0,0), 'NEXT KANJI')
                AlreadyShowed = False
                Puntuación_lock.release()


            #Boton de EXIT
            if finishBtn.isOver(x,y):
                break
            
            
            
        # si solo tenemos levantado el anular
        elif upFingers[1] and not upFingers[2]:
            
            #si esta en la pizarra y no esta cerrada
            if whiteBoard.isOver(x, y) and not hideBoard:

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

