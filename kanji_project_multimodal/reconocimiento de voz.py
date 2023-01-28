import os
from gtts import gTTS
import speech_recognition as sr
import threading

# Inicializar el reconocedor de voz
r = sr.Recognizer(lan = 'es-ES')

# Inicializar el micrófono
mic = sr.Microphone()

def speak(text):
    tts = gTTS(text, lang='es')
    tts.save("output.mp3")
    os.system("start output.mp3")


def voice_command_thread():
    while True:
        # Escuchar al usuario
        with mic as source:
            audio = r.listen(source)

        try:
            command = r.recognize_google(audio).lower()
            print("Comando reconocido: " + command)

            if command == "asistente":
                speak("¿Qué desea?")

                with mic as source:
                    audio = r.listen(source)

                try:
                    command = r.recognize_google(audio).lower()
                    print("Comando reconocido: " + command)

                except sr.UnknownValueError:
                    speak("No se pudo reconocer el comando de voz")

        except sr.UnknownValueError:
            print("No se pudo reconocer el comando de voz")

voice_thread = threading.Thread(target=voice_command_thread)
voice_thread.start()
