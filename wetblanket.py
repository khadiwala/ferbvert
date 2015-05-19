import speech_recognition as sr
import os


def detect(audio):
    try:
        transcript = r.recognize(audio)
        print("You said " + transcript)
        if 'what she said' in transcript.lower():
            os.system('say "stop"')
    except LookupError:
        print("Could not understand audio")


while True:
    with sr.Microphone() as source:
        r = sr.Recognizer()
        audio = r.listen(source)
        detect(audio)
