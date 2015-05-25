from twss import get_classifier
import speech_recognition as sr
import os

clf = get_classifier()
r = sr.Recognizer()


def detect(audio):
    try:
        transcript = r.recognize(audio)
        print("You said " + transcript)
        if clf(transcript.lower()):
            os.system('say "thats what she said"')
    except LookupError:
        print("Could not understand audio")

with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)

r.pause_threshold = 0.5

print "starting to listen"
while True:
    with sr.Microphone() as source:
        audio = r.listen(source)
        detect(audio)
