import pyttsx3
tts = pyttsx3.init()
tts.setProperty('rate',150)
tts.setProperty('volume',0.9)
tts.say("Mouse Detected")
tts.runAndWait()