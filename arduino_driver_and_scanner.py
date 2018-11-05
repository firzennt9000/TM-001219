import cv2
import numpy as np
import time
import sys
import serial
import os

arduino = serial.Serial('/dev/cu.usbmodem1421', 57600)

target_path = sys.argv[1]

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
path = 'data/'
for i in range(6):
	ret, frame = cap.read()
	if ret:
		cv2.imwrite(path + str(i) + '.png', frame)
	arduino.write('g'.encode())
	time.sleep(1.200)

arg = ''
for i in range(6):
	arg += ' ' + path + str(i) + '.png'
print(arg)
cap.release()
arduino.close()
os.system('python proof_of_concept.py ' + target_path + arg)

