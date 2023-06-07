import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from PaT import PanTompkinsDetector

def Read(file):

	f = open(file, 'r')
	lines = f.readlines()
	data = []
	for line in lines:
		data.append(eval(line))

	return data

data = []

for i in range(10):
	file = r'./data/' + str(i + 1) + '.txt'
	data.append(Read(file))

data = np.array(data)

# 检测 QRS

signal = data[7]
fs = 200
detector = PanTompkinsDetector(fs)
qrs_peaks = detector.detect_qrs(signal)