import os
from src.ser.inference import prediction
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def hello_world(file):
	image = file.read()
	emotions, emotion = prediction(image_bytes=image)
	return emotions, emotion


PATH = "/media/makhataei/Backups/555"
# train_df = pd.read_csv("/media/makhataei/Backups/555/data_train.csv", on_bad_lines='skip')
train_df = pd.read_csv("/home/makhataei/Projects/CallCanterQC/src/ser/tester.csv", on_bad_lines='skip')
a=train_df.values
# a = os.listdir(PATH)

for file in a:
	try:
		# y,sr= librosa.core.load(f"{PATH}/{file[0]}")
		filo = open(f"{PATH}/{file[0]}",'rb')
		ent, bb = hello_world(filo)
		with open(f'{PATH}/4123.csv', 'a') as fileee:
			fileee.writelines(
				# f'{files}-{filess}, {bb},{ent[0]},{ent[1]},{ent[2]},{ent[3]},{ent[4]},{ent[5]},{ent[6]},{ent[7]}\n')
				f'{file[0]},{file[1]},{ent[0]},{ent[1]},{ent[2]},{ent[3]}\n')
	except:
		pass

train_df = pd.read_csv("/media/makhataei/Backups/555/data_test.csv", on_bad_lines='skip')
a=train_df.values
# a = os.listdir(PATH)

for file in a:
	try:
		# y,sr= librosa.core.load(f"{PATH}/{file[0]}")
		filo = open(f"{PATH}/{file[0]}",'rb')
		ent, bb = hello_world(filo)
		with open(f'{PATH}/4201.csv', 'a') as fileee:
			fileee.writelines(
				# f'{files}-{filess}, {bb},{ent[0]},{ent[1]},{ent[2]},{ent[3]},{ent[4]},{ent[5]},{ent[6]},{ent[7]}\n')
				f'{file[0]},{file[1]},{ent[0]},{ent[1]},{ent[2]},{ent[3]}\n')
	except:
		pass
