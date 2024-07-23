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
train_df = pd.read_csv("/media/makhataei/Backups/555/data_train.csv", on_bad_lines='skip')
a=train_df.values
# a = os.listdir(PATH)

for file in a:
	try:
		y,sr= librosa.core.load(f"{PATH}/{file[0]}")
		S = librosa.feature.melspectrogram(y=y, sr=sr)
		plt.figure(frameon=False)
		librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
		plt.savefig(f'{PATH}/{file[1]}/{file[0].split(".")[0].split("/")[1]}.png', bbox_inches='tight', pad_inches=0)
		plt.close()
	except:
		pass
