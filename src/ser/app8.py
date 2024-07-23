import os
from src.ser.inference import prediction
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PATH = "/media/makhataei/Backups/555/ravdess/Audio_Speech_Actors_01-24"
# train_df = pd.read_csv("/media/makhataei/Backups/555/data_train.csv", on_bad_lines='skip')
# a=train_df.values
a = os.listdir(PATH)
zinger = {'01': 'neutral', '02' : 'calm', '03' : 'happy', '04' : 'sad', '05' : 'angry', '06' : 'fearful', '07' : 'disgust', '08' : 'surprised'}
for actor in a:
    b = os.listdir(f"{PATH}/{actor}")
    for file in b:
        with open(f'{PATH}/../tesr.csv', 'a') as fileee:
            fileee.writelines(
                # f'{files}-{filess}, {bb},{ent[0]},{ent[1]},{ent[2]},{ent[3]},{ent[4]},{ent[5]},{ent[6]},{ent[7]}\n')
                f'{PATH}/{actor}/{file},{zinger[file.split("-")[2]]}\n')

zam = f'{PATH}/../tesr.csv'
PATH = "/media/makhataei/Backups/1/ShEMO/1"
a = os.listdir("/media/makhataei/Backups/1/ShEMO/1")
zinger = {'A':'angry', 'H':'happy', 'N':'neutral', 'S':'sad', 'W':'suprised', 'F':'fearful'}

for file in a:
    with open(zam, 'a') as fileee:
        fileee.writelines(
        f'{PATH}/{file},{zinger[file[3]]}\n')
