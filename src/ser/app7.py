import os
from src.ser.inference import prediction
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.ser.pred import predict_ser


def hello_world(file):
    image = file.read()
    emotions, emotion = prediction(image_bytes=image)
    return emotions, emotion


PATH = "/media/makhataei/Backups/555"
# train_df = pd.read_csv("/media/makhataei/Backups/555/data_train.csv", on_bad_lines='skip')
train_df = pd.read_csv("/home/makhataei/Projects/CallCanterQC/src/ser/tester.csv", on_bad_lines='skip')
a = train_df.values
# a = os.listdir(PATH)

for file in a:
    try:
        # y,sr= librosa.core.load(f"{PATH}/{file[0]}")
        ent = predict_ser(f"{PATH}/{file[0]}")
        zinger = {"label":None,"score":0}
        for zico in ent:
            if zico["score"]>zinger["score"]:
                zinger["label"]=zico["label"]
                zinger["score"]=zico["score"]

        # filo = open(f"{PATH}/{file[0]}",'rb')
        # ent, bb = hello_world(filo)
        with open(f'{PATH}/ken.csv', 'a') as fileee:
            fileee.writelines(
                # f'{files}-{filess}, {bb},{ent[0]},{ent[1]},{ent[2]},{ent[3]},{ent[4]},{ent[5]},{ent[6]},{ent[7]}\n')
                f'{file[0]},{file[1]},{zinger["label"]},{ent[0]["label"]}: {int(ent[0]["score"]*10000)/100},{ent[1]["label"]}: {int(ent[1]["score"]*10000)/100},{ent[2]["label"]}: {int(ent[2]["score"]*10000)/100},{ent[3]["label"]}: {int(ent[3]["score"]*10000)/100},{ent[4]["label"]}: {int(ent[4]["score"]*10000)/100}\n')
    except:
        pass

train_df = pd.read_csv("/media/makhataei/Backups/555/data_test.csv", on_bad_lines='skip')
a = train_df.values
# a = os.listdir(PATH)

for file in a:
    try:
        # y,sr= librosa.core.load(f"{PATH}/{file[0]}")
        filo = open(f"{PATH}/{file[0]}", 'rb')
        ent, bb = hello_world(filo)
        with open(f'{PATH}/4201.csv', 'a') as fileee:
            fileee.writelines(
                # f'{files}-{filess}, {bb},{ent[0]},{ent[1]},{ent[2]},{ent[3]},{ent[4]},{ent[5]},{ent[6]},{ent[7]}\n')
                f'{file[0]},{file[1]},{ent[0]},{ent[1]},{ent[2]},{ent[3]}\n')
    except:
        pass
