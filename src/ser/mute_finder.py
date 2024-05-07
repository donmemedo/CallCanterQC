import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd

SR=8000


def mute_finder(path,filename,samplerate,decibels):
	y, sr = librosa.core.load(f'{path}/{filename}', sr=samplerate)
	index = librosa.effects.split(y, top_db=decibels)
	yt = librosa.effects.remix(y, index)
	sf.write(f'{path}/{filename.split(".")[0]}-{decibels}.{filename.split(".")[1]}', yt, samplerate=samplerate)

