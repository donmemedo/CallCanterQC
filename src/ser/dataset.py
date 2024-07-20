import os
from src.ser.inference import prediction
import librosa
import numpy as np
import matplotlib.pyplot as plt
def hello_world(file):
	image = file.read()
	emotions, emotion = prediction(image_bytes=image)
	return emotions, emotion

#
# zzz=os.listdir(f'/media/makhataei/Backups/sep/MossFormer2/MossFormer2_standalone/test_samples/GPU_outputs/')
# for models in zzz:
# 	a = os.listdir(
# 		f'/media/makhataei/Backups/sep/MossFormer2/MossFormer2_standalone/test_samples/GPU_outputs/{models}')
# 	for files in a:
# 		b = os.listdir(
# 			f'/media/makhataei/Backups/sep/MossFormer2/MossFormer2_standalone/test_samples/GPU_outputs/{models}/{files}')
# 		for filess in b:
# 			try:
# 				filo = open(
# 					f'/media/makhataei/Backups/sep/MossFormer2/MossFormer2_standalone/test_samples/GPU_outputs/{models}/{files}/{filess}',
# 					'rb')
# 				ent, bb = hello_world(filo)
# 				print(f'{files}-{filess}: {bb}')
# 				with open(f'{models}.csv', 'a') as fileee:
# 					fileee.writelines(
# 						f"{files}-{filess}, {bb},{ent[0]},{ent[1]},{ent[2]},{ent[3]},{ent[4]},{ent[5]},{ent[6]},{ent[7]}\n")
# 			except:
# 				pass
#

#
#
PATH = "/home/makhataei/Downloads/11111/audio"
a = os.listdir(PATH)

for file in a:
	try:
		y,sr= librosa.core.load(f"{PATH}/{file}")
		S = librosa.feature.melspectrogram(y=y, sr=sr)
		plt.figure(frameon=False)
		librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
		plt.savefig(f'{PATH}/../pic/{file.split(".")[0]}.png', bbox_inches='tight', pad_inches=0)
		plt.close()
		with open(f'{PATH}/../ShEMO.csv', 'a') as trainee:
			trainee.writelines(
				f'{file},{file.split(".")[0]}.png,{file.split(".")[0][3]}\n')


	except:
		pass
#

# y,sr= librosa.core.load(r'/media/makhataei/Backups/sep/MossFormer2/MossFormer2_standalone/test_samples/mossformer2-librimix-2spk/model_output1CPU/index2.wav')
# S = librosa.feature.melspectrogram(y=y, sr=sr)
# plt.figure(frameon=False)
# librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
# plt.savefig('out.png', bbox_inches='tight', pad_inches=0)
