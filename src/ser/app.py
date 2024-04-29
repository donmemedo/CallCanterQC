import os
from src.ser.inference import prediction


def emotion_detector(file):
	image = file.read()
	emotions, emotion = prediction(image_bytes=image)
	return emotions, emotion


zzz=os.listdir(f'/media/makhataei/Backups/sep/MossFormer2/MossFormer2_standalone/test_samples/GPU_outputs/')
for models in zzz:
	a = os.listdir(
		f'/media/makhataei/Backups/sep/MossFormer2/MossFormer2_standalone/test_samples/GPU_outputs/{models}')
	for files in a:
		b = os.listdir(
			f'/media/makhataei/Backups/sep/MossFormer2/MossFormer2_standalone/test_samples/GPU_outputs/{models}/{files}')
		for filess in b:
			try:
				filo = open(
					f'/media/makhataei/Backups/sep/MossFormer2/MossFormer2_standalone/test_samples/GPU_outputs/{models}/{files}/{filess}',
					'rb')
				ent, bb = emotion_detector(filo)
				print(f'{files}-{filess}: {bb}')
				with open(f'{models}.csv', 'a') as fileee:
					fileee.writelines(
						# f'{files}-{filess}, {bb},{ent[0]},{ent[1]},{ent[2]},{ent[3]},{ent[4]},{ent[5]},{ent[6]},{ent[7]}\n')
						f'{files}-{filess}, {bb},{ent[0]},{ent[1]},{ent[2]},{ent[3]},{ent[4]},{ent[5]}\n')
			except:
				pass



PATH = "/home/makhataei/Projects/CallCanterQC/datasets/ShEMO/audio"
a = os.listdir(PATH)

for file in a:
	try:
		filo = open(f"{PATH}/{file}", "rb")
		ent, bb = emotion_detector(filo)
		print(f"{file}: {bb}")
		with open("ShEMO.csv", "a") as fileee:
			fileee.writelines(
				# f"{file}, {bb},{ent[0]},{ent[1]},{ent[2]},{ent[3]},{ent[4]},{ent[5]},{ent[6]},{ent[7]}\n"
				f"{file}, {bb},{ent[0]},{ent[1]},{ent[2]},{ent[3]},{ent[4]},{ent[5]}\n"
			)
	except:
		pass