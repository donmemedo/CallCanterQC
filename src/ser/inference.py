import torch
from src.ser.commons import get_model, get_tensor

# # ToDo: 6 Grooups
# class_names = [
#     "angry",
#     "fearful",
#     "happy",
#     "neutral",
#     "sad",
#     "surprised",
# ]
# model = get_model()
#
#
# def prediction(image_bytes):
#     tensor = get_tensor(image_bytes)
#     outputs = model(tensor)
#     _, zz = torch.sort(outputs, descending=True)
#     _, prediction = outputs.max(1)
#     emotions = []
#     for z in zz.tolist()[0]:
#         emotions.append(class_names[z])
#         # print("old")
#     category = prediction.item()
#     emotion = class_names[category]
#
#     return emotions, emotion


# ToDo: 4 Grooups
class_names = [
    "angry",
    "happy",
    "neutral",
    "sad",
]
model = get_model()


def prediction(image_bytes):
    tensor = get_tensor(image_bytes)
    outputs = model(tensor)
    _, zz = torch.sort(outputs, descending=True)
    _, prediction = outputs.max(1)
    emotions = []
    for z in zz.tolist()[0]:
        emotions.append(class_names[z])
        # print("old")
    category = prediction.item()
    emotion = class_names[category]

    return emotions, emotion
