import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import time

torch.set_default_tensor_type(torch.cuda.FloatTensor)
# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

data_dir="/home/makhataei/Projects/CallCanterQC/datasets/ShEMO"
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
val_data = datasets.ImageFolder(data_dir + '/val', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=train_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True,generator=torch.Generator(device='cuda'))
testloader = torch.utils.data.DataLoader(test_data, batch_size=64,generator=torch.Generator(device='cuda'))
valloader=torch.utils.data.DataLoader(val_data, batch_size=64,shuffle=True,generator=torch.Generator(device='cuda'))

images,labels=next(iter(trainloader))

image=images[0].numpy()
label=labels[0]
model =models.densenet121(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(1024,512),nn.LeakyReLU(),nn.Linear(512,6))
criterion = nn.CrossEntropyLoss()
OLD_Model="SER_densenet122.pt"
NEW_Model="SER_densenet123.pt"

def train(n_epochs, trainloader, testloader, resnet, optimizer, criterion):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    running_loss = 0

    for epoch in range(n_epochs):

        for inputs, labels in trainloader:
            # Move input and label tensors to the default device
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            start = time.time()
            logps = resnet(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        resnet.eval()
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                logps = resnet(inputs)
                batch_loss = criterion(logps, labels)
                valid_loss += batch_loss.item()

                # Calculate accuracy

                top_p, top_class = logps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            if valid_loss <= valid_loss_min:
                print("Validation loss decreased  Saving model")
                torch.save(resnet.state_dict(), NEW_Model)
                valid_loss_min = valid_loss

            print(f"Device = cuda; Time per batch: {(time.time() - start):.3f} seconds")
            print(f"Epoch /{epoch}.. "
                  f"Train loss: {running_loss / len(trainloader):.3f}.. "
                  f"Val loss: {valid_loss / len(testloader):.3f}.. "
                  f"Val accuracy: {accuracy / len(testloader):.3f}")
            running_loss = 0
            resnet.train()

for param in model.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=1e-8)
model.load_state_dict(torch.load(OLD_Model), strict=False)
train(3000,trainloader,valloader, model, optimizer, criterion)
model.load_state_dict(torch.load(NEW_Model))

valid_loss = 0
accuracy = 0
with torch.no_grad():
    model.eval()
    for images, labels in testloader:
        images, lables = images.cuda(), labels.cuda()
        logps = model(images)
        batch_loss = criterion(logps, labels)
        valid_loss += batch_loss.item()
        top_p, top_class = logps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
print("LOSS-" + str(valid_loss / len(testloader)))

print("ACCURACY-" + str(accuracy * 100 / len(testloader)))

emotions=test_data.classes
classes=test_data.classes
# emotions

# track test loss
test_loss = 0.0
batch_size=64
class_correct = list(0. for i in range(6))
class_total = list(0. for i in range(6))

# class_correct = list(0. for i in range(8))
# class_total = list(0. for i in range(8))


with torch.no_grad():
    model.eval()
    # iterate over test data
    for batch_idx, (data, target) in enumerate(testloader):
        # move tensors to GPU if CUDA is available
        data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy()) # if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss/len(testloader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(6):
    # for i in range(8):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
