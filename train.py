
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import json
import random
import os
import argparse

# %matplotlib inline
# %config InlineBackend.figure_format = "retina"

def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    

    parser.add_argument('--arch', 
                        type=str, 
                        default="vgg19",
                        help='Architecture: vgg19')

    parser.add_argument('--save_dir', 
                        type=str,
                        default="checkpoint.pth",
                        help='Save directory')
 
    parser.add_argument('--learning_rate', 
                        type=float,
                        default=0.001,
                        help='Learning rate: 0.001')
    
    parser.add_argument('--hidden_units', 
                        type=int, 
                        default=25088,
                        help='Hidden units for DNN classifier as int : 25088')
    
    parser.add_argument('--epochs', 
                        type=int,
                        default=1,
                        help='Number of epochs: 5')

    parser.add_argument('--gpu', 
                        action="store_true", 
                        default="gpu",
                        help='Use GPU')
    # Parse args
    args = parser.parse_args()
    return args


args = arg_parser()


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'



# TODO: Define your transforms for the training, validation, and testing sets
train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

# TODO: Using the image datasets and the trainforms, define the dataloaders

datasets = [train_dataset, valid_dataset, test_dataset]

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

dataloaders = [trainloader, validloader, testloader]

# Assigning to a pretrained model

model = models.vgg19(pretrained=True)
model

# Freeze parameters so we don't backprop through them
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.gpu == "cpu":
    device = torch.device("cpu")

for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(4096, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

model.to(device)

def training(epochs):

    epochs = epochs
    steps = 0
    cuda = torch.cuda.is_available()

    if cuda:
        model.cuda()
    else:
        model.cpu()

    running_loss = 0
    accuracy = 0

    for e in range(epochs):

        train_mode = 0
        valid_mode = 1

        for mode in [train_mode, valid_mode]:   
            if mode == train_mode:
                model.train()
            else:
                model.eval()

            count = 0

            for inputs, labels in dataloaders[mode]:
                count += 1
                if cuda == True:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                # Forward
                output = model.forward(inputs)
                loss = criterion(output, labels)
                # Backward
                if mode == train_mode:
                    loss.backward()
                    optimizer.step()                

                running_loss += loss.item()
                ps = torch.exp(output).data
                equality = (labels.data == ps.max(1)[1])
                accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()

            if mode == train_mode:
                print("\nTraining Loss: {:.4f}  ".format(running_loss/count))
            else:
                print("Validation Loss: {:.4f}  ".format(running_loss/count),
                  "Accuracy: {:.4f}".format(accuracy))

            running_loss = 0


def test_validate():

    model.eval()
    accuracy = 0
    cuda = torch.cuda.is_available()

    if cuda:
        model.cuda()
    else:
        model.cpu()

    pass_count = 0

    for data in dataloaders[2]:
        pass_count += 1
        images, labels = data

        if cuda == True:
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            images, labels = Variable(images), Variable(labels)

        output = model.forward(images)
        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("\nTesting Accuracy: {:.4f}".format(accuracy/pass_count))


def save_model(epochs=args.epochs, data_dir=args.save_dir, arch=args.arch):
    training(epochs)
    test_validate()
    model.class_to_idx = datasets[0].class_to_idx

    checkpoint = {"input_size": 25088,
             "output_size": 102,
               "archi": arch,
              "epochs": epochs,
              "classifier": model.classifier,
              "optimizer": optimizer.state_dict,
              "class_to_idx": model.class_to_idx,
             "state_dict": model.state_dict()}

    torch.save(checkpoint, data_dir)
    
# Calling all the functions and saving the model    
save_model()