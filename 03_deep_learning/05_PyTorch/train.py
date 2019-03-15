
import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
import json
import numpy as np
from my_model import Classifier_Network
from my_helper import save_checkpoint, do_deep_learning
import argparse
from workspace_utils import active_session

parser = argparse.ArgumentParser(description='train, print training loss, validation loss, and validation accuracy as network trains and save a checkpoint')
# positional argument checkpoint filename to save the model
parser.add_argument('checkpoint_path', help='full path to checkpoint')
# network arch vgg16 or vgg13
parser.add_argument('--arch', help='vgg16 or vgg13', default='vgg16')
# learning rate
parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.001)
# epochs
parser.add_argument('--epochs', type=int, help='number of epochs', default=3)
# hidden units
parser.add_argument('--hidden_units', type=int, help='number of hidden units', default=256)
# GPU mode
parser.add_argument('--gpu', help='enable GPU mode for inference (disabled by default)', action='store_true')

args = parser.parse_args()
print(args)

root_dir = "."
data_dir = root_dir + '/flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(42),     # rotate between -42° and +42°
                                       transforms.RandomResizedCrop(224), # random resized with default scale and crop 224x224 center pixels
                                       transforms.RandomHorizontalFlip(), # use default probability p=0.5
                                       transforms.ToTensor(),             # convert to tensor - indeed, network inputs must be a tensor
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])  # normalize according to the trained network on ImageNet dataset
                                      ])

test_transforms = transforms.Compose([transforms.Resize(225),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                     ])
valid_transforms = transforms.Compose([transforms.Resize(225),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                     ])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

if args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
else:
    model = models.vgg16(pretrained=True)

# Freeze parameters of the deep neural network so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier_input_size = model.classifier[0].in_features
classifier_output_size = 102 # number of flower species
classifier_hidden_layers = [args.hidden_units]
dropout = 0.5

model.classifier = Classifier_Network(classifier_input_size, classifier_output_size, classifier_hidden_layers, dropout)
#print(model)

# train the deep network
epochs = args.epochs
print_every = 40
learning_rate = args.learning_rate
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

if args.gpu:
    device = 'cuda'
else:
    device = 'cpu'

with active_session():
    do_deep_learning(model, train_loader, valid_loader, epochs, print_every, criterion, optimizer, device)

save_checkpoint(model, train_data, classifier_input_size, classifier_output_size, classifier_hidden_layers, dropout, epochs, learning_rate, args.arch, optimizer, args.checkpoint_path)
