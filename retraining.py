from __future__ import print_function, division
from utilities import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from multiprocessing import freeze_support

# Parameters
test_split = 0.2
batch_size = 1
class_names = ['melanoma']

if __name__ == '__main__':
    freeze_support()

    csv_file = 'data/ISIC_2019_Training_GroundTruth.csv'
    df = pd.read_csv(csv_file)



    transformed_dataset = MelanomaDataset(csv_file='data/ISIC_2019_Training_GroundTruth.csv',
                                               root_dir='data/ISIC_2019_Training_Input/',
                                               transform=transforms.Compose([
                                                   Rescale(256),
                                                   ToTensor()
                                               ]))

    val_count = int(len(transformed_dataset)*test_split)*batch_size
    train_count = int(len(transformed_dataset)-val_count)*batch_size

    train_set, val_set = torch.utils.data.random_split(transformed_dataset, [train_count, val_count])

    train_loader = DataLoader(train_set, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=True, num_workers=1)


    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': train_count, 'val': val_count}



    # Training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, device,
                           num_epochs=25)