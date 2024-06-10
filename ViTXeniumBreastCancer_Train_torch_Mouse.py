###############needs conda env: torch_env

import os
import numpy as np
from sklearn.model_selection import train_test_split

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import pandas as pd

##import DataXeniumBreastCancer_test as DataObj
import DataXeniumBreastCancer_Mouse as DataObj
DataObj.ifNorm = True
DataObj.X, DataObj.Y_filtered, DataObj.voxel_ids = DataObj.load_data(DataObj.data_set, DataObj.ifNorm)


torch.manual_seed(1024)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training using:')
print(device)

torch.backends.cudnn.enabled = False



logger_dic = {}


data_set = DataObj.data_set
logger_dic['data'] = data_set

logger_dic['radius'] = DataObj.radius
logger_dic['ifNorm'] = DataObj.ifNorm


import sys
sys.path.insert(1, '/projects/li-lab/Yue/SpatialAnalysis/py') ##~wont work, has to start with /Users
import VTransformerLib_torch as MyVit

MyVit.if_froze_vit = False
logger_dic['if_froze_vit'] = MyVit.if_froze_vit

MyVit.learning_rate = 0.0001 ################ best 0.0001
logger_dic['learning_rate'] = MyVit.learning_rate 

MyVit.weight_decay = 0.0001
logger_dic['weight_decay'] = MyVit.weight_decay 

MyVit.batch_size = 64 ################
logger_dic['batch_size'] = MyVit.batch_size 

#epochss = [75,57] #for different models
#MyVit.num_epochs = epochss[int(DataObj.args.data_id)] ################ 30

MyVit.num_epochs = 500 ################ 30

logger_dic['num_epochs'] = MyVit.num_epochs

MyVit.image_size = DataObj.image_size  # We'll resize input images to this size
logger_dic['image_size'] = MyVit.image_size

MyVit.patch_size = 32  # Size of the patches to be extract from the input images 
logger_dic['patch_size'] = MyVit.patch_size

#MyVit.projection_dim = 512
#logger_dic['projection_dim'] = MyVit.projection_dim

#MyVit.num_heads = 16 ##affect training accuracy a lot
#logger_dic['num_heads'] = MyVit.num_heads

#MyVit.transformer_units = [
#    MyVit.projection_dim * 2,
#   MyVit.projection_dim,
#]  # Size of the transformer layers
#MyVit.transformer_layers = 1
#logger_dic['transformer_layers'] = MyVit.transformer_layers

MyVit.mlp_head_units = [4096,512]  # Size of the dense layers of the final classifier
logger_dic['mlp_head_units'] = ('-').join([str(x) for x in MyVit.mlp_head_units])

print(logger_dic)

(MyVit.img_height, MyVit.img_width) = (MyVit.image_size, MyVit.image_size)
MyVit.image_sizes = (MyVit.img_height, MyVit.img_width)
MyVit.input_shape = (MyVit.img_height, MyVit.img_width, 3)


DataObj.X = np.array(DataObj.X).reshape((len(DataObj.X), 3, MyVit.image_size, MyVit.image_size))

train_data, test_data, train_labels, test_labels = train_test_split(DataObj.X, DataObj.Y_filtered, test_size = 0.2 , random_state=1024)


MyVit.num_class = train_labels.shape[1]
#######remove bias of cell density, stnet norm
from sklearn.preprocessing import normalize

#train_labels_norm = np.log(normalize(train_labels+1, axis=1, norm='l1'))
#test_labels_norm = np.log(normalize(test_labels+1, axis=1, norm='l1'))

#train_labels_norm = np.log(train_labels+1)
#test_labels_norm = np.log(test_labels+1)

x_train = np.array(train_data)
x_test = np.array(test_data)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

print(
    x_train.shape,
    x_test.shape,
    y_train.shape,
    y_test.shape
)

x_train_ids = range(x_train.shape[0])
x_train_ids, x_valid_ids, y_train, y_valid = train_test_split(x_train_ids, y_train, test_size = 0.2 , random_state=1024)

x_valid = x_train[x_valid_ids,:]##oom issues
x_train = x_train[x_train_ids,:]
#x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2 , random_state=1024)

#train_scaler, x_train = MyVit.norm_img(x_train, scaler = None)
#_, x_valid = MyVit.norm_img(x_valid, train_scaler)
#_, x_test = MyVit.norm_img(x_test, train_scaler)


# Create iterators for the Data loaded using DataLoader module
def np_to_dataLoader(my_x, my_y):
    tensor_x = torch.Tensor(my_x) # transform to torch tensor
    tensor_y = torch.Tensor(my_y)

    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = DataLoader(my_dataset, batch_size = MyVit.batch_size) # create your dataloader
    return my_dataloader


train_data_loader = np_to_dataLoader(x_train, y_train)
valid_data_loader = np_to_dataLoader(x_valid, y_valid)
test_data_loader = np_to_dataLoader(x_test, y_test)


train_data_size = x_train.shape[0]
valid_data_size = x_valid.shape[0]
test_data_size = x_test.shape[0]

import sys
sys.path.insert(1, '/projects/li-lab/Yue/SpatialAnalysis/py/PyTorchPretrainedViT') ##~wont work, has to start with /Users
from PyTorchPretrainedViT.pytorch_pretrained_vit import ViT
model = ViT('L_32_imagenet1k', pretrained=True, image_size=MyVit.image_size, patches = MyVit.patch_size)

# Freeze model parameters

for param in model.parameters():

    param.requires_grad = (not MyVit.if_froze_vit)

# Change the final layer of pretrained Vit Model for Transfer Learning
fc_inputs = model.fc.in_features
print(fc_inputs)

modules = []

if len(MyVit.mlp_head_units) > 1:
    for i in range(len(MyVit.mlp_head_units)):

        if i == 0:
            modules.append(nn.Linear(fc_inputs, MyVit.mlp_head_units[i]))

        else:
            modules.append(nn.Linear(MyVit.mlp_head_units[i-1], MyVit.mlp_head_units[i]))

        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.5))

        if i == (len(MyVit.mlp_head_units)-1):
            modules.append(nn.Linear(MyVit.mlp_head_units[i], MyVit.num_class))

        
else:
    modules.append(nn.Linear(MyVit.mlp_head_units[0], MyVit.num_class))


model.fc = nn.Sequential(*modules)

model = model.to(device)

loss_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=MyVit.learning_rate, weight_decay = MyVit.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

from transformers import ViTImageProcessor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image_mean, image_std = processor.image_mean, processor.image_std

normalize = transforms.Normalize(mean=image_mean, std=image_std)

image_aug = torch.nn.Sequential(
    #transforms.CenterCrop(MyVit.image_size-20),
    transforms.RandomRotation(degrees=(0, 180)),
    #transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    normalize,
    transforms.Resize(MyVit.image_size)
)

current_best_val_loss = 1000000000
for epoch in range(MyVit.num_epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, MyVit.num_epochs))
        # Set to training mode
        model.train()
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        #train_acc = 0.0
        valid_loss = 0.0
        #valid_acc = 0.0
        test_loss = 0.0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = image_aug(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Clean existing gradients
            optimizer.zero_grad()
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            # Compute loss
            loss = loss_criterion(outputs, labels)
            # Backpropagate the gradients
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # Update the parameters
            optimizer.step()
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            # Compute the accuracy
            #ret, predictions = torch.max(outputs.data, 1)
            #correct_counts = predictions.eq(labels.data.view_as(predictions))
            # Convert correct_counts to float and then compute the mean
            #acc = torch.mean(correct_counts.type(torch.FloatTensor))
            # Compute total accuracy in the whole batch and add to train_acc
            #train_acc += acc.item() * inputs.size(0)
            #print("Batch number: {:03d}, Training: Loss: {:.4f}".format(i, loss.item()))

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: SGD lr %.10f -> %.10f" % (epoch, before_lr, after_lr))

            # Validation - No gradient tracking needed
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()
            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)
                # Compute loss
                loss = loss_criterion(outputs, labels)
                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)
                # Calculate validation accuracy
                #ret, predictions = torch.max(outputs.data, 1)
                #correct_counts = predictions.eq(labels.data.view_as(predictions))
                # Convert correct_counts to float and then compute the mean
                #acc = torch.mean(correct_counts.type(torch.FloatTensor))
                # Compute total accuracy in the whole batch and add to valid_acc
                #valid_acc += acc.item() * inputs.size(0)
                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}".format(j, loss.item()))
            


            if valid_loss < current_best_val_loss:
                current_best_val_loss = valid_loss
                logger_dic['mse_valid_best'] = current_best_val_loss
                logger_dic['best_epoch'] = epoch
                torch.save(model, "saved_models/ViT_pretrained_"+data_set+"_"+str(len(DataObj.X))+".pt")

                

            
            for j, (inputs, labels) in enumerate(test_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)
                # Compute loss
                loss = loss_criterion(outputs, labels)
                # Compute the total loss for the batch and add it to valid_loss
                test_loss += loss.item() * inputs.size(0)
                # Calculate validation accuracy
                #ret, predictions = torch.max(outputs.data, 1)
                #correct_counts = predictions.eq(labels.data.view_as(predictions))
                # Convert correct_counts to float and then compute the mean
                #acc = torch.mean(correct_counts.type(torch.FloatTensor))
                # Compute total accuracy in the whole batch and add to valid_acc
                #valid_acc += acc.item() * inputs.size(0)
                #print("Test Batch number: {:03d}, Test: Loss: {:.4f}".format(j, loss.item()))


        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 
        #avg_train_acc = train_acc/float(train_data_size)
        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/valid_data_size 
        #avg_valid_acc = valid_acc/float(valid_data_size)

        avg_test_loss = test_loss/test_data_size 

        epoch_end = time.time()
        print("Epoch : {:03d}, Training: Loss: {:.10f}, nttValidation : Loss : {:.10f}, Time: {:.4f}s".format(epoch, avg_train_loss, avg_valid_loss, epoch_end-epoch_start))

        #print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, nttValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))


torch.save(model, "saved_models/ViT_pretrained_"+data_set+"_"+str(len(DataObj.X))+"_final.pt")

mse_tst = avg_test_loss
print(f"Test best mse: {round(mse_tst, 10)}")
logger_dic['mse_test'] = mse_tst
logger_dic['mse_valid'] = avg_valid_loss

pth = 'output/logs/logVitPretrained_Xenium_'+data_set+'.json'
import json

if not os.path.isfile(pth):
    with open(pth, 'w') as outfile:
        json.dump([], outfile)

with open(pth, 'r') as openfile:
 
    # Reading from json file
    json_list = json.load(openfile)

json_list.append(logger_dic)

with open(pth, 'w') as outfile:
    json.dump(json_list, outfile)





