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

import DataXeniumBreastCancer_SingleGene as DataObj
torch.manual_seed(1024)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--patient_id', help='Gene')

logger_dic = {}

args, unknown = parser.parse_known_args()

if args.patient_id:
    patient_id = int(args.patient_id)
    print('patient: ')
    print(patient_id)
else:
    patient_id = 0
    print('using default number of patient: 0')

patients = ['Stage1', 'Stage2']

tst_patient = patients[patient_id] ##'BC23810' #patient with largest number of voxels
logger_dic['test_patient'] = tst_patient



data_set = DataObj.data_set
logger_dic['data'] = data_set

logger_dic['radius'] = DataObj.radius

import sys
sys.path.insert(1, '/projects/li-lab/Yue/SpatialAnalysis/py') ##~wont work, has to start with /Users
import VTransformerLib_torch as MyVit

MyVit.if_froze_vit = False
logger_dic['if_froze_vit'] = MyVit.if_froze_vit

MyVit.learning_rate = 0.0001 ################
logger_dic['learning_rate'] = MyVit.learning_rate 

MyVit.weight_decay = 0.0001
logger_dic['weight_decay'] = MyVit.weight_decay 

MyVit.batch_size = 64 ################
logger_dic['batch_size'] = MyVit.batch_size 

logger_dic['clip'] = 1

best_epochss = {
    'POSTN':[],
    'IL7R':[],
    'FASN':[]
}
MyVit.num_epochs = 50 ################ 30
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

MyVit.mlp_head_units = [1024, 512]  # Size of the dense layers of the final classifier
logger_dic['mlp_head_units'] = ('-').join([str(x) for x in MyVit.mlp_head_units])

(MyVit.img_height, MyVit.img_width) = (MyVit.image_size, MyVit.image_size)
MyVit.image_sizes = (MyVit.img_height, MyVit.img_width)
MyVit.input_shape = (MyVit.img_height, MyVit.img_width, 3)


DataObj.X = np.array(DataObj.X).reshape((len(DataObj.X), 3, MyVit.image_size, MyVit.image_size))

#2-fold cross validation
np.random.seed(0)
random_indx = np.random.choice(len(DataObj.X), len(DataObj.X)//2)
print(random_indx.shape)

mask = np.ones(len(DataObj.X), dtype=bool)
mask[random_indx] = False

train_data = np.array(DataObj.X)[random_indx,:]
print(train_data.shape)
test_data = np.array(DataObj.X)[mask,:]

train_labels = np.array(DataObj.Y_filtered)[random_indx]
test_labels = np.array(DataObj.Y_filtered)[mask]

if tst_patient == 'Stage2':

    train_data = np.array(DataObj.X)[mask,:]
    test_data = np.array(DataObj.X)[random_indx,:]

    train_labels = np.array(DataObj.Y_filtered)[mask]
    test_labels = np.array(DataObj.Y_filtered)[random_indx]


MyVit.num_class = 1
#######remove bias of cell density, stnet norm

x_train = train_data
x_test = test_data
y_train = train_labels
y_test = test_labels

print(
    x_train.shape,
    x_test.shape,
    y_train.shape,
    y_test.shape
)



# Create iterators for the Data loaded using DataLoader module
def np_to_dataLoader(my_x, my_y):
    tensor_x = torch.Tensor(my_x) # transform to torch tensor
    tensor_y = torch.Tensor(my_y)

    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = DataLoader(my_dataset, batch_size = MyVit.batch_size) # create your dataloader
    return my_dataloader


train_data_loader = np_to_dataLoader(x_train, y_train)
test_data_loader = np_to_dataLoader(x_test, y_test)


train_data_size = x_train.shape[0]
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

        #modules.append(nn.LeakyReLU())
        #modules.append(nn.Dropout(0.5))

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

current_best_val_loss = 100000000
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
            # Update the parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)

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

                
            if test_loss < current_best_val_loss:
                current_best_test_loss = test_loss
                logger_dic['mse_test_best'] = current_best_test_loss
                logger_dic['best_epoch'] = epoch


        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 

        avg_test_loss = test_loss/test_data_size 

        epoch_end = time.time()
        print("Epoch : {:03d}, Training: Loss: {:.10f}, nttValidation : Loss : {:.10f}, Time: {:.4f}s".format(epoch, avg_train_loss, avg_test_loss, epoch_end-epoch_start))

        #print("Epoch : {:03d}, Training: Loss: {:.10f}, Accuracy: {:.4f}%, nttValidation : Loss : {:.10f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))


torch.save(model, "saved_models/ViT_pretrained_XeniumBreastCancerDataSingleGene_"+DataObj.TargetGene+"_"+data_set+"_"+tst_patient+".pt")
logger_dic['Gene'] = DataObj.TargetGene
mse_tst = avg_test_loss
print(f"Test best mse: {round(mse_tst, 2)}")
logger_dic['mse_test'] = mse_tst

pth = 'output/logs/logVitPretrained_XeniumSingleGene.json'
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





