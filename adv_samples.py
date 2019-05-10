import os, time, random, copy, csv
import numpy as np
from PIL import Image
import pickle
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import foolbox
from Adversary import AdvDataset, Generate_Adversarial_Samples
from train_sdr import *
from data_helpers import *
from general_helpers import *

hiddens = [28*28, 100, 100, 100]
num_classes = 10
model_sdr_gradient_update = SDR_MLP(hiddens, num_classes)

model_path = os.path.join('saved_models', 'together', 'model_epoch_15_ckpt_test.pt')
model_sdr_gradient_update.load_state_dict(torch.load(model_path))
model_sdr_gradient_update = model_sdr_gradient_update.cuda()
print ('Model is loaded\n')

print('Testing model on regular images...')

trainset, testset, splited = split_MNIST(train_val_split=True, save_path=None)
_, val_loader, _ = load_MNIST(64, trainset, testset, splited, 0)

model_sdr_gradient_update.eval()
_ = validate_stochastic(val_loader, model_sdr_gradient_update, nn.CrossEntropyLoss().cuda())# approx 1% increase

# Run this to generate new samples
method = 'LocalSearch'
adv_samples, adv_tgts = Generate_Adversarial_Samples(model_sdr_gradient_update, val_loader, num_classes, method) #Generate Adv Samples
np.save('adv_samples/val_adv_samples_MINIST_{}.npy'.format(method), adv_samples) #Save the adversarial samples
np.save('adv_samples/val_adv_targets_MINIST_{}.npy'.format(method), adv_tgts) 
print ('Done!')
