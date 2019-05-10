
import numpy as np
import torch
from torch.utils.data import Dataset
import foolbox

import random
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from train_sdr import SDR_MLP, MLP


'''
Available attacks:
- Attacks implemented from foolbox
- EOT attacks proposed by https://arxiv.org/abs/1802.00420 that 
  strategically target stochastic models

'''

all_attacks = {'GradientAttack': foolbox.attacks.GradientAttack,
               'GradientSignAttack': foolbox.attacks.GradientSignAttack,
               'IterativeGradient': foolbox.attacks.IterativeGradientAttack,
               'IterativeGradientSign': foolbox.attacks.IterativeGradientSignAttack,
               'Linf_BasicIterative': foolbox.attacks.LinfinityBasicIterativeAttack,
               'L1_BasicIterative': foolbox.attacks.L1BasicIterativeAttack,
               'L2_BasicIterative': foolbox.attacks.L2BasicIterativeAttack,
               'ProjectedGradientDescent': foolbox.attacks.ProjectedGradientDescentAttack,
               'RandomStartProjectedGradientDescent': foolbox.attacks.RandomStartProjectedGradientDescentAttack,
               'MomentumIterative': foolbox.attacks.MomentumIterativeAttack,
               'LBFGSAttack': foolbox.attacks.LBFGSAttack,
               'DeepFool': foolbox.attacks.DeepFoolAttack,
               'LocalSearch': foolbox.attacks.LocalSearchAttack}


class AdvDataset(Dataset):
    def __init__(self, adv_samples, target_list, num_classes): 
        self.adv_samples = adv_samples
        self.target_list = target_list
        self.num_class = num_classes

    def __len__(self):
        return self.adv_samples.shape[0]

    def __getitem__(self, idx):
        img = self.adv_samples[idx]
        img = torch.from_numpy(img)
        label = self.target_list[idx]
        return img, label

def Generate_Adversarial_Samples(model, loader, num_classes, attack_type = 'GradientSignAttack'):
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    fmodel = foolbox.models.PyTorchModel(model, bounds=(-1, 1), num_classes=num_classes)
    attack = all_attacks[attack_type](fmodel)
    # attack = foolbox.attacks.FGSM(fmodel)
    
    adv_samples = None
    adv_tgts = None
    
    for batch_num, (data, tgts) in enumerate(loader):
        if batch_num % 15 == 0: print(batch_num)
        if torch.cuda.is_available():
            data = data.cuda()
            tgts = tgts.cuda()
        for i in range(data.shape[0]):
            w = np.asarray(data[i].view(1,28,28).cpu())   
            l = tgts[i].cpu().numpy()
            attack_result = attack(w, l)
            
            if type(attack_result) != type(None):
                if type(adv_samples) == type(None):
                    adv_samples = attack_result.reshape(1,1,28,28)
                    adv_tgts = [int(l)]
                else:
                    adv_samples = np.concatenate([adv_samples, attack_result.reshape(1,1,28,28)])
                    adv_tgts.append(int(l))

    return adv_samples, np.array(adv_tgts)

def EOT_Generate_Adversarial_Samples(model, loader, num_classes, attack_type):
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    fmodel = foolbox.models.PyTorchModel(model, bounds=(-1, 1), num_classes=num_classes)
    attack = all_attacks[attack_type](fmodel)
    # attack = foolbox.attacks.FGSM(fmodel)
    
    adv_samples = None
    adv_tgts = None

    num_att_samples = 5
    
    for batch_num, (data, tgts) in enumerate(loader):
        if batch_num % 15 == 0: print(batch_num)
        if torch.cuda.is_available():
            data = data.cuda()
            tgts = tgts.cuda()
        for i in range(data.shape[0]):
            w = np.asarray(data[i].view(1,28,28).cpu())   
            l = tgts[i].cpu().numpy()
            attack_result = attack(w, l)

            for i in range(num_att_samples - 1):
                attack_result += attack(w, l)

            attack_result /= num_att_samples
            
            if type(attack_result) != type(None):
                if type(adv_samples) == type(None):
                    adv_samples = attack_result.reshape(1,1,28,28)
                    adv_tgts = [int(l)]
                else:
                    adv_samples = np.concatenate([adv_samples, attack_result.reshape(1,1,28,28)])
                    adv_tgts.append(int(l))

    return adv_samples, np.array(adv_tgts)


# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# # train_loader = data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8)

# def eot_gradient_attack(model, criterion, dataset, num_classes = 10):

#     num_samples = len(dataset)
#     random_ids = list(range(num_samples))
#     # random.shuffle(random_ids)

#     for i in random_ids:
#         img, tgt = dataset.__getitem__(i)
#         img.requires_grad_(True)
#         adv_tgt = np.random.randint(0, num_classes)
#         while adv_tgt == tgt:
#             adv_tgt = np.random.randint(0, num_classes)
#         adv_tgt = torch.tensor(adv_tgt).view(-1)
#         tgt = torch.tensor(tgt).view(-1)
#         pred = model(img)
#         print(torch.argmax(pred))

#         # loss = criterion(pred, adv_tgt)
#         loss = criterion(pred, tgt)
#         loss.backward()

#         # img.requires_grad_(False)
#         new_img = img + img.grad

#         print(torch.argmax(model(new_img)))

#         return 
























