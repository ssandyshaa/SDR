
import random, itertools
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import face_project as face

def load_DATA(dataset, batch_size):
    print('[load_DATA]Loading the {} dataset'.format(dataset))
    if dataset == 'MINIST':
        return load_MINIST(batch_size)
    elif dataset == 'CIFAR10':
        return load_CIFAR(batch_size)

'''
example call:

MNIST_data = split_MNIST(batch_size = 64, train_val_split = True, save_to = 'saved_split')
train_loader, val_loader, test_loader = load_MNIST(MNIST_data)
'''

class CV_wrapper(Dataset):
    def __init__(self, all_train_dataset, splited_idxs):
        self.all_train_data = all_train_dataset
        self.splited_idxs = splited_idxs

    def __len__(self):
        return len(self.splited_idxs)

    def __getitem__(self, idx):
        return self.all_train_data.__getitem__(self.splited_idxs[idx])

def CV_split(train_size, num_fold):
    all_idxs = list(range(train_size))
    random.shuffle(all_idxs)
    size_per_fold = train_size//num_fold
    result = []
    for i in range(num_fold):
        result.append(all_idxs[i*size_per_fold: (i+1)*size_per_fold])
    return result

def split_MNIST(train_val_split, save_path, num_fold = 5): # call once per experiment
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # temporary, will come back to modify later
    splited = CV_split(len(testset), num_fold)
    return trainset, testset, splited

def load_MNIST(batch_size, trainset, testset, splited, curr_fold): # call @ each fold
    curr_val = splited[curr_fold]
    curr_train = splited[:curr_fold] + splited[curr_fold+1:]
    curr_train = list(itertools.chain(*curr_train))

    # real_trainset = CV_wrapper(trainset, curr_train)
    valset = CV_wrapper(testset, curr_val)
    real_trainset = trainset

    train_loader = data.DataLoader(real_trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader  = data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader

# def load_MINIST(batch_size):
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#     valset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#     train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
#     val_loader  = data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8)

#     return train_loader, val_loader

def load_CIFAR(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    valset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader  = data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, val_loader

def load_FACE(batch_size):
    img_list, label_list, num_classes = face.parse_data(os.path.join("hw2p2_check", "train_data", "medium"))
    trainset = face.FaceDataset(img_list, label_list, num_classes)
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)

    img_list, label_list, num_classes = face.parse_data(os.path.join("hw2p2_check", "validation_classification", "medium"))
    valset = face.FaceDataset(img_list, label_list, num_classes)
    val_loader = data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)
    return train_loader, val_loader


