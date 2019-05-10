import os, time, random, copy, csv
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

################################################
# Dataset classes & functions
################################################

def parse_data(path):
    count = 0
    img_list = []
    ID_list = []
    for root, dirs, files in os.walk(path): 
        for file in files: # all files under the same class folder
            if file.endswith('.jpg') and not file.startswith('._'):
                fpath = os.path.join(root, file)
                img_list.append(fpath)
                ID_list.append(int(root.split('/')[-1]))
        #         count += 1
        #         if count >= 6000:
        #             break
        # if count >= 6000:
        #     break
    num_classes = len(set(ID_list))
    print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), num_classes))
    return img_list, ID_list, num_classes

def parse_data_test(path):
    count = 0
    img_dict = dict()
    for root, dirs, files in os.walk(path): 
        for file in files: # all files under the same class folder
            if file.endswith('.jpg') and not file.startswith('._'):
                fpath = os.path.join(root, file)
                fidx = int(file.replace('.jpg', ''))
                img_dict[fidx] = fpath
    return img_dict

class FaceDataset(Dataset):
    def __init__(self, file_list, target_list, num_classes):
        self.file_list = file_list
        self.target_list = target_list
        self.num_class = num_classes

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx])
        img = torchvision.transforms.ToTensor()(img)
        label = self.target_list[idx]
        return img, label

class FaceDataset_test(Dataset):
    def __init__(self, img_dict):
        self.imgs = img_dict

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = torchvision.transforms.ToTensor()(img)
        return img

################################################
# Network n loss classes & functions
################################################

def init_weights(module):
    if type(module) == nn.Conv2d or type(module) == nn.Linear:
        torch.nn.init.xavier_normal_(module.weight.data)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        stride = 1
        self.downsample = None
        if in_channels != out_channels: # downsampling with stride = 2 required
            stride = 2
            down_layer = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                                   kernel_size = 2, stride = stride, bias = False)
            down_bn = nn.BatchNorm2d(num_features = out_channels)
            self.downsample = nn.Sequential(down_layer, down_bn)


        self.block = nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                                             kernel_size = 3, stride = stride, padding = 1, bias = False), # half the map size if need be
                                   nn.BatchNorm2d(num_features = out_channels),
                                   nn.ReLU(inplace = True),
                                   nn.Conv2d(in_channels = out_channels, out_channels = out_channels,
                                             kernel_size = 3, stride = 1, padding = 1, bias = False), # preserve map size
                                   nn.BatchNorm2d(num_features = out_channels))

        self.final_nonlinear = nn.ReLU(inplace = True)

    def forward(self, x):
        # I referenced the original pytorch resnet impelmentation here: https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet18
        residual = x
        out = self.block(x)
        if self.downsample != None:
            residual = self.downsample(residual)
        out += residual
        out = self.final_nonlinear(out)
        return out

class Face_CNN(nn.Module):
    def __init__(self, num_in_channels, nums_hidden_channels, num_classes):
        super().__init__()
        # Before residual components
        in_layer = nn.Conv2d(in_channels = num_in_channels, 
                             out_channels = nums_hidden_channels[0],
                             kernel_size = 3, stride = 1, padding = 1, bias = False)
        in_bn = nn.BatchNorm2d(num_features = nums_hidden_channels[0])
        in_nonlinear = nn.ReLU(inplace = True)
        self.layers = [in_layer, in_bn, in_nonlinear]

        # Residual components
        for i in range(len(nums_hidden_channels) - 1):
            in_channels = nums_hidden_channels[i]
            out_channels = nums_hidden_channels[i+1]
            self.layers.append(ResBlock(in_channels, out_channels))

        # After residual components: can either go to an MLP clf or flatten as an embedding
        self.layers = nn.Sequential(*self.layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear_clf = nn.Linear(nums_hidden_channels[-1], num_classes, bias = False)

    def forward(self, x, evalMode = False):
        output = x
        output = self.layers(output) # conv layers
        output = self.avgpool(output).view(output.size(0), -1) # embedding
        clf_out = self.linear_clf(output)
        return output, clf_out

class Face_CNN_plus(nn.Module):
    def __init__(self, face_CNN, in_channels, out_channels, num_classes):
        super().__init__()

        self.prev = face_CNN.layers
        # for param in self.prev.parameters():
        #     param.requires_grad = False

        self.one_more_conv = nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                                                     kernel_size = 3, stride = 2, padding = 1, bias = False),
                                           nn.BatchNorm2d(num_features = out_channels),
                                           nn.ReLU(inplace = True))
        self.one_more_linear = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                             nn.Linear(out_channels, num_classes))

    def forward(self, x):
        x = self.prev(x)
        emb_x = self.one_more_conv(x) 
        clf_x = self.one_more_linear(emb_x)
        return emb_x, clf_x

class Face_CNN_XL(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.feature_extractor = feature_extractor.layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.new_mlp = nn.Sequential(nn.Linear(512, num_classes))

    def forward(self, x):
        embd = self.feature_extractor(x)
        embd = self.avgpool(embd).view(embd.size(0), -1)
        clf = self.new_mlp(embd)
        return embd, clf

################################################
# Training n testing classes & functions
################################################

def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, task = 'Classification'):
    print('[LOG] Start Training...')
    
    model.train()

    for epoch in range(num_epochs):
        curr_avg_loss = 0.0
        avg_loss = 0.0

        curr_t = time.time()

        for batch_num, (data, tgts) in enumerate(train_loader):
            if CUDA:
                data = data.cuda()
                tgts = tgts.cuda()

            optimizer.zero_grad()
            outputs = model(data)[1]

            loss = criterion(outputs, tgts.long())
            loss.backward()
            optimizer.step()

            curr_avg_loss += loss.item()

            btemp = 50

            if batch_num % btemp == btemp - 1:
                t_elapsed = (time.time() - curr_t)/batch_num
                est_t_rm = t_elapsed * (6425 - batch_num) / 60
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}\tTime-Remaining: {:.3f}mins'.format(epoch+1, batch_num+1, curr_avg_loss/btemp, est_t_rm))
                curr_avg_loss = 0.0

            torch.cuda.empty_cache()
            del data
            del tgts
            del loss

        epoch_i = epoch + 4

        model_path = 'Res18_0308_epoch_%i.pt' % epoch_i
        save_model(model, model_path)
        print('[LOG] Finish training the current epoch, model saved to %s' % model_path)

        if task == 'Classification':
            val_loss, val_acc = test_classify(model, val_loader, criterion)
            train_loss, train_acc = test_classify(model, train_loader, criterion)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.format(train_loss, train_acc, val_loss, val_acc))

        elif task == 'Verification':
            val_loss, val_acc = test_verify(model, val_loader, criterion)
            train_loss, train_acc = test_verify(model, train_loader, criterion)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.format(train_loss, train_acc, val_loss, val_acc))

def save_model(model, path):
    torch.save(model.state_dict(), path)

def test_classify(model, loader, criterion):
    model.eval()
    test_loss = 0.0
    acc = 0
    total = 0

    for batch_num, (data, tgts) in enumerate(loader):
        if CUDA:
            data = data.cuda()
            tgts = tgts.cuda()
        outputs = model(data)[1]

        _, pred_labels = torch.max(F.softmax(outputs, dim = 1), 1)
        pred_labels = pred_labels.view(-1)

        loss = criterion(outputs, tgts.long())

        acc += torch.sum(torch.eq(pred_labels, tgts)).item()
        batch_size = len(tgts)
        total += batch_size
        test_loss += loss.item() * batch_size

        del data
        del tgts

    model.train()
    return test_loss/total, acc/total

def main():
    batch_size = 128

    print('[LOG] Loading training and validation data...')
    img_list, label_list, num_classes = parse_data('hw2p2_check/train_data/medium')
    train_dataset = FaceDataset(img_list, label_list, num_classes)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4, drop_last = False)
    
    img_list, label_list, num_classes = parse_data('hw2p2_check/validation_classification/medium')
    val_dataset = FaceDataset(img_list, label_list, num_classes)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, num_workers = 4, drop_last = False)

    print('[LOG] Setting up model...')
    hiddens = [64, 64, 64, 128, 128, 256, 256, 512, 512]
    model = Face_CNN(3, hiddens, num_classes)
    if CUDA: model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 0.0001)
    criterion = nn.CrossEntropyLoss()

    #augmented_model = Face_CNN_plus(model, 512, 512 * 2, num_classes)

    model.load_state_dict(torch.load('Res18_0308_epoch_3.pt'))
    num_epochs = 1
    train(model, optimizer, criterion, train_dataloader, val_dataloader, num_epochs)

def test_main(model_path = 'Res18_0308_epoch_4.pt'):
    batch_size = 128
    num_classes = 2300

    print('[LOG] Loading testing data...')
    test_imgs = parse_data_test('hw2p2_check/test_classification/medium')
    test_dataset = FaceDataset_test(test_imgs)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, num_workers = 4, drop_last = False)

    print('[LOG] Setting up model...')
    hiddens = [64, 64, 64, 128, 128, 256, 256, 512, 512]
    model = Face_CNN(3, hiddens, num_classes)
    if CUDA: model = model.cuda()
    #augmented_model = Face_CNN_plus(model, 512, 512 * 2, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print('[LOG] Start testing...')
    out = torch.zeros(len(test_imgs))
    batch_num = 0
    for data in test_dataloader:
        if CUDA: data = data.cuda()

        outputs = model(data)[1]
        _, pred_labels = torch.max(F.softmax(outputs, dim = 1), 1)
        pred_labels = pred_labels.view(-1)

        l = batch_num * batch_size
        out[l:l + len(pred_labels)] = pred_labels.view(-1)

        del data
        batch_num += 1

    idxs = np.arange(len(test_imgs))
    out = out.numpy()
    out = np.stack((idxs, out)).T
    np.savetxt('linhongl_0309_c_1.csv', out, fmt = '%d', header = 'id,label', comments = '', delimiter = ',')
    print("[Testing] Finished testing and writing csv file!... ")

################################################
# Main
################################################

if __name__ == '__main__':
    CUDA = torch.cuda.is_available()
    if CUDA: print("[GPU STATUS] Using GPU for this task...")
    else: print("[GPU STATUS] GPU currently unavailable... Using CPU for this task.")
    #main() # training main() function 
    test_main()

