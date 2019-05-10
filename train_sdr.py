import argparse, os, time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from general_helpers import *
from data_helpers import *

from sdr_modules import *
from resnet_sdr import *

parser = argparse.ArgumentParser(description = "PyTorch model training")
parser.add_argument("--epochs", default = 100, type = int, help = "Number of training epochs")
parser.add_argument("--batch_size", default = 64, type = int, help = "Size of mini-batch")
parser.add_argument("--learning_rate", default = 1e-3, type = float, help = "Initial learning rate")
#parser.add_argument("--momentum", default = 0.9, type = float, help = "Momentum)
parser.add_argument("--weight_decay", default = 1e-4, type = float, help = "Weight decay")
parser.add_argument("--dropout_prob", default = 0.0, type = float, help = "Droptout probability")
parser.add_argument("--experiment_id", default = "ResNet-18", type = str, help = "Name of experiment")
parser.add_argument("--sdr", default = False, help = "Whether to use the stochastic delta rule")
parser.add_argument("--beta", default = 5.0, type = float, help = "SDR beta value")
parser.add_argument("--zeta", default = 0.99, type = float, help = "SDR zeta value")
parser.add_argument("--zeta_drop", default = 1, type = int, help = "Control rate of zeta drop") # what is this? --> parabolic annealing
parser.add_argument("--file_path", default = False, help = "Path to save parameter weights and standard deviations")
parser.add_argument("--std_init", default = "xavier", type = str, help = "Initialization method to standard deviations")

parser.add_argument("--print_freq", default = 10, type = int, help = "Frequency to print training statistics")
parser.add_argument("--ckpt_freq", default= 5, type = int, help = "Frequency to save model specifics")
parser.add_argument('--load_prev', default = False)

parser.add_argument('--dataset', default = 'MINIST', type = str)
parser.add_argument('--std_update_rule', default = 'none', type = str)

#########################################################################################################

class MLP(nn.Module):
    def __init__(self, hiddens, num_classes):
        super().__init__()
        self.sdr = False
        layers = []
        for i in range(len(hiddens) - 1):
            l = nn.Linear(hiddens[i], hiddens[i+1])
            nn.init.xavier_normal_(l.weight.data)
            layers.append(l)
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hiddens[-1], num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        return self.layers(x)

class SDR_MLP(nn.Module):
    def __init__(self, hiddens, num_classes):
        super().__init__()
        self.sdr = True
        layers = []
        for i in range(len(hiddens) - 1):
            layers.append(SDR_Linear(hiddens[i], hiddens[i+1]))
            layers.append(nn.ReLU())
        layers.append(SDR_Linear(hiddens[-1], num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        return self.layers(x)

def build_model(dataset, std_update_rule):
    if dataset == 'MINIST':
        hiddens = [28*28, 100, 100, 100]
        num_classes = 10

        if std_update_rule == 'gradient':
            model = SDR_MLP(hiddens, num_classes)

        elif std_update_rule == 'decay':
            model = MLP(hiddens, num_classes)
            model.sdr = True

        elif std_update_rule == 'none':
            model = MLP(hiddens, num_classes)

    # std_update_rule = 'decay' for CNN
    elif dataset == 'CIFAR10':
        num_classes = 10
        if std_update_rule == "gradient":
            model = resnet18()
        else:
            raise NotImplementedError

    elif dataset == 'FACE':
        hiddens = [64, 64, 64, 128, 128, 256, 256, 512, 512]

        if std_update_rule == 'gradient':
            raise NotImplementedError

        elif std_update_rule == 'decay':
            model = face.Face_CNN(3, hiddens, 2300)
            model.sdr = True

        elif std_update_rule == 'none':
            model = face.Face_CNN(3, hiddens, 2300)

    return model

def init_std_halved_xavier(parameter):
    lower = 0.0
    upper = np.sqrt(2 / np.product(parameter.shape)) * 0.5
    std_init = torch.randn(parameter.shape)
    max_val = torch.max(std_init)
    min_val = torch.min(std_init)
    std_init = ((upper - lower) / (max_val - min_val)).float() * (std_init - min_val)
    return std_init

def adjust_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 after half
    of training and 75% of training, according to:
    https://github.com/noahfl/sdr-densenet-pytorch/blob/master/train.py
    """
    if epoch + 1 == args.epochs//2 or epoch + 1 == int(args.epochs // (4/3)):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        print('Decayed learning rate by 10')

def load_prev_model(args, model):
    # load std
    #std_np = np.load(os.path.join('model', 'std', 'last_std_epoch_0_ckpt.npy'))
    std_np = np.load(os.path.join(args.file_path, 'std', 'std_epoch_0_ckpt.npy'))
    model.std_lst = [torch.from_numpy(std) for std in std_np]
    # load params
    #model.load_state_dict(torch.load(os.path.join('model', 'weight','last_model_epoch_0_ckpt.pt')))
    model.load_state_dict(torch.load(os.path.join(args.file_path, 'weight','model_epoch_0_ckpt.pt')))

def main(args):
    if torch.cuda.is_available():
        print('Using GPU for this task...')
        device = torch.device('cuda')
    else:
        print('Using CPU for this task...')
        device = torch.device('cpu')

    print("Constructing train and validation loader...")
    train_loader, val_loader = load_DATA(DATASET, BATCH_SIZE)
    print("Constructing model...")
    model = build_model(DATASET, STD_UPDATE_RULE)
    model = model.to(device)

    # dropout_prob = 0.0 if args.sdr else args.dropout_prob

    print("Configuring model...")
    if STD_UPDATE_RULE == 'decay':
        model.beta = args.beta
        model.zeta = args.zeta
        model.zeta_init = args.zeta
        model.zeta_drop = args.zeta_drop
        model.std_lst = []
        model.zeta_orig = args.zeta

    if args.file_path:
        if STD_UPDATE_RULE == 'decay':
            # init_weights = [np.asarray(p.data.cpu()) for p in model.parameters()]
            fname_w = os.path.join(args.file_path, "weight", "init_weights_{}.pt".format(args.experiment_id))
            torch.save(model.state_dict(), fname_w)
            # del init_weights
        else:
            fname_w = os.path.join(args.file_path, "together", "init_%s.pt" % args.experiment_id)
            torch.save(model.state_dict(), fname_w)

    print("Training model...")
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

    if args.std_init == "xavier":
        std_init_fn = init_std_halved_xavier

    train_losses, train_accuracies, val_accuracies = [], [], []
    for e in range(args.epochs):

        print("Training on epoch {}...".format(e+1))
        train_loss, train_acc = train(args, train_loader, model, criterion, optimizer, e, device, std_init_fn)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        print("Evaluating on validation set...")
        val_acc = validate(val_loader, model, criterion, e, device, DATASET)
        print("Evaluation completed!")
        val_accuracies.append(val_acc)

        if args.std_update_rule == 'decay' and (e + 1) % args.zeta_drop == 0: # parabolic annealing
            print('Decreased zeta according to zeta_drop')
            # uncomment if more than 200 layers
            # "larger networks benefit from longer exposure to noise"
            # model.zeta = model.zeta_orig ** ((e + 1) // model.zeta_drop)

            lambda_ = 0.1
            model.zeta = model.zeta_orig * np.power(np.e, -(lambda_ * e))

        adjust_learning_rate(args, optimizer, e)

    print('Finished!')

def train(args, train_loader, model, criterion, optimizer, curr_epoch, device, std_init_fn = init_std_halved_xavier):
    losses = AverageMeter(None)
    accuracies = AverageMeter(None)
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # if args.dataset == 'MINIST': # the default dims are for CNN
        #     inputs = inputs.reshape(-1, 28*28)

        if args.std_update_rule == 'decay':
            if curr_epoch == 0 and i == 0:
                if args.load_prev:
                    print('Load last trained model')
                    load_prev_model(args, model)
                else:
                    for param in model.parameters(): # initialize std for all params
                        std = std_init_fn(param) # type torch.tensor
                        model.std_lst.append(std)
                    if args.file_path:
                        fname_std = os.path.join(args.file_path, "std", "init_std.npy")
                        std_lst_ = [np.asarray(std) for std in model.std_lst]
                        np.save(fname_std, std_lst_)
                        del std_lst_
            elif i in [args.batch_size // 2 - 1, args.batch_size - 1]:
                for j, param in enumerate(model.parameters()):
                    model.std_lst[j] = model.zeta * (torch.abs(model.beta * param.grad) + model.std_lst[j].cuda())

            store = []
            for j, param in enumerate(model.parameters()):
                store.append(param.data)
                param.data = torch.distributions.Normal(param.data, model.std_lst[j].cuda()).sample()
                #print(torch.max(param.data - store[j]))
        elif args.load_prev and args.std_update_rule == "gradient":
            # load trained sdr-gradient module
            model.load_state_dict(torch.load("sdr_grad_cifar_epoch_29_ckpt.pt"))
        # outputs = model(inputs)[1]
        outputs = model(inputs)

        if args.std_update_rule == 'decay':
            for sampled_param, stored_param in zip(model.parameters(), store):
                sampled_param.data = stored_param

        loss = criterion(outputs, labels)
        correct, batch_size  = accuracy(outputs, labels)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(100.0 * correct / batch_size, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del inputs
        del labels
        #del store
        torch.cuda.empty_cache()

        if i % args.print_freq == 0:
            print('[%d, %5d] loss: %.3f, accuracy: %.3f' % (curr_epoch+1, i+1, losses.avg, accuracies.avg))

    if args.file_path and curr_epoch % args.ckpt_freq == 0:
        if args.std_update_rule == 'decay':
            std_lst_ = [np.asarray(std.cpu()) for std in model.std_lst]
            fname_std = os.path.join(args.file_path, "std", "std_epoch_{}_ckpt.npy".format(curr_epoch))
            np.save(fname_std, std_lst_)
            fname_w = os.path.join(args.file_path, "weight", "model_epoch_{}_ckpt.pt".format(curr_epoch))
            torch.save(model.state_dict(), fname_w)
            del std_lst_
        else:
            fname = os.path.join(args.file_path, 'together', "model_epoch_{}_ckpt_{}.pt".format(curr_epoch, args.experiment_id))
            torch.save(model.state_dict(), fname)

    return losses.avg, accuracies.avg

def validate(val_loader, model, criterion, curr_epoch, device, dataset):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # outputs = model(inputs)[1]

            # if dataset == 'MINIST': # the default dims are for CNN
            #     inputs = inputs.reshape(-1, 28*28)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            batch_correct, batch_size = accuracy(outputs, labels)
            correct += batch_correct
            total += batch_size
            del inputs
            del labels
            torch.cuda.empty_cache()
    res = 100.0 * correct / total
    print("Classification accuracy on the validation set is {:.3f}".format(res))
    model.train()
    return res


if __name__ == "__main__":
    args = parser.parse_args()
    DATASET = args.dataset
    BATCH_SIZE = args.batch_size
    STD_UPDATE_RULE = args.std_update_rule

    main(args)
