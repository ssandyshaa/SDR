
import torch

class AverageMeter(object):
    def __init__(self, alpha = 0.9):
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        if self.alpha is None:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
        else:
            self.avg = self.alpha * val + (1 - self.alpha) * val

def accuracy(outputs, labels):
    correct, total = 0, 0
    with torch.no_grad():
        batch_size = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted)
        correct = (predicted == labels).sum().item()
    return correct, batch_size


def validate_regular(dataloader, model, criterion):
    '''
    Regular inference.

    '''
    correct, total = 0, 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            batch_correct, batch_size = accuracy(outputs, labels)
            correct += batch_correct
            total += batch_size
            del inputs
            del labels
            torch.cuda.empty_cache()
        
    res = 100.0 * correct / total
    print("Classification accuracy {:.3f}".format(res))
    model.train()
    return res  

def validate_stochastic(dataloader, model, criterion, num_inferences = 10):
    '''
    Inference function for models with more randomness.
    Do inference multiple times to 'stablize' results.

    '''
    print('Running each sample on the stochastic model for {} times.'.format(num_inferences))
    correct, total = 0, 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            for j in range(num_inferences - 1):        
                outputs += model(inputs)
            outputs = outputs/num_inferences
            loss = criterion(outputs, labels)
            batch_correct, batch_size = accuracy(outputs, labels)
            correct += batch_correct
            total += batch_size
            del inputs
            del labels
            torch.cuda.empty_cache()
        
    res = 100.0 * correct / total
    print("Classification accuracy {:.3f}".format(res))
    model.train()
    return res   

def validate_decay(dataloader, model, criterion, stds, num_inferences = 1):
    '''
    Inference function for models with pretrained stds under the 'decay' setting.

    '''
    print('Running each sample on the stochastic model for {} times.'.format(num_inferences))
    correct, total = 0, 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            outputs = torch.zeros(model(inputs).shape).cuda()
            
            for ii in range(num_inferences):
                store = []
                for j, param in enumerate(model.parameters()):  
                    store.append(param.data)
                    param.data = torch.distributions.Normal(param.data, torch.tensor(stds[j]).cuda()).sample()

                outputs += model(inputs)

                for sampled_param, stored_param in zip(model.parameters(), store):
                    sampled_param.data = stored_param
            
            outputs = outputs/num_inferences
            loss = criterion(outputs, labels)
            batch_correct, batch_size = accuracy(outputs, labels)
            correct += batch_correct
            total += batch_size
            del inputs
            del labels
            torch.cuda.empty_cache()
        
    res = 100.0 * correct / total
    print("Classification accuracy {:.3f}".format(res))
    model.train()
    return res 














# def findNremove(path,pattern,maxdepth=1):
#     cpath=path.count(os.sep)
#     for r,d,f in os.walk(path):
#         if r.count(os.sep) - cpath <maxdepth:
#             for files in f:
#                 if files.startswith(pattern):
#                     try:
#                         #print "Removing %s" % (os.path.join(r,files))
#                         os.remove(os.path.join(r,files))
#                     except Exception as e:
#                         print(e)
#                     else:
#                         print("%s removed" % (os.path.join(r,files)))

                        