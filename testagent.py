import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import ast

from torchvision import datasets
import torchvision.transforms
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
normalize = True # whether or not to normalize the data

def weights_to_vec(cnn):
    '''
    transforms weights in a CNN object into a vector.
    weights in output vector are ordered as follows:
    conv1.0.weight, conv1.0.bias, conv2.0.weight, conv2.0.bias, out.weight, out.bias.
    '''
    d = cnn.state_dict()
    out = torch.cat([d[key].view(-1) for key in d.keys()], 0)
    return out

def vec_to_weights(vec, cnn):
    '''
    Writes a vector of weights to the weights of cnn.
    '''
    counter = 0
    for param in cnn.parameters():
        size = param.size() # size of the current parameter block
        parnumber = param.view(-1).size()[0] # nimber of parameters in the current block
        param.data = vec[counter:counter+parnumber].view(size)
        counter += parnumber


class CNN(nn.Module):
    '''
    A CNN model with approx. 30.000 trainable parameters.
    '''
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
        self.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1) # x.size(0) is batch size, -1 flattens each item
            output = self.out(x)
        return output

class CNN2(nn.Module):
    '''
    Model with reduced complexity (4.380 parameters instead of almost 30.000), but almost
    identical performance for standard backpropagation learning.
    '''
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(160, 10)
        #self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 160)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        x = self.fc1(x)
        return x


class EvoAgent:
    '''
    Class for an evolutionary agent.
    :param lr: learning rate
    :param sigma: deviation used in sampling
    :param n: number of samples generated in each training iteration
    :param modeltype: can be used to use different CNN models, e.g. model=1 or model=2. Custom models can easily be added
    :param model: allows to load a saved model. Model has to be file containing a list with model parameters (same format as produced by `self.save()`). New model is initialized if model == ''.
    '''
    def __init__(self, lr = 0.00002, sigma = 0.01, n = 100, modeltype = 1, model = ''):
        if modeltype == 1:
            self.net = CNN()
        else:
            self.net = CNN2()
        if model != '':
            self.load(model)

        self.lr = lr # learning rate
        self.sigma = sigma # deviation of the Gaussian
        self.n = n # number of samples generated in each training step (to be precise, 2n samples are generated each iteration)
        self.parnumber = sum(p.numel() for p in self.net.parameters()) # total number of model parameters
        self.dist = Normal(torch.zeros(self.parnumber), torch.ones(self.parnumber))
        self.train_data = datasets.MNIST(
            root = 'data',
            train = True,
            transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                        ]) if normalize else torchvision.transforms.ToTensor(),
            download = True,
        )

        self.test_data = datasets.MNIST(
            root = 'data',
            train = False,
            transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                        ]) if normalize else torchvision.transforms.ToTensor(),
            download = True,
        )

        self.loaders = {
            'train' : torch.utils.data.DataLoader(self.train_data,
                                                  batch_size=100,
                                                  shuffle=True,
                                                  num_workers=1),

            'test'  : torch.utils.data.DataLoader(self.test_data,
                                                  batch_size=100,
                                                  shuffle=True,
                                                  num_workers=1),
        }

    def load(self, model):
        try:
            file = open(model, 'r')
            params = file.readline()
            params = ast.literal_eval(params)
            params = torch.tensor(params, dtype=torch.float32)
            vec_to_weights(params, self.net)
            print('Loaded model successfully!')
        except:
            print('Failed to load model! Make sure you pass the correct file name, and the file contains a list with the model parameters in the first line.')

    def save(self, name):
        '''
        Saves current model parameters formatted by weights_to_vec and then transformed into list.
        :param name: name of the file to save model in, e.g. 'model.txt'.
        '''
        file = open(name, 'w')
        file.write(str(weights_to_vec(self.net).tolist()))
        file.close()

    def test(self, rounds = 5):
        '''
        tests the model on `rounds` training sets and returns performance.
        '''
        self.net.eval()
        with torch.no_grad():
            accuracy = []
            i = 0
            for images, labels in self.loaders['train']:
                net_outs = self.net(images)
                pred_y = torch.max(net_outs, 1)[1].data.squeeze()
                accuracy.append((pred_y == labels).sum().item()/float(labels.size(0)))
                i += 1
                if i >= rounds:
                    break
        return sum(accuracy)/len(accuracy)

    def test_on_testset(self, rounds = 20):
        '''
        tests the model on `rounds` test sets and returns performance.
        '''
        self.net.eval()
        with torch.no_grad():
            accuracy = []
            i = 0
            for images, labels in self.loaders['test']:
                net_outs = self.net(images)
                pred_y = torch.max(net_outs, 1)[1].data.squeeze()
                accuracy.append((pred_y == labels).sum().item()/float(labels.size(0)))
                i += 1
                if i >= rounds:
                    break
        return sum(accuracy)/len(accuracy)

    def train(self, rounds = 1000):
        '''
        trains agent by weighting noise proportional to its performance, as in the paper "Evolution Strategies as a
        Scalable Alternative to Reinforcement Learning".
        '''
        for i in tqdm(range(rounds)):
            curr_weights = deepcopy(weights_to_vec(self.net))
            print('current absolute weight average: {}'.format(curr_weights.abs().mean()))
            curr_perf = self.test(50)
            noise = np.array([np.array(self.dist.sample()) for _ in range(self.n)], dtype=float)
            noise = np.append(noise, [-noise[j] for j in range(self.n)])
            weight_update = np.zeros(self.parnumber)
            tests = []
            for j in range(2*self.n):
                weights = torch.tensor(curr_weights + self.sigma*noise[j], dtype=torch.float32) # note: tensor+np.array returns tensor
                vec_to_weights(weights, self.net) # sets agent's weights to current noisy weights
                perf = self.test()
                weight_update += (perf - curr_perf)*noise[j]
                tests.append(perf)
            print('noise performance: {}'.format(tests))
            print('avg noise performance: {}'.format(sum(tests)/len(tests)))
            weight_update = self.lr/(2*self.n*self.sigma)*weight_update
            print('weight update mean: {}'.format(torch.tensor(weight_update).abs().mean()))
            new_weights = torch.tensor(curr_weights + weight_update, dtype=torch.float32)
            vec_to_weights(new_weights, self.net)

    def train_elite(self, rounds = 1000, elite = 0.1):
        '''
        Trains the agent by choosing an elite set in each iteration. the average weight out of this elite set will be used as mean in the following iteration.
        :param elite: the top `elite` agents in each iteration will be used in the weight update.
        '''
        for i in tqdm(range(rounds)):
            curr_weights = deepcopy(weights_to_vec(self.net))
            print('current absolute weight average: {}'.format(curr_weights.abs().mean()))
            curr_perf = self.test(50)
            noise = np.array([np.array(self.dist.sample()) for _ in range(self.n)], dtype=float)
            noise = np.concatenate((noise, -noise))
            tests = []
            for j in range(2*self.n):
                weights = torch.tensor(curr_weights + self.sigma*noise[j], dtype=torch.float32) # note: tensor+np.array returns tensor
                vec_to_weights(weights, self.net)
                tests.append(self.test())
            noise = [[noise[j], tests[j]] for j in range(2*self.n)] # inserts each noise's test result into the respective array element
            #print('noise: {}'.format(noise))
            #print('noise[0]: {}'.format(noise[0]))
            elite_share = int(2*self.n*elite) # number of elite agents
            #print('number of elite agents: {}'.format(elite_share))
            noise = sorted(noise, key=lambda x: x[1]) # sort noise by its performance
            #print('noise performance: {}'.format([n[1] for n in noise]))
            print('performance of elite: {}'.format([n[1] for n in noise[2*self.n-elite_share:]]))
            weight_update = np.zeros(self.parnumber)
            for j in range(elite_share):
                weight_update += self.sigma*noise[-j-1][0]
            weight_update /= elite_share
            print('weight update mean: {}'.format(torch.tensor(weight_update).abs().mean()))
            new_weights = torch.tensor(curr_weights + weight_update, dtype=torch.float32)
            vec_to_weights(new_weights, self.net)
if __name__ == '__main__':
    a = EvoAgent()
