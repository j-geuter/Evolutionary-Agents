import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import ast
from scipy.special import softmax

from torchvision import datasets
import torchvision.transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from matplotlib import pyplot as plt
import math
import os
import random
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_files(categories = 8, per_category = 1000, rand = True, names = None, not_names = None, dir = '/home/jonathan/Documents/Studium/VaiosProject/FewShotModel/data/'):
	'''
	Loads data. NOTE: Change directory accordingly. CAREFUL: attempting to load multiple categories at once (50+) can kill the call.
	:param categories: number of categories to load. If set to 'ALL', loads all categories. WARNING: this will most likely kill the call.
	:param per_category: number of samples to load per category. Set to `None` to load complete category.
	:param rand: if True, chooses random categories to load. If False, either loads categories from `names` if passed or the first `categories` from alphabetical order.
	:param names: optional argument to pass specific names of categories to load. Iterable that contains strings. Needs `rand` to be set to False. Ignores the `categories` argument if passed.
	:param not_names: Iterable that contains strings. If passed, the selection of names makes sure not to take any names in `not_names`.
	:param dir: directory where data is stored.
	:return: dictionary with category names as keys, and np.arrays as data.
	'''
	filenames = sorted(os.listdir(dir))
	if categories == 'ALL':
		categories = len(filenames)
	if not rand:
		if names:
			for i in range(len(names)):
				if names[i].endswith('.npy'):
					names[i] = names[i][:-4]
			data = {name: None for name in names}
		else:
			filenames = [filename for filename in filenames if not filename in not_names]
			data = {name[:-4]: None for name in filenames[:categories]}
	else:
		if not_names == None:
			not_names = []
		for i in range(len(not_names)):
			if not not_names[i].endswith('.npy'):
				not_names[i] += '.npy'
		cats = random.sample([filename for filename in filenames if not filename in not_names], categories)
		data = {name[:-4]: None for name in cats}
	for name in data:
		data[name] = np.load(dir+name+'.npy')[:per_category]
	return data

def show_images(images, titles = None):
	'''
	Shows images from the dataset.
	:param images: np.array either of shape n*784 or n*28*28.
	'''
	n = len(images)
	if not images.shape == (n, 28, 28):
		images = np.reshape(images, (n, 28, 28))
	if titles == None:
		titles = [f'Image {i}' for i in range(len(images))]
	root = int(math.sqrt(n)) + 1
	fig = plt.figure()
	position = 1
	for image in images:
		fig.add_subplot(root, root, position)
		plt.imshow(image)
		plt.axis('off')
		plt.title(titles[position-1])
		position += 1
	plt.show()

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
		parnumber = param.view(-1).size()[0] # number of parameters in the current block
		param.data = vec[counter:counter+parnumber].view(size)
		counter += parnumber

class CNN(nn.Module):
	'''
	First layer network. The CNN that will learn abstract features of the data. ~50,000 trainable parameters.
	'''
	def __init__(self):
		super(CNN, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 10, 5, 1, 2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(10, 20, 5, 1, 2),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(20, 40, 5, 1, 2),
			nn.ReLU(),
			nn.MaxPool2d(2, padding=1),
		)
		self.out = nn.Linear(40*4*4, 40)
		self.requires_grad_(False)

	def forward(self, x):
		with torch.no_grad():
			x = self.conv1(x)
			x = self.conv2(x)
			x = self.conv3(x)
			x = x.view(x.size(0), -1) # x.size(0) is batch size, -1 flattens each item
			output = self.out(x)
		return output

class FCNN(nn.Module):
	'''
	Small classifier network used in training the first layer. 82 trainable parameters.
	'''
	def __init__(self):
		super(FCNN, self).__init__()
		self.fc1 = nn.Linear(40, 2)


	def forward(self, x):
		x = self.fc1(x)
		x = F.softmax(x, 0)
		return x


class Classifier:
	'''
	Class for classifiers that correspond to an EvoAgent object and feature a CNN2 network. Data should be preprocessed by layer 1 network.
	:param traindata_cat: training data from the classifier's category.
	:param traintargets_cat: targets for `traindata_cat`.
	:param traindata_other: training data from other random categories.
	:param traintargets_other: targets for `traindata_other`.
	:param testdata_cat: test data from the classifier's category.
	:param testtargets_cat: targets for `testdata_cat`.
	:param testdata_other: test data from other random categories.
	:param testtargets_other: targets for `testdata_other`.
	:param lr: learning rate.
	'''
	def __init__(self, evoagent, traindata_cat, traintargets_cat, traindata_other, traintargets_other, testdata_cat, testtargets_cat, testdata_other, testtargets_other, lr = 0.1):
		self.evoagent = evoagent
		self.net = FCNN()
		self.traindata_cat = traindata_cat
		self.traintargets_cat = traintargets_cat
		self.traindata_other = traindata_other
		self.traintargets_other = traintargets_other
		self.testdata_cat = testdata_cat
		self.testtargets_cat = testtargets_cat
		self.testdata_other = testdata_other
		self.testtargets_other = testtargets_other
		self.lr = lr
		self.optimizer = optim.SGD(self.net.parameters(), lr=lr)

	def train(self):
		'''
		Trains classifier on training data and returns score on test data (between 0 and 1).
		'''
		traindata = torch.cat((self.traindata_cat, self.traindata_other))
		traintargets = torch.cat((self.traintargets_cat, self.traintargets_other))
		n = len(traindata)
		perm = torch.randperm(n)
		traindata = traindata[perm] # shuffle data
		traintargets = traintargets[perm]
		self.net.train()
		for i in range(n//5): # training loop
			out = self.net(traindata[i:i+5])
			#print(f"Classifier training prediction: {out}")
			#print(f"Classifier training data batch: {traindata[i:i+5]}")
			self.optimizer.zero_grad()
			loss = F.binary_cross_entropy(out, traintargets[i:i+5])
			#print(f"Batch loss: {loss}")
			#print(f"Classifier training targets batch: {traintargets[i:i+5]}")
			loss.backward()
			self.optimizer.step()
		self.net.eval()
		testdata = torch.cat((self.testdata_cat, self.testdata_other))
		testtargets = torch.cat((self.testtargets_cat, self.testtargets_other))
		#print(f"Classifier test data targets: {testtargets}")
		targets_guess = self.net(testdata)
		#print(f"Classifier test data target guess: {targets_guess}")
		m = len(testtargets)
		compare = [torch.argmax(testtargets[i])==torch.argmax(targets_guess[i]) for i in range(m)]
		#print(f"Classifier performance: {sum(compare)/m}")
		return sum(compare)/m



class EvoAgent:
	'''
	Class for an evolutionary agent.
	:param lr: learning rate
	:param sigma: deviation used in sampling
	:param n: number of samples generated in each training iteration
	:param train_nb: number of samples used in every classifier for training. Half of these will be from its category, the other half from 10 random, other categories.
	:param test_nb: number of samples used in every classifier for testing.
	:param model: allows to load a saved model. Model has to be file containing a list with model parameters (same format as produced by `self.save()`). New model is initialized if model == ''.
	:param categories_ratio: ratio of how many categories are for training.
	:param dir: directory of data. NOTE: Change accordingly.
	'''
	def __init__(self, lr = 0.075, sigma = 0.01, n = 500, train_nb = 100, test_nb = 100, model = '', categories_ratio = 0.7, dir = '/home/jonathan/Documents/Studium/VaiosProject/FewShotModel/data/'):
		categories = os.listdir(dir)
		nb_train = int(categories_ratio*len(categories))
		self.train_cats = random.sample(categories, nb_train) # names of categories used for training
		self.test_cats = [cat for cat in categories if cat not in self.train_cats] # names of categories used for testing
		self.sigma = sigma # deviation of the Gaussian
		self.lr = lr
		self.n = n # number of samples generated in each training step (to be precise, 2n samples are generated each iteration)
		self.train_nb = train_nb
		self.test_nb = test_nb
		self.net = CNN()
		self.parnumber = sum(p.numel() for p in self.net.parameters()) # total number of model parameters
		self.dist = Normal(torch.zeros(self.parnumber), torch.ones(self.parnumber))

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

	def train(self, rounds = 100, nb_classifiers = 8):
		'''
		Trains agent by creating noise samples in each iteration, and then updating the network's weights using a weighted average of the noise.
		:param rounds: number of training rounds.
		:param nb_classifiers: number of classifiers generated for each noise sample.
		:return: `plot_data` which tracks the average noise performance for each training iteration.
		'''
		avg_perf = 0
		plot_data = [] # collects the average noise performances for each iteration. can be plotted with self.plot
		for i in tqdm(range(rounds)):
			print('\n--------------------------------------------------------------------------------')
			curr_weights = deepcopy(weights_to_vec(self.net))
			print('current absolute weight average: {}'.format(curr_weights.abs().mean()))
			noise = np.array([np.array(self.dist.sample()) for _ in range(self.n)], dtype=float)
			noise = np.concatenate((noise, -noise))
			noise_weights = []
			performances = []
			### a bunch of data preprocessing incoming ###
			classifier_cats = random.sample(self.train_cats, nb_classifiers)
			classifier_data_cat = [load_files(categories=1, per_category=100, rand=False, names=[cat])[cat[:-4]] for cat in classifier_cats] # contains 50 samples from each classifier's category
			classifier_train_cat = [data[:-50] for data in classifier_data_cat]
			classifier_train_cat = [self.net(torch.tensor(data.reshape(50, 1, 28, 28), dtype=torch.float32)) for data in classifier_train_cat] # preprocess data by first layer network
			train_cat_targets = torch.tensor([[1, 0] for i in range(50)], dtype=torch.float32)
			classifier_test_cat = [data[50:] for data in classifier_data_cat]
			classifier_test_cat = [self.net(torch.tensor(data.reshape(50, 1, 28, 28), dtype=torch.float32)) for data in classifier_test_cat]
			test_cat_targets = torch.tensor([[1, 0] for i in range(50)], dtype=torch.float32)
			other_data = load_files(categories=10, per_category=10, rand=True, not_names=classifier_cats)
			other_data_train = np.concatenate([item[1][:-5] for item in other_data.items()]) # np.array with data from other categories used in training
			other_data_train = self.net(torch.tensor(other_data_train.reshape(50, 1, 28, 28), dtype=torch.float32))
			other_train_targets = torch.tensor([[0, 1] for i in range(len(other_data_train))], dtype=torch.float32) # targets for training data from other categories
			other_data_test = np.concatenate([item[1][5:] for item in other_data.items()]) # same as above for testing
			other_data_test = self.net(torch.tensor(other_data_test.reshape(50, 1, 28, 28), dtype=torch.float32))
			other_test_targets = torch.tensor([[0, 1] for i in range(len(other_data_test))], dtype=torch.float32)
			### end of bunch ###
			for j in range(2*self.n):
				classifiers = [Classifier(self, classifier_train_cat[i], train_cat_targets, other_data_train, other_train_targets, classifier_test_cat[i], test_cat_targets, other_data_test, other_test_targets) for i in range(nb_classifiers)]
				scores = [classifier.train() for classifier in classifiers]
				#print(f'classifier scores: {scores}')
				noise_weights.append(max(0, float(sum(scores))/nb_classifiers-avg_perf)) # performance of 'good' noise. only remember noise that performed better than the previous average
				performances.append(float(sum(scores))/nb_classifiers) # contains actual performances
			avg_perf = sum(performances)/len(performances)
			plot_data.append(avg_perf)
			print('avg noise performance: {}'.format(avg_perf))
			print(f'min and max noise performance: {min(performances)}, {max(performances)}')
			noise_weights = np.array(noise_weights)
			noise_weights = softmax(100*noise_weights) # enforces better weighting in the update while bounding the update step size
			print(f'top ten noise weights: {sorted(noise_weights)[-10:]}')
			print(f'max noise weight: {max(noise_weights)}')
			noise_weights = noise_weights.reshape(len(noise_weights), 1)
			weight_update = self.lr/(2*self.n*self.sigma)*sum(noise_weights*noise)
			print('weight update mean: {}'.format(torch.tensor(weight_update).abs().mean()))
			new_weights = (curr_weights + weight_update).to(torch.float32)
			vec_to_weights(new_weights, self.net)
		return plot_data

	def plot(self, data, x = None, x_l = 'Iteration', y_l = 'Performance'):
		'''
		Plots `data`.
		:param data: list or similar containing data to be plotted.
		:param x: x-axis data. If None, set to [i for i in range(len(data))].
		:param x_l: x-axis label
		:param y_l: y-axis label
		'''
		if x == None:
			x = [i for i in range(len(data))]
		plt.plot(x, data)
		plt.xlabel(x_l)
		plt.ylabel(y_l)
		plt.show()


if __name__ == '__main__':
	a = EvoAgent()
