import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
import torch.nn.functional as F

from copy import deepcopy
import numpy as np
from tqdm import tqdm
import ast
from scipy.special import softmax
import math
import os
import random
import matplotlib.pyplot as plt

from utils import *
from networks import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
			self.optimizer.zero_grad()
			loss = F.binary_cross_entropy(out, traintargets[i:i+5])
			loss.backward()
			self.optimizer.step()
		return self.test(self.testdata_cat, self.testdata_other, self.testtargets_cat, self.testtargets_other)

	def test(self, testdata_cat, testdata_other, testtargets_cat, testtargets_other):
		self.net.eval()
		testdata = torch.cat((testdata_cat, testdata_other))
		testtargets = torch.cat((testtargets_cat, testtargets_other))
		targets_guess = self.net(testdata)
		m = len(testtargets)
		compare = [torch.argmax(testtargets[i])==torch.argmax(targets_guess[i]) for i in range(m)]
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
	def __init__(self, lr = 0.075, sigma = 0.01, n = 300, train_nb = 40, test_nb = 100, model = '', categories_ratio = 0.7, dir = '/home/jonathan/Documents/Studium/VaiosProject/FewShotModel/data/'):
		categories = os.listdir(dir)
		nb_train = int(categories_ratio*len(categories))
		self.train_cats = random.sample(categories, nb_train) # names of categories used for training
		self.test_cats = [cat for cat in categories if cat not in self.train_cats] # names of categories used for testing
		self.sigma = sigma # deviation of the Gaussian
		self.lr = lr
		self.n = n # number of samples generated in each training step (to be precise, 2n samples are generated each iteration)
		self.train_nb = train_nb
		self.test_nb = test_nb
		if (train_nb+test_nb)%20 != 0 or train_nb%10 != 0 or test_nb%2 != 0:
			print(f"WARNING: {train_nb=} + {test_nb=} should be a multiple of 20, {train_nb=} a multiple of 10 and {test_nb=} a multiple of 2.")
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
			train_nb = self.train_nb//2
			test_nb = self.test_nb//2
			classifier_cats = random.sample(self.train_cats, nb_classifiers)
			classifier_data_cat = [load_files(categories=1, per_category=train_nb+test_nb, rand=False, names=[cat])[cat[:-4]] for cat in classifier_cats] # contains 50 samples from each classifier's category
			classifier_train_cat = [data[:train_nb] for data in classifier_data_cat]

			train_cat_targets = torch.tensor([[1, 0] for i in range(train_nb)], dtype=torch.float32)
			classifier_test_cat = [data[-test_nb:] for data in classifier_data_cat]

			test_cat_targets = torch.tensor([[1, 0] for i in range(test_nb)], dtype=torch.float32)
			other_nb = (train_nb+test_nb)//10
			other_data = load_files(categories=10, per_category=other_nb, rand=True, not_names=classifier_cats)
			other_data_train = np.concatenate([item[1][:train_nb//10] for item in other_data.items()]) # np.array with data from other categories used in training

			other_train_targets = torch.tensor([[0, 1] for i in range(train_nb)], dtype=torch.float32) # targets for training data from other categories
			other_data_test = np.concatenate([item[1][-test_nb//10:] for item in other_data.items()]) # same as above for testing

			other_test_targets = torch.tensor([[0, 1] for i in range(test_nb)], dtype=torch.float32)
			### end of bunch ###
			for j in range(2*self.n):
				weights = (curr_weights + self.sigma*noise[j]).to(torch.float32)
				vec_to_weights(weights, self.net)
				tmp_classifier_train_cat = [self.net(torch.tensor(data.reshape(train_nb, 1, 28, 28), dtype=torch.float32)) for data in classifier_train_cat] # preprocess data by first layer network
				tmp_classifier_test_cat = [self.net(torch.tensor(data.reshape(test_nb, 1, 28, 28), dtype=torch.float32)) for data in classifier_test_cat]
				tmp_other_data_train = self.net(torch.tensor(other_data_train.reshape(train_nb, 1, 28, 28), dtype=torch.float32))
				tmp_other_data_test = self.net(torch.tensor(other_data_test.reshape(test_nb, 1, 28, 28), dtype=torch.float32))
				classifiers = [Classifier(self, tmp_classifier_train_cat[i], train_cat_targets, tmp_other_data_train, other_train_targets, tmp_classifier_test_cat[i], test_cat_targets, tmp_other_data_test, other_test_targets) for i in range(nb_classifiers)]
				scores = [classifier.train() for classifier in classifiers]
				#print(f'classifier scores: {scores}')
				noise_weights.append(max(0, float(sum(scores))/nb_classifiers-avg_perf)) # performance of 'good' noise. only remember noise that performed better than the previous average
				#TODO: maybe only update the avg_perf in each iteration if it surpasses the previous avg_perf, s.t. the threshold for 'good' noise cannot be reduced?
				#TODO: we could save the weights of the classifiers s.t. each offspring gets the exact same classifiers upon initialization
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

	def train_elite(self, rounds = 100, nb_classifiers = 8, elite = 0.1):
		'''
		Trains agent by creating noise samples in each iteration, and then updating the network's weights using the average of the elite set, which is the best
		`elite` of its offspring.
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
			train_nb = self.train_nb//2
			test_nb = self.test_nb//2
			classifier_cats = random.sample(self.train_cats, nb_classifiers)
			classifier_data_cat = [load_files(categories=1, per_category=train_nb+test_nb, rand=False, names=[cat])[cat[:-4]] for cat in classifier_cats] # contains samples from each classifier's category
			classifier_train_cat = [data[:train_nb] for data in classifier_data_cat]
			train_cat_targets = torch.tensor([[1, 0] for i in range(train_nb)], dtype=torch.float32)
			classifier_test_cat = [data[-test_nb:] for data in classifier_data_cat]
			test_cat_targets = torch.tensor([[1, 0] for i in range(test_nb)], dtype=torch.float32)
			other_nb = (train_nb+test_nb)//10
			other_data = load_files(categories=10, per_category=other_nb, rand=True, not_names=classifier_cats)
			other_data_train = np.concatenate([item[1][:train_nb//10] for item in other_data.items()]) # np.array with data from other categories used in training
			other_train_targets = torch.tensor([[0, 1] for i in range(train_nb)], dtype=torch.float32) # targets for training data from other categories
			other_data_test = np.concatenate([item[1][-test_nb//10:] for item in other_data.items()]) # same as above for testing
			other_test_targets = torch.tensor([[0, 1] for i in range(test_nb)], dtype=torch.float32)
			### end of bunch ###
			for j in range(2*self.n):
				weights = (curr_weights + self.sigma*noise[j]).to(torch.float32)
				vec_to_weights(weights, self.net)
				tmp_classifier_train_cat = [self.net(torch.tensor(data.reshape(train_nb, 1, 28, 28), dtype=torch.float32)) for data in classifier_train_cat] # preprocess data by first layer network
				tmp_classifier_test_cat = [self.net(torch.tensor(data.reshape(test_nb, 1, 28, 28), dtype=torch.float32)) for data in classifier_test_cat]
				tmp_other_data_train = self.net(torch.tensor(other_data_train.reshape(train_nb, 1, 28, 28), dtype=torch.float32))
				tmp_other_data_test = self.net(torch.tensor(other_data_test.reshape(test_nb, 1, 28, 28), dtype=torch.float32))
				classifiers = [Classifier(self, tmp_classifier_train_cat[i], train_cat_targets, tmp_other_data_train, other_train_targets, tmp_classifier_test_cat[i], test_cat_targets, tmp_other_data_test, other_test_targets) for i in range(nb_classifiers)]
				scores = [classifier.train() for classifier in classifiers]
				#print(f'Avg classifier score: {sum(scores)/nb_classifiers}')
				performances.append(float(sum(scores))/nb_classifiers) # contains performances of each noise sample as a value between 0 and 1
			avg_perf = sum(performances)/len(performances)
			plot_data.append(avg_perf)
			print('avg noise performance: {}'.format(avg_perf))
			print(f'min and max noise performance: {min(performances)}, {max(performances)}')
			noise = [[noise[j], performances[j]] for j in range(2*self.n)]
			elite_share = int(2*self.n*elite)
			noise = sorted(noise, key=lambda x: x[1])
			elite_performance = [n[1] for n in noise[-elite_share:]]
			print(f'Elite performance: {elite_performance[0]} to {elite_performance[-1]}')
			elite_set = np.array([noise[j][0] for j in range(2*self.n-elite_share, 2*self.n)])
			weight_update = self.sigma*sum(elite_set)/elite_share
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
