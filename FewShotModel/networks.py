import torch
import torch.nn as nn
import torch.nn.functional as F

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
