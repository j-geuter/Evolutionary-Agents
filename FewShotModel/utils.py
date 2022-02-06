import os
import random
import matplotlib.pyplot as plt
import torch
import numpy as np


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
