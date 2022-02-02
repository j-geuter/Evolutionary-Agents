# Evolutionary Models
Repo for models that learn using evolutionary strategies, i.e. that learn gradient-free.

## MNIST Classification
MNISTClassification features a model that can learn to classify the MNIST dataset using gradient-free methods. I have been able to train models to achieve ~70% accuracy after a few hours of training. `model.py` features two different training methods, one creating a weighted average of all offspring weights created in each iteration (`train`), the other creating an elite set of offspring and using their average for the next iteration (`train_elite`). I've had more success with the elite set training algorithm.

## FewShotModel
This model (`EvoAgent`) features two layers. The first layer is a larger, universal CNN (class `CNN`) that learns abstract features of input data. The second layer is a small classifier (class `Classifier`) with a fully connected network (class `FCNN`) learning with the data provided by the first layer. The idea is to make the first layer so good at extracting features from its input that the secondary networks can learn to classify using only very small amounts of data. This idea is also known as few shot learning.
