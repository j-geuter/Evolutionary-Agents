# Evolutionary Models
Repo for models that learn using evolutionary strategies, i.e. that learn gradient-free.

## MNIST Classification
MNISTClassification features a model that can learn to classify the MNIST dataset using gradient-free methods. I have been able to train models to achieve ~70% accuracy after a few hours of training. `model.py` features two different training methods, one creating a weighted average of all offspring weights created in each iteration (`train`), the other creating an elite set of offspring and using their average for the next iteration (`train_elite`). I've had more success with the elite set training algorithm.

## 2LayerModel
