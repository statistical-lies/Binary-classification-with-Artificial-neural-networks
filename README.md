# Binary-classification-with-Artificial-neural-networks
Introduction 
In the field of machine learning, there is an area called Artificial Neural Network that consists of layers of interconnected nodes, often known as artificial neurons or units, these interconnected nodes form the general architecture of the network. There is an input layer, one or more hidden layers, and an output layer are the traditional divisions of the layers. Each neuron takes in inputs, processes them, and then sends an output to the layer above it. Weights are used to represent the connections between neurons in an ANN and control their strength and importance. In order to reduce the error between the projected output and the actual output, the network modifies these weights during training based on the input data and intended output. Artificial neural networks come in a variety of shapes and sizes, including feedforward neural networks, recurrent neural networks (RNNs), convolutional neural networks (CNNs), and more sophisticated architectures like generative adversarial networks (GANs) and transformer networks. For the purpose of this project, we will focus on feedforward neural networks.
![image](https://github.com/statistical-lies/Binary-classification-with-Artificial-neural-networks/assets/59277986/d26a7768-8802-4775-8827-b506925df813)

# Feedforward

A feedforward neural network, sometimes referred to as a multilayer perceptron (MLP), is a type of artificial neural network in which data only travels in one way, from the input layer via one or more hidden layers to the output layer. It is among the most prevalent and basic architectural designs used in neural network models. The diagram below shows a feedforward neural network




![image](https://github.com/statistical-lies/Binary-classification-with-Artificial-neural-networks/assets/59277986/89a7ac9b-212b-4de3-89c3-845aae30a107)

# Compile Keras Model

The model compilation process comes after the model definition. The model compilation is carried out using tensor flow. Parameters are set for model training and predictions during the compilation process. Distributed memory or CPU/GPU can be used in the background.
The loss function that will be used to calculate the weights for the various layers must be specified. The optimizer switches between different weight sets while adjusting the learning rate. In this instance, the loss function will be the Binary Cross Entropy. We will employ ADAM, a powerful stochastic gradient descent (SGD) technique, in the optimizer's scenario.
It is widely employed for tuning. We will gather and report the classification accuracy, as stated by the metrics argument because it is a classification problem. In this instance, accuracy will be used.
