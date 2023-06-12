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


# Input Layer 

The input layer in an artificial neural network (ANN) is the initial layer that receives the input data and passes it to the subsequent layers for processing. It serves as the interface between the external world and the network.

The configuration and size of the input layer are determined by the dimensionality and nature of the input data. Here are a few key points about the input layer in an ANN:

Neurons as input nodes: Each feature or attribute of the input data is typically represented by a separate input neuron in the input layer. For example, if the input data is an image with dimensions 32x32 pixels and each pixel is considered a feature, the input layer would consist of 32x32 = 1024 input neurons.

Encoding and scaling: Input data is usually preprocessed and scaled before being fed into the neural network. Common preprocessing steps include normalization to a common scale (e.g., between 0 and 1) or standardization (e.g., mean of 0 and standard deviation of 1). This ensures that all features contribute proportionately to the learning process and prevents certain features from dominating the network due to their larger magnitude.

Activation functions: Unlike the hidden layers, which employ various activation functions to introduce nonlinearity, the input layer typically does not use any activation function. Its purpose is mainly to transmit the input data to the subsequent layers without modifying or transforming it.

Input layer size: The number of neurons in the input layer is equal to the dimensionality of the input data. For example, if the input data is a vector of length n, the input layer will have n neurons. The input layer size is fixed and determined by the problem domain and the type of data being processed.

Compatibility with the data: The input layer's architecture must be compatible with the input data format. For instance, if the input is an image, the input layer should be configured as a two-dimensional grid of neurons, matching the image's dimensions. Similarly, for sequential data like text or time series, the input layer is often designed to handle sequences, such as a one-dimensional array of neurons.

The input layer acts as the starting point of information flow in the neural network. It receives the raw data, processes it, and passes it through the subsequent layers, allowing the network to learn and extract meaningful patterns and representations from the input.

# Hidden layers 

Hidden layers in an artificial neural network (ANN) refer to the layers between the input layer and the output layer. They are called "hidden" because they are not directly observable from the outside, unlike the input and output layers.

In an ANN, information flows from the input layer, through the hidden layers, and finally to the output layer. Each hidden layer consists of multiple artificial neurons, also known as nodes or units. The number of hidden layers and the number of neurons in each layer are hyperparameters that need to be determined during the design and training of the network.

The purpose of having hidden layers is to allow the network to learn complex patterns and relationships in the data. Each neuron in a hidden layer receives inputs from the previous layer, applies a nonlinear activation function to the weighted sum of those inputs, and produces an output. The outputs from the neurons in one hidden layer serve as inputs to the neurons in the next hidden layer until the final output layer is reached.

The presence of multiple hidden layers in an ANN allows for the formation of hierarchical representations of the data. Each hidden layer can learn increasingly abstract features or representations of the input data. This hierarchical processing enables ANNs to model and understand complex relationships in the data, making them capable of solving a wide range of tasks, including image and speech recognition, natural language processing, and predictive modeling.

The choice of the number of hidden layers and the number of neurons in each layer depends on the specific problem being solved, the complexity of the data, and the available computational resources. Deep neural networks,which have multiple hidden layers, have shown impressive performance in various domains but may require more data and computational power for training.


# Output Layers 

The output layer in an artificial neural network (ANN) is the final layer of neurons that produces the network's output or prediction. It is the last layer through which information flows, following the propagation of data through the hidden layers and potentially other intermediate layers.

The structure and configuration of the output layer depend on the nature of the task the network is designed to solve. Here are a few common scenarios:

Classification: In classification tasks, where the goal is to assign input data to specific classes or categories, the output layer typically consists of one neuron per class. The output neurons often employ activation functions like softmax, which generates a probability distribution over the classes, indicating the likelihood of the input belonging to each class. The class with the highest probability is usually considered the predicted class.

Regression: In regression tasks, where the goal is to predict a continuous numerical value, the output layer generally consists of a single neuron. The output neuron may have a linear activation function or no activation function at all, directly outputting the predicted numerical value.

Multi-label classification: In some cases, the task may involve assigning multiple labels to an input. The output layer can be configured to have multiple neurons, where each neuron corresponds to one label. The activation function used for each output neuron can be sigmoid, which produces a value between 0 and 1, representing the probability or confidence of the input being associated with that label.

Other tasks: Depending on the problem at hand, the output layer can have different configurations. For example, in sequence-to-sequence tasks such as machine translation, the output layer may have a recurrent structure to generate output sequences. Similarly, in generative models like generative adversarial networks (GANs), the output layer may represent the generated data.

It's important to note that the number of neurons in the output layer is determined by the dimensionality of the problem. Each neuron typically represents a distinct aspect or prediction associated with the task. The choice of activation function in the output layer depends on the specific problem requirements and the desired range or interpretation of the output values.

