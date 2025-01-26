# CUDA Neural Network Project

## Overview
This project was built from scratch using CUDA to gain practical experience in designing and implementing neural network components and to explore CUDA programming for developing GPU accelerating machine learning models. Inspired by PyTorch, this project allows creating modular neural networks allowing different combination of layers. The project includes key components such as linear layers, activation layers, and the overall neural network structure. 

## Capabilities

### 1. **Custom Neural Network Architecture**
- Implemented modular neural network components, including:
  - **Linear Layers**: Handles matrix multiplications and bias addition for each layer.
  - **Activation Layers**: Supports common activation functions like ReLU, Sigmoid, and Tanh.

### 2. **GPU Acceleration with CUDA**
- Leveraged CUDA programming to:
  - Write custom kernels for matrix operations.
  - Optimize activation function computations using parallel processing.
  - Manage memory allocation and data transfers between host and device.

### 3. **Scalability**
- Designed the framework to support larger neural network architectures by:
  - Abstracting common operations into reusable modules.
  - Providing flexibility to integrate additional layers and configurations.

## Features
- **Linear Layer**
  - Implements forward propagation with matrix multiplication and bias addition.
  - Backpropagation support for calculating gradients.

- **Activation Layer**
  - Includes common activation functions:
    - ReLU
    - Sigmoid

- **Modular Layer Arrangement**
  - Allows diverse combinations of layer types and number of parameters.

## Key Learning Outcomes
1. **CUDA Programming**
   - Developed proficiency in writing custom CUDA kernels for deep learning operations.
   - Gained insights into thread-level parallelism and GPU memory management.

2. **Low-Level Neural Network Implementation**
   - Built a solid understanding of neural network operations, including matrix multiplications, activation functions, and gradient calculations.

3. **Performance Optimization**
   - Explored techniques for optimizing GPU operations, reducing execution time and memory overhead.

## Future Improvements
- **Add Support for Additional Layers**
  - Convolutional layers for image-based tasks.
  - Dropout layers for regularization.

- **Implement Advanced Optimizers**
  - Include optimizers like Adam to improve training convergence.

- **Allow Batch Processing**
  - Allow back propagation and forward pass on mini batches to improve training.
  
## File Structure
- **`ActivationLayer.cu`**: Implements activation functions and their forward/backward propagation.
- **`LinearLayer.cu` & `LinearLayer.hpp`**: Implements linear layer operations for neural networks.
- **`NNLayer.hpp`**: Base class for neural network layers.
- **`NeuralNetwork.cu` & `NeuralNetwork.hpp`**: Defines the overall neural network structure and workflows.
- **`NeuralNetworkDriver.cu`**: Contains the driver program for testing and demonstrating the neural network.

## How to Use
This project is designed as a learning and experimentation framework for implementing neural networks with CUDA. To use this project:

1. **Include the Necessary Files**
   - Incorporate `NeuralNetwork.hpp`, `LinearLayer.hpp`, `ActivationLayer.hpp`, and other relevant files into your project.

2. **Create a Neural Network**
   - Use the provided classes to define your neural network architecture. For example:
     ```cpp
     NeuralNetwork nn;
     nn.addLayer(new LinearLayer(input_size, hidden_size));
     nn.addLayer(new ActivationLayer("ReLU"));
     nn.addLayer(new LinearLayer(hidden_size, output_size));
     ```

3. **Train the Network**
   - Provide training data and implement the training loop to optimize weights and biases using backpropagation and gradient descent.

4. **Test the Network**
   - Pass input data through the network to perform inference and evaluate performance.

5. **Modify and Extend**
   - Experiment with different layer configurations, activation functions, and optimization techniques to enhance the framework.
