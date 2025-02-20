#include <stdio.h>

#include <iostream>
#include <vector>

#include "ActivationLayer.hpp"
#include "LinearLayer.hpp"
#include "NNLayer.hpp"
#include "NeuralNetwork.hpp"
#include "Optimizer.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main()
{
    // Initialize weights and biases (assumes your NeuralNetwork constructor
    // will use the first part of these arrays for the layers that actually have weights)
    // float host_weights[] = {
        // 1.62f, -0.61f, -0.53f, -1.07f, 0.87f, -2.30f, 1.74f, -0.76f,
        // 0.32f, -0.25f, 1.46f, -2.06f, -0.32f, -0.38f, 1.13f, -1.10f,
        // -0.17f, -0.88f, 0.04f, 0.58f, -1.10f, 1.14f, 0.90f, 0.50f,
        // 0.90f, -0.68f, -0.12f, -0.94f, -0.27f, 0.53f, -0.69f, -0.40f,
        // Second layer weights (using next 4 values)
        // -0.69f, -0.85f, -0.67f, -0.01f
        // (Extra values if any will be ignored by your NeuralNetwork initialization)
        // -1.12f, 0.23f, 1.66f, 0.74f, -0.19f, -0.89f, -0.75f, 1.69f, 0.05f, 
        // -0.64f, 0.19f, 2.10f, 0.12f, 0.62f, 0.30f, -0.35f, -1.14f, -0.35f, 
        // -0.21f, 0.59f, 0.84f, 0.93f, 0.29f, 0.89f, -0.75f, 1.25f, 0.51f, 
        // -0.30f, 0.49f, -0.08f, 1.13f, 1.52f, 2.19f, -1.40f, -1.44f, -0.50f, 
        // 0.16f, 0.88f, 0.32f, -2.02f
    // };
    // float host_biases[] = {
        // -0.31f, 0.83f, 0.23f, 0.76f,  // first layer biases (4 values)
        // -0.22f                        // second layer bias (1 value)
        // (Extra values ignored)
        // -0.20f, 0.19f, 0.41f, 0.20f, 0.12f, -0.67f
    // };

    float host_weights[] = { 1.62f, -0.61f, -0.53f, -1.07f, 0.87f, -2.30f, 1.74f, -0.76f};
    float host_biases[] = {-0.31f, 0.83f};

    // Initialize input activations (single input)
    // float host_activations[] = {0.38f, 0.12f, 1.13f, 1.20f, 0.19f, -0.38f, -0.64f, 0.42f};
    float host_activations[] = {0.38f, 0.12f, 1.13f, 1.20f};

    // Create layers: Two Linear layers with interleaved Activation layers.
    // The first linear layer: 8 inputs -> 4 outputs.
    // The second linear layer: 4 inputs -> 1 output.
    std::vector<NNLayer*> layers = {
        // new LinearLayer(8, 4),
        // new ActivationLayer(4, ActivationLayer::ActivationType::SIGMOID),
        new LinearLayer(4, 2),
        new ActivationLayer(2, ActivationLayer::ActivationType::SIGMOID)
    };

    // Create neural network with specific weights and biases.
    NeuralNetwork nn(layers, host_weights, host_biases);

    // Use ADAM optimizer: note that we now pass beta1, beta2, and epsilon.
    Optimizer optimizer(&nn, 0.01f, Optimizer::OptimizerType::SGD, Optimizer::LossType::MAE, 0.9f, 0.999f, 1e-8f);
    
    std::vector<float> weight_grads = optimizer.get_weights_gradient();
    std::vector<float> bias_grads = optimizer.get_biases_gradient();
    std::vector<float> weights_first_moment = optimizer.get_weights_first_moment();
    std::vector<float> biases_first_moment = optimizer.get_biases_first_moment();
    std::vector<float> weights_second_moment = optimizer.get_weights_second_moment();
    std::vector<float> biases_second_moment = optimizer.get_biases_second_moment();

    std::cout << "initial weight_grads: " << std::endl;
    for (float weight : weight_grads) {
        std::cout << weight << ", ";
    }
    std::cout << std::endl;

    std::cout << "initial bias_grads: " << std::endl;
    for (float bias : bias_grads) {
        std::cout << bias << ", ";
    }
    std::cout << std::endl;

    std::vector<float> target = {0.5f, 0.5f};
    for (int i = 0; i < 1; i++) {
        optimizer.zero_grad();  
        // Forward pass
        nn.forward(host_activations);
        // Backward pass and update using ADAM
        optimizer.backward(target);

        // Print network output
        std::vector<float> results = nn.get_results();
        std::cout << "Iteration " << i << " output = ";
        for (float result : results) {
            std::cout << result << ", ";
        }
        std::cout << "loss = " << optimizer.get_loss(target) << std::endl;

        weight_grads = optimizer.get_weights_gradient();
        bias_grads = optimizer.get_biases_gradient();
        weights_first_moment = optimizer.get_weights_first_moment();
        biases_first_moment = optimizer.get_biases_first_moment();
        weights_second_moment = optimizer.get_weights_second_moment();
        biases_second_moment = optimizer.get_biases_second_moment();

        // print the gradients
        std::cout << "Gradients: " << std::endl;
        std::cout << "Weights: " << std::endl;
        // print weights (2x4)
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {   
                std::cout << weight_grads[i * 4 + j] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Biases: " << std::endl;
        for (float bias : bias_grads) {
            std::cout << bias << ", ";
        }
        std::cout << std::endl;

        // print the first moments
        std::cout << "First moments: " << std::endl;
        std::cout << "Weights: " << std::endl;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                std::cout << weights_first_moment[i * 4 + j] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Biases: " << std::endl;
        for (float bias : biases_first_moment) {
            std::cout << bias << ", ";
        }
        std::cout << std::endl;

        // print the second moments
        std::cout << "Second moments: " << std::endl;
        std::cout << "Weights: " << std::endl;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                std::cout << weights_second_moment[i * 4 + j] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Biases: " << std::endl;
        for (float bias : biases_second_moment) {
            std::cout << bias << ", ";
        }
        std::cout << std::endl;

        optimizer.step();
        // std::cout << "Step " << i << " complete" << std::endl;
    }

    return 0;
}