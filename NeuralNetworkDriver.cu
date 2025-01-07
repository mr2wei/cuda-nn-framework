#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "NNLayer.hpp"  
#include "NeuralNetwork.hpp"
#include "LinearLayer.hpp"
#include "ActivationLayer.hpp"

#include <stdio.h>
#include <vector>
#include <iostream>

int main()
{
    // // Define the shape of the network
    // const int shape_length = 4;
    // int shape[shape_length] = { 8, 6, 4, 1 };

    // Initialize weights and biases as in MultiLayer/kernel.cu
    float host_weights[] = {
        1.62f, -0.61f, -0.53f, -1.07f, 0.87f, -2.30f, 1.74f, -0.76f, 0.32f, -0.25f, 1.46f, -2.06f, -0.32f, -0.38f, 1.13f, 
        -1.10f, -0.17f, -0.88f, 0.04f, 0.58f, -1.10f, 1.14f, 0.90f, 0.50f, 0.90f, -0.68f, -0.12f, -0.94f, -0.27f, 0.53f, 
        -0.69f, -0.40f, -0.69f, -0.85f, -0.67f, -0.01f, -1.12f, 0.23f, 1.66f, 0.74f, -0.19f, -0.89f, -0.75f, 1.69f, 0.05f, 
        -0.64f, 0.19f, 2.10f, 0.12f, 0.62f, 0.30f, -0.35f, -1.14f, -0.35f, -0.21f, 0.59f, 0.84f, 0.93f, 0.29f, 0.89f, -0.75f, 
        1.25f, 0.51f, -0.30f, 0.49f, -0.08f, 1.13f, 1.52f, 2.19f, -1.40f, -1.44f, -0.50f, 0.16f, 0.88f, 0.32f, -2.02f};
    float host_biases[] = {-0.31f, 0.83f, 0.23f, 0.76f, -0.22f, -0.20f, 0.19f, 0.41f, 0.20f, 0.12f, -0.67f};

    // Initialize input activations
    float host_activations[] = {0.38f, 0.12f, 1.13f, 1.20f, 0.19f, -0.38f, -0.64f, 0.42f};

    // float* device_activations;
    // cudaMalloc(&device_activations, sizeof(host_activations));
    // cudaMemcpy(device_activations, host_activations, sizeof(host_activations), cudaMemcpyHostToDevice);

    // Create layers
    std::vector<NNLayer*> layers = {
        new LinearLayer(8, 4),
        new ActivationLayer(4, ActivationLayer::ActivationType::SIGMOID),
        new LinearLayer(4, 1),
        new ActivationLayer(1, ActivationLayer::ActivationType::SIGMOID)
    };

    // Create neural network with specific weights and biases
    NeuralNetwork nn(layers, host_weights, host_biases);

    float target = 0.5f;
    float learning_rate = 0.01f;
    for (int i = 0; i < 3; i++) {
        // Perform forward pass
        nn.forward(host_activations);

        // Retrieve and print activations
        // std::vector<float> activations = nn.get_activations();
        // std::cout << "Activations length: " << activations.size() << std::endl;
        // for (float activation : activations) {
        //     std::cout << activation << std::endl;
        // }

        // print z-values
        // std::vector<float> z_values = nn.get_z_values();
        // std::cout << "Z-values length: " << z_values.size() << std::endl;
        // for (float z_value : z_values) {
        //     std::cout << z_value << std::endl;
        // }

        // print weights gradient
        std::vector<float> weights_gradient = nn.get_weights_gradient();
        std::cout << "Weights gradient: " << std::endl;
        for (float weight_gradient : weights_gradient) {
            std::cout << weight_gradient << std::endl;
        }

        // print biases gradient
        std::vector<float> biases_gradient = nn.get_biases_gradient();
        std::cout << "Biases gradient: " << std::endl;
        for (float bias_gradient : biases_gradient) {
            std::cout << bias_gradient << std::endl;
        }

        // print input gradient
        std::vector<float> input_gradient = nn.get_input_gradient();
        std::cout << "Input gradient: " << std::endl;
        for (float input_gradient : input_gradient) {
            std::cout << input_gradient << std::endl;
        }

        // print results
        std::vector<float> results = nn.get_results();
        for (float result : results) {
            std::cout << result << std::endl;
        }

        // backward pass and step
        nn.zero_gradients();
        nn.backward(target);
        nn.step(learning_rate);
        std::cout << "Step " << i << " complete" << std::endl;
    }

    // print the weights and biases
    // std::vector<float> weights = nn.get_weights();
    // std::vector<float> biases = nn.get_biases();
    // std::cout << "Weights length: " << weights.size() << std::endl;
    // std::cout << "Biases length: " << biases.size() << std::endl;
    // for (float weight : weights) {
    //     std::cout << weight << std::endl;
    // }
    // for (float bias : biases) {
    //     std::cout << bias << std::endl;
    // }


    return 0;
}
