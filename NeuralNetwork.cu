#include "NeuralNetwork.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>

// Helper function to initialize common parts of the constructors
void NeuralNetwork::initialize(std::vector<NNLayer*> layers, float* host_weights, float* host_biases) {
    total_weights = 0;
    total_b_z_a = 0;
    total_input_gradient = 0;
    
    for (NNLayer* layer : layers) {
        if (!layer->is_activation_layer) {
            shape.push_back(layer->num_inputs);
        }
        total_input_gradient += layer->num_inputs;
    }
    shape.push_back(layers.back()->num_outputs);

    
    for (int i = 0; i < shape.size() - 1; i++) {
        total_weights += shape[i] * shape[i + 1];
        total_b_z_a += shape[i + 1];
    }


    bool rand_weights_biases = false;

    if (!host_weights && !host_biases) {
        // Create random weights and biases
        host_weights = new float[total_weights];
        host_biases = new float[total_b_z_a];

        int offset = 0;
        for (int i = 0; i < shape.size() - 1; i++) {
            glorot_uniform_weights(shape[i], shape[i + 1], host_weights, offset);
            offset += shape[i] * shape[i + 1];
        }
        
        for (int i = 0; i < total_b_z_a; i++) {
            host_biases[i] = 0;
        }
        rand_weights_biases = true;
    }

    // Allocate memory for weights, biases, z_values, and activations
    cudaMalloc(&device_weights, total_weights * sizeof(float));
    cudaMalloc(&device_biases, total_b_z_a * sizeof(float));
    cudaMalloc(&device_z_values, total_b_z_a * sizeof(float));
    cudaMalloc(&device_activations, total_b_z_a * sizeof(float));

    // Copy weights, biases, z_values, and activations to device if provided
    if (host_weights && host_biases) {
        cudaMemcpy(device_weights, host_weights, total_weights * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_biases, host_biases, total_b_z_a * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Initialize layers
    size_t weights_offset = 0, b_z_a_offset = 0, input_gradient_offset = 0;
    for (int i = 0; i < layers.size(); i++) {
        // if the first layer is an activation layer, raise an error
        if (i == 0 && layers[i]->is_activation_layer) {
            throw std::runtime_error("First layer cannot be an activation layer");
        }

        // Check for consecutive activation layers
        if (i > 0 && layers[i]->is_activation_layer && layers[i-1]->is_activation_layer) {
            throw std::runtime_error("Cannot have consecutive activation layers");
        }

        layers[i]->weights = device_weights + weights_offset;
        layers[i]->biases = device_biases + b_z_a_offset;
        layers[i]->z_values = device_z_values + b_z_a_offset;
        layers[i]->activations = device_activations + b_z_a_offset;

        cudaMalloc(&layers[i]->prev_input, layers[i]->num_inputs * sizeof(float));

        if (!layers[i]->is_activation_layer) {
            weights_offset += layers[i]->num_inputs * layers[i]->num_outputs;
        }

        // Only increment offsets if the next layer is not an activation layer
        bool next_layer_is_activation = (i < layers.size() - 1) && layers[i + 1]->is_activation_layer;
        if (!next_layer_is_activation) {
            b_z_a_offset += layers[i]->num_outputs;
        }
    }

    if (rand_weights_biases) {
        delete[] host_weights;
        delete[] host_biases;
    }
}

void NeuralNetwork::glorot_uniform_weights(int input_size, int output_size, float* host_weights, int offset) {
    float limit = std::sqrt(6.0f / (input_size + output_size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-limit, limit);
    for (int i = 0; i < input_size * output_size; i++) {
        host_weights[i + offset] = dis(gen);
    }
}

NeuralNetwork::NeuralNetwork(std::vector<NNLayer*> layers)
    : layers(layers) {
    initialize(layers, nullptr, nullptr);
}

NeuralNetwork::NeuralNetwork(std::vector<NNLayer*> layers, float* host_weights, float* host_biases)
    : layers(layers) {
    initialize(layers, host_weights, host_biases);
}

NeuralNetwork::~NeuralNetwork() {
    cudaFree(device_weights);
    cudaFree(device_biases);
    cudaFree(device_z_values);
    cudaFree(device_activations);
}

/**
 * @brief Forward pass through the neural network
 * 
 * @param input: input to the neural network
 */
void NeuralNetwork::forward(float* input) {
    // reset activations and z_values
    cudaMemset(device_activations, 0, total_b_z_a * sizeof(float));
    cudaMemset(device_z_values, 0, total_b_z_a * sizeof(float));

    // Allocate device memory for input
    float* device_input;
    cudaMalloc(&device_input, layers[0]->num_inputs * sizeof(float));
    cudaMemcpy(device_input, input, layers[0]->num_inputs * sizeof(float), cudaMemcpyHostToDevice);
    
    float* current_input = device_input;

    for (int i = 0; i < layers.size(); i++) {
        NNLayer* layer = layers[i];
        if (layer->is_activation_layer) {
            // For activation layers, the z_values are the input
            layer->forward(layer->z_values);
        } else {
            layer->forward(current_input);
        }
        current_input = layer->activations;
    }

    cudaFree(device_input);
}

std::vector<float> NeuralNetwork::get_activations() {
    float* activations_host = new float[total_b_z_a];
    cudaError_t err = cudaMemcpy(activations_host, device_activations, total_b_z_a * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        delete[] activations_host;
        throw std::runtime_error("Failed to copy activations from device: " + std::string(cudaGetErrorString(err)));
    }
    
    std::vector<float> result(activations_host, activations_host + total_b_z_a);
    delete[] activations_host;  // Clean up the temporary array
    return result;
}

std::vector<float> NeuralNetwork::get_z_values() {
    float* z_values_host = new float[total_b_z_a];
    cudaError_t err = cudaMemcpy(z_values_host, device_z_values, total_b_z_a * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        delete[] z_values_host;
        throw std::runtime_error("Failed to copy z_values from device: " + std::string(cudaGetErrorString(err)));
    }
    
    std::vector<float> result(z_values_host, z_values_host + total_b_z_a);
    delete[] z_values_host;
    return result;
}

std::vector<float> NeuralNetwork::get_weights() {
    float* weights_host = new float[total_weights];
    cudaError_t err = cudaMemcpy(weights_host, device_weights, total_weights * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        delete[] weights_host;
        throw std::runtime_error("Failed to copy weights from device: " + std::string(cudaGetErrorString(err)));
    }
    
    std::vector<float> result(weights_host, weights_host + total_weights);
    delete[] weights_host;
    return result;
}

std::vector<float> NeuralNetwork::get_biases() {
    float* biases_host = new float[total_b_z_a];
    cudaError_t err = cudaMemcpy(biases_host, device_biases, total_b_z_a * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        delete[] biases_host;
        throw std::runtime_error("Failed to copy biases from device: " + std::string(cudaGetErrorString(err)));
    }
    
    std::vector<float> result(biases_host, biases_host + total_b_z_a);
    delete[] biases_host;
    return result;
}


std::vector<float> NeuralNetwork::get_results() {
    // Get all values from device
    std::vector<float> activations = get_activations();
    std::vector<float> z_values = get_z_values();
    
    // Get number of outputs from last layer
    int num_outputs = layers.back()->num_outputs;
    int offset = total_b_z_a - num_outputs;

    // Create result vector with proper size
    std::vector<float> result(num_outputs);

    // If last layer is activation layer, return last activations
    // Otherwise return last z_values
    if (layers.back()->is_activation_layer) {
        std::copy(activations.begin() + offset, activations.end(), result.begin());
    } else {
        std::copy(z_values.begin() + offset, z_values.end(), result.begin());
    }

    return result;
}

float NeuralNetwork::get_loss(std::vector<float> target) {
    std::vector<float> results = get_results();
    float loss = 0;
    for (int i = 0; i < results.size(); i++) {
        loss += pow(results[i] - target[i], 2);
    }
    return loss / results.size();
}

float NeuralNetwork::get_loss(float target) {
    if (layers.back()->num_outputs != 1) {
        throw std::runtime_error("Only one target value given but last layer has multiple outputs");
    }
    std::vector<float> results = get_results();
    return pow(results[0] - target, 2);
}

void NeuralNetwork::save_model(std::string filename) {
    std::ofstream file(filename);

    // first line is the shape
    // second line is the order of the layers L = Linear, R = ReLU, S = Sigmoid
    // third line is the weights
    // fourth line is the biases

    for (int shape : shape) {
        file << shape << " ";
    }
    file << std::endl;

    for (int i = 0; i < layers.size(); i++) {
        file << layers[i]->type << " ";
    }
    file << std::endl;

    std::vector<float> weights = get_weights();
    for (float weight : weights) {
        file << weight << " ";
    }
    file << std::endl;

    std::vector<float> biases = get_biases();
    for (float bias : biases) {
        file << bias << " ";
    }
    file << std::endl;

    file.close();
}
