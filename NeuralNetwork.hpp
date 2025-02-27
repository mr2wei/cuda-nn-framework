#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "NNLayer.hpp"
#include <vector>
#include <string>

class NeuralNetwork {
public:
    std::vector<NNLayer*> layers;

    std::vector<int> shape;

    int batch_size;

    size_t total_weights, total_b_z_a, total_input_gradient;

    float* device_weights;
    float* device_biases;
    float* device_z_values;
    float* device_activations;

    NeuralNetwork(std::vector<NNLayer*> layers, int batch_size = 64);
    NeuralNetwork(std::vector<NNLayer*> layers, float* host_weights, float* host_biases, int batch_size = 64);
    ~NeuralNetwork();

    void initialize(std::vector<NNLayer*> layers, float* host_weights, float* host_biases);
    void glorot_uniform_weights(int input_size, int output_size, float* host_weights, int offset);
    void forward(float* input); 
    void forward(std::vector<float> input);
    void forward(std::vector<std::vector<float>> input);

    std::vector<float> get_activations();
    std::vector<float> get_z_values();
    std::vector<float> get_weights();
    std::vector<float> get_biases();
    std::vector<float> get_results();
    
    float get_loss(std::vector<float> target);
    float get_loss(float target);

    void save_model(std::string filename);
};

#endif