#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "NNLayer.hpp"
#include <vector>
#include <string>

class NeuralNetwork {
public:
    std::vector<NNLayer*> layers;

    std::vector<int> shape;

    size_t total_weights, total_b_z_a, total_input_gradient;

    float* device_weights;
    float* device_biases;
    float* device_z_values;
    float* device_activations;
    float* device_weights_gradient;
    float* device_biases_gradient;
    float* device_input_gradient;

    NeuralNetwork(std::vector<NNLayer*> layers);
    NeuralNetwork(std::vector<NNLayer*> layers, float* host_weights, float* host_biases);
    ~NeuralNetwork();

    void initialize(std::vector<NNLayer*> layers, float* host_weights, float* host_biases);
    void glorot_uniform_weights(int input_size, int output_size, float* host_weights, int offset);
    void forward(float* input); 
    void backward(std::vector<float> target);
    void backward(float target);
    void step(float learning_rate);
    void zero_gradients();

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