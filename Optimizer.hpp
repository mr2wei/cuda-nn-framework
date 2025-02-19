#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include "NeuralNetwork.hpp"

class Optimizer {
private:
    NeuralNetwork* nn;
    float learning_rate;

    float* device_input_gradient;
    float* device_weights_gradient;
    float* device_biases_gradient;

    float* device_weights_first_moment;
    float* device_weights_second_moment;
    float* device_biases_first_moment;
    float* device_biases_second_moment;

    void SGD_step();
    void ADAM_step();

public:
    enum OptimizerType {
        SGD,
        ADAM
    };
    
    OptimizerType optimizer_type;

    Optimizer(NeuralNetwork* nn, float learning_rate, OptimizerType optimizer_type);
    void backward(std::vector<float> target);
    void backward(float target);
    void step();
    void zero_grad();
};

#endif