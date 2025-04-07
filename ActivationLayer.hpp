#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include "NNLayer.hpp"

class ActivationLayer : public NNLayer {
public:
    enum ActivationType {
        SIGMOID = 'S',
        RELU = 'R',
    };

    ActivationType activation_type;

    ActivationLayer(int inputs, ActivationType type)
    : NNLayer(inputs, type), activation_type(type)
    {}

    void forward(float* input, int batch_size) override;
    void backward(float* output_gradient, float* input_gradient, float* weights_gradient, float* biases_gradient, int batch_size);
};

#endif