#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP

#include "NNLayer.hpp"


class LinearLayer : public NNLayer {
public:
    LinearLayer(int inputs, int outputs)
    : NNLayer(inputs, outputs, 'L')
    {}

    void forward(float* input, int batch_size) override;
    void backward(float* output_gradient, float* input_gradient, float* weights_gradient, float* biases_gradient, int batch_size);
};

#endif