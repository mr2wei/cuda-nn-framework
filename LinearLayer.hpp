#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP

#include "NNLayer.hpp"


class LinearLayer : public NNLayer {
public:
    LinearLayer(int inputs, int outputs)
    : NNLayer(inputs, outputs, 'L')
    {}

    void forward(float* input) override;
    void backward(float* output_gradient) override;
    void step(float learning_rate) override;
};

#endif