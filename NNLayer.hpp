#ifndef NNLAYER_HPP
#define NNLAYER_HPP

class NNLayer {
public:
    int num_inputs;
    int num_outputs;
    float* weights; // all of these are pointers to a centralised storage in an overarching management method/class
    float* biases;
    float* z_values;
    float* activations;
    float* prev_input;
    bool is_activation_layer;
    char type;

    NNLayer(int inputs, int outputs, char type)
    : num_inputs(inputs), num_outputs(outputs), is_activation_layer(false), type(type)
    {}

    NNLayer(int inputs, char type)
    : num_inputs(inputs), num_outputs(inputs), is_activation_layer(true), type(type)
    {} // only for activation functions

    virtual void forward(float* input) = 0;
    virtual void backward(float* output_gradient, float* input_gradient, float* weights_gradient, float* biases_gradient) = 0;
};

#endif