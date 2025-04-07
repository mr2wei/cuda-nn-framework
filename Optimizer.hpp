#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include "NeuralNetwork.hpp"

class Optimizer {
private:
    NeuralNetwork* nn;
    float learning_rate;

    float beta1;
    float beta2;
    float epsilon;
    int adam_iteration;

    float* device_input_gradient;
    float* device_weights_gradient;

    float* device_biases_gradient;



    float* device_weights_first_moment;
    float* device_weights_second_moment;
    float* device_biases_first_moment;
    float* device_biases_second_moment;

    void SGD_step();
    void ADAM_step();

    float MSE_loss(std::vector<float> target);
    float MAE_loss(std::vector<float> target);

    void MSE_loss_backward(std::vector<float> target, float* loss_gradient);
    void MAE_loss_backward(std::vector<float> target, float* loss_gradient);

public:
    enum OptimizerType {
        SGD,
        ADAM
    };

    enum LossType {
        MSE,
        MAE
    };

    OptimizerType optimizer_type;
    LossType loss_type;

    Optimizer(NeuralNetwork* nn, float learning_rate, OptimizerType optimizer_type, LossType loss_type, int batch_size = 64, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    void backward(std::vector<float> target);
    void backward(float target);
    void step();
    void zero_grad();

    float get_loss(std::vector<float> target);
    float get_loss(float target);

    std::vector<float> get_input_gradient();
    std::vector<float> get_weights_gradient();
    std::vector<float> get_biases_gradient();
    std::vector<float> get_weights_first_moment();
    std::vector<float> get_weights_second_moment();
    std::vector<float> get_biases_first_moment();
    std::vector<float> get_biases_second_moment();
};

#endif