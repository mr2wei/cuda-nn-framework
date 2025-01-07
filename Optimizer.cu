#include "Optimizer.hpp"

Optimizer::Optimizer(NeuralNetwork* nn, float learning_rate, OptimizerType optimizer_type) {
    this->nn = nn;
    this->learning_rate = learning_rate;
    this->optimizer_type = optimizer_type;

    cudaMalloc(&device_input_gradient, nn->total_input_gradient * sizeof(float));
    cudaMalloc(&device_weights_gradient, nn->total_weights * sizeof(float));
    cudaMalloc(&device_biases_gradient, nn->total_b_z_a * sizeof(float));
    cudaMalloc(&device_weights_first_moment, nn->total_weights * sizeof(float));
    cudaMalloc(&device_weights_second_moment, nn->total_weights * sizeof(float));
    cudaMalloc(&device_biases_first_moment, nn->total_b_z_a * sizeof(float));
    cudaMalloc(&device_biases_second_moment, nn->total_b_z_a * sizeof(float));
}

__global__ void GD_update_weights_kernel(
    float* weights, float* weights_gradient, float* biases, float* biases_gradient,
    float learning_rate, int total_weights, int num_outputs
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_weights) {
        weights[index] -= learning_rate * weights_gradient[index];
    }

    if (index < num_outputs) {
        biases[index] -= learning_rate * biases_gradient[index];
    }
}

void Optimizer::SGD_step() {
    for (int i = 0; i < nn->layers.size(); i++) {
        int total_weights = nn->layers[i]->num_outputs * nn->layers[i]->num_inputs;
        int threads_per_block = 256;
        int blocks = (total_weights + threads_per_block - 1) / threads_per_block;
        GD_update_weights_kernel<<<blocks, threads_per_block>>>(nn->layers[i]->weights, nn->layers[i]->weights_gradient, nn->layers[i]->biases, nn->layers[i]->biases_gradient, learning_rate, total_weights, nn->layers[i]->num_outputs);
        cudaDeviceSynchronize();
    }
}

void Optimizer::ADAM_step() {
    // TODO: Implement ADAM step
}

void Optimizer::zero_grad() {
    
}

void Optimizer::step() {
    if (optimizer_type == OptimizerType::SGD) {
        SGD_step();
    } else if (optimizer_type == OptimizerType::ADAM) {
        ADAM_step();
    }
}
