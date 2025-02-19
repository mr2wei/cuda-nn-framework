#include "Optimizer.hpp"
#include <stdexcept>
#include <string>
#include <vector>

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

void Optimizer::SGD_step() {
    int weights_gradient_offset = 0, biases_gradient_offset = 0;
    for (int i = 0; i < nn->layers.size(); i++) {
        if (!nn->layers[i]->is_activation_layer) {
            int total_weights = nn->layers[i]->num_outputs * nn->layers[i]->num_inputs;
            int threads_per_block = 256;
            int blocks = (total_weights + threads_per_block - 1) / threads_per_block;
            GD_update_weights_kernel<<<blocks, threads_per_block>>>(
                nn->layers[i]->weights, 
                device_weights_gradient + weights_gradient_offset, 
                nn->layers[i]->biases, 
                device_biases_gradient + biases_gradient_offset, 
                learning_rate, 
                total_weights, 
                nn->layers[i]->num_outputs);
            cudaDeviceSynchronize();
            weights_gradient_offset += total_weights;
            biases_gradient_offset += nn->layers[i]->num_outputs;
        }
    }
}

void Optimizer::ADAM_step() {
    // TODO: Implement ADAM step
}

void Optimizer::backward(std::vector<float> target) {
    std::vector<float> results = nn->get_results();
    float* loss_gradient = new float[results.size()];
    for (int i = 0; i < results.size(); i++) {
        loss_gradient[i] = (2.0f / results.size()) * (results[i] - target[i]);
    }

    float* device_loss_gradient;
    cudaError_t error = cudaMalloc(&device_loss_gradient, results.size() * sizeof(float));
    if (error != cudaSuccess) {
        delete[] loss_gradient;
        throw std::runtime_error("Error allocating device_loss_gradient: " + std::string(cudaGetErrorString(error)));
    }

    error = cudaMemcpy(device_loss_gradient, loss_gradient, results.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(device_loss_gradient);
        delete[] loss_gradient;
        throw std::runtime_error("Error copying loss gradient to device: " + std::string(cudaGetErrorString(error)));
    }

    float* current_loss_gradient = device_loss_gradient;

    int weights_gradient_offset = 0, biases_gradient_offset = 0, input_gradient_offset = 0;
    for (int i = nn->layers.size() - 1; i >= 0; i--) {
        nn->layers[i]->backward(current_loss_gradient, 
                                device_input_gradient + input_gradient_offset, 
                                device_weights_gradient + weights_gradient_offset, 
                                device_biases_gradient + biases_gradient_offset);
        current_loss_gradient = device_input_gradient + input_gradient_offset;
        if (!nn->layers[i]->is_activation_layer) {
            weights_gradient_offset += nn->layers[i]->num_outputs * nn->layers[i]->num_inputs;
            biases_gradient_offset += nn->layers[i]->num_outputs;
        }
        input_gradient_offset += nn->layers[i]->num_inputs;
    }

    error = cudaFree(device_loss_gradient);
    if (error != cudaSuccess) {
        delete[] loss_gradient;
        throw std::runtime_error("Error freeing device_loss_gradient: " + std::string(cudaGetErrorString(error)));
    }
    delete[] loss_gradient;
}

void Optimizer::backward(float target) {
    std::vector<float> target_vector(1, target);
    backward(target_vector);
}

void Optimizer::zero_grad() {
    cudaMemset(device_input_gradient, 0, nn->total_input_gradient * sizeof(float));
    cudaMemset(device_weights_gradient, 0, nn->total_weights * sizeof(float));
    cudaMemset(device_biases_gradient, 0, nn->total_b_z_a * sizeof(float));
    cudaMemset(device_weights_first_moment, 0, nn->total_weights * sizeof(float));
    cudaMemset(device_weights_second_moment, 0, nn->total_weights * sizeof(float));
    cudaMemset(device_biases_first_moment, 0, nn->total_b_z_a * sizeof(float));
    cudaMemset(device_biases_second_moment, 0, nn->total_b_z_a * sizeof(float));

    // check for any errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("Error zeroing gradients: " + std::string(cudaGetErrorString(error)));
    }
}

void Optimizer::step() {
    if (optimizer_type == OptimizerType::SGD) {
        SGD_step();
    } else if (optimizer_type == OptimizerType::ADAM) {
        ADAM_step();
    }
}
