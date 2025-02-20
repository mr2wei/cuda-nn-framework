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

__global__ void ADAM_update_weights_kernel(
    float* weights, float* weights_gradient, float* biases, float* biases_gradient,
    float* weights_first_moment, float* weights_second_moment, float* biases_first_moment, float* biases_second_moment,
    float learning_rate, int total_weights, int num_outputs,
    float beta1, float beta2, float epsilon,
    float beta1_pow_iteration, float beta2_pow_iteration
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_weights) {
        weights_first_moment[index] = beta1 * weights_first_moment[index] + (1 - beta1) * weights_gradient[index];
        weights_second_moment[index] = beta2 * weights_second_moment[index] + (1 - beta2) * weights_gradient[index] * weights_gradient[index];
        
        float m_hat = weights_first_moment[index] / (1 - beta1_pow_iteration);
        float v_hat = weights_second_moment[index] / (1 - beta2_pow_iteration);
        weights[index] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }

    if (index < num_outputs) {
        biases_first_moment[index] = beta1 * biases_first_moment[index] + (1 - beta1) * biases_gradient[index];
        biases_second_moment[index] = beta2 * biases_second_moment[index] + (1 - beta2) * biases_gradient[index] * biases_gradient[index];
        
        float m_hat = biases_first_moment[index] / (1 - beta1_pow_iteration);
        float v_hat = biases_second_moment[index] / (1 - beta2_pow_iteration);
        biases[index] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
}


Optimizer::Optimizer(NeuralNetwork* nn, float learning_rate, OptimizerType optimizer_type, LossType loss_type, float beta1, float beta2, float epsilon) {
    this->nn = nn;
    this->learning_rate = learning_rate;

    this->optimizer_type = optimizer_type;
    this->loss_type = loss_type;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->epsilon = epsilon;
    this->adam_iteration = 0;



    cudaMalloc(&device_input_gradient, nn->total_input_gradient * sizeof(float));
    cudaMalloc(&device_weights_gradient, nn->total_weights * sizeof(float));
    cudaMalloc(&device_biases_gradient, nn->total_b_z_a * sizeof(float));
    cudaMalloc(&device_weights_first_moment, nn->total_weights * sizeof(float));
    cudaMalloc(&device_weights_second_moment, nn->total_weights * sizeof(float));
    cudaMalloc(&device_biases_first_moment, nn->total_b_z_a * sizeof(float));
    cudaMalloc(&device_biases_second_moment, nn->total_b_z_a * sizeof(float));

    // check for any errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("Error allocating device memory: " + std::string(cudaGetErrorString(error)));
    }

    // set the first moments to 0
    cudaMemset(device_weights_first_moment, 0, nn->total_weights * sizeof(float));
    cudaMemset(device_weights_second_moment, 0, nn->total_weights * sizeof(float));
    cudaMemset(device_biases_first_moment, 0, nn->total_b_z_a * sizeof(float));
    cudaMemset(device_biases_second_moment, 0, nn->total_b_z_a * sizeof(float));
}

float Optimizer::MSE_loss(std::vector<float> target) {


    std::vector<float> results = nn->get_results();
    float loss = 0;
    for (int i = 0; i < results.size(); i++) {
        loss += pow(results[i] - target[i], 2);
    }
    return loss / results.size();
}

float Optimizer::MAE_loss(std::vector<float> target) {
    std::vector<float> results = nn->get_results();
    float loss = 0;
    for (int i = 0; i < results.size(); i++) {
        loss += abs(results[i] - target[i]);
    }
    return loss / results.size();
}

void Optimizer::MSE_loss_backward(std::vector<float> target, float* loss_gradient) {
    std::vector<float> results = nn->get_results();
    for (int i = 0; i < results.size(); i++) {
        loss_gradient[i] = (2.0f / results.size()) * (results[i] - target[i]);
    }
}

void Optimizer::MAE_loss_backward(std::vector<float> target, float* loss_gradient) {
    std::vector<float> results = nn->get_results();
    for (int i = 0; i < results.size(); i++) {
        if (results[i] > target[i]) {
            loss_gradient[i] = 1.0f / results.size();
        } else if (results[i] < target[i]) {
            loss_gradient[i] = -1.0f / results.size();
        } else {
            loss_gradient[i] = 0;
        }
    }
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
    int weights_gradient_offset = 0, biases_gradient_offset = 0;
    float beta1_pow_iteration = pow(beta1, adam_iteration + 1);
    float beta2_pow_iteration = pow(beta2, adam_iteration + 1);
    
    for (int i = 0; i < nn->layers.size(); i++) {
        if (!nn->layers[i]->is_activation_layer) {
            int total_weights = nn->layers[i]->num_outputs * nn->layers[i]->num_inputs;
            int threads_per_block = 256;
            int blocks = (total_weights + threads_per_block - 1) / threads_per_block;
            ADAM_update_weights_kernel<<<blocks, threads_per_block>>>(
                nn->layers[i]->weights, 
                device_weights_gradient + weights_gradient_offset, 
                nn->layers[i]->biases, 
                device_biases_gradient + biases_gradient_offset,
                device_weights_first_moment + weights_gradient_offset, 
                device_weights_second_moment + weights_gradient_offset, 
                device_biases_first_moment + biases_gradient_offset, 
                device_biases_second_moment + biases_gradient_offset,
                learning_rate, 
                total_weights, 
                nn->layers[i]->num_outputs,
                beta1, 
                beta2, 
                epsilon,
                beta1_pow_iteration,
                beta2_pow_iteration);
            weights_gradient_offset += total_weights;
            biases_gradient_offset += nn->layers[i]->num_outputs;

        }
    }
    
    cudaDeviceSynchronize();
    adam_iteration++;
}


void Optimizer::backward(std::vector<float> target) {

    std::vector<float> results = nn->get_results();
    float* loss_gradient = new float[results.size()];
    switch (loss_type) {
        case LossType::MSE:
            MSE_loss_backward(target, loss_gradient);
            break;
        case LossType::MAE:
            MAE_loss_backward(target, loss_gradient);
            break;
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

    int weights_gradient_offset = nn->total_weights;
    int biases_gradient_offset = nn->total_b_z_a;
    int input_gradient_offset = nn->total_input_gradient;
    for (int i = nn->layers.size() - 1; i >= 0; i--) {
        
        if (!nn->layers[i]->is_activation_layer) {
            weights_gradient_offset -= nn->layers[i]->num_outputs * nn->layers[i]->num_inputs;
            biases_gradient_offset -= nn->layers[i]->num_outputs;
        }
        input_gradient_offset -= nn->layers[i]->num_inputs;

        // std::cout << "weights_gradient_offset: " << weights_gradient_offset << std::endl;
        // std::cout << "biases_gradient_offset: " << biases_gradient_offset << std::endl;
        // std::cout << "input_gradient_offset: " << input_gradient_offset << std::endl;

        nn->layers[i]->backward(current_loss_gradient, 
                                device_input_gradient + input_gradient_offset, 
                                device_weights_gradient + weights_gradient_offset, 
                                device_biases_gradient + biases_gradient_offset);
        current_loss_gradient = device_input_gradient + input_gradient_offset;
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
    // check for any errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("Error zeroing gradients: " + std::string(cudaGetErrorString(error)));
    }
}

void Optimizer::step() {
    switch (optimizer_type) {
        case OptimizerType::SGD:
            SGD_step();
            break;
        case OptimizerType::ADAM:
            ADAM_step();
            break;
    }
}

float Optimizer::get_loss(std::vector<float> target) {
    switch (loss_type) {
        case LossType::MSE:
            return MSE_loss(target);
        case LossType::MAE:
            return MAE_loss(target);
    }
}

float Optimizer::get_loss(float target) {
    std::vector<float> target_vector(1, target);
    return get_loss(target_vector);
}

std::vector<float> Optimizer::get_input_gradient() {
    std::vector<float> input_gradient(nn->total_input_gradient);
    cudaError_t error = cudaMemcpy(input_gradient.data(), device_input_gradient, nn->total_input_gradient * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        throw std::runtime_error("Error copying input gradient to host: " + std::string(cudaGetErrorString(error)));
    }
    return input_gradient;
}


std::vector<float> Optimizer::get_weights_gradient() {
    std::vector<float> weights_gradient(nn->total_weights);
    cudaError_t error =     cudaMemcpy(weights_gradient.data(), device_weights_gradient, nn->total_weights * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        throw std::runtime_error("Error copying weights gradient to host: " + std::string(cudaGetErrorString(error)));
    }
    return weights_gradient;
}

std::vector<float> Optimizer::get_biases_gradient() {
    std::vector<float> biases_gradient(nn->total_b_z_a);
    cudaError_t error = cudaMemcpy(biases_gradient.data(), device_biases_gradient, nn->total_b_z_a * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        throw std::runtime_error("Error copying biases gradient to host: " + std::string(cudaGetErrorString(error)));
    }
    return biases_gradient;
}

std::vector<float> Optimizer::get_weights_first_moment() {
    std::vector<float> weights_first_moment(nn->total_weights);
    cudaMemcpy(weights_first_moment.data(), device_weights_first_moment, nn->total_weights * sizeof(float), cudaMemcpyDeviceToHost);
    return weights_first_moment;
}   

std::vector<float> Optimizer::get_weights_second_moment() {
    std::vector<float> weights_second_moment(nn->total_weights);
    cudaMemcpy(weights_second_moment.data(), device_weights_second_moment, nn->total_weights * sizeof(float), cudaMemcpyDeviceToHost);
    return weights_second_moment;
}   

std::vector<float> Optimizer::get_biases_first_moment() {
    std::vector<float> biases_first_moment(nn->total_b_z_a);
    cudaMemcpy(biases_first_moment.data(), device_biases_first_moment, nn->total_b_z_a * sizeof(float), cudaMemcpyDeviceToHost);
    return biases_first_moment;
}   

std::vector<float> Optimizer::get_biases_second_moment() {
    std::vector<float> biases_second_moment(nn->total_b_z_a);
    cudaMemcpy(biases_second_moment.data(), device_biases_second_moment, nn->total_b_z_a * sizeof(float), cudaMemcpyDeviceToHost);
    return biases_second_moment;
}      
