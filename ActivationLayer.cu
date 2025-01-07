#include "ActivationLayer.hpp"
#include <stdexcept>

__global__ void sigmoid_kernel(float *activations, float *input, int num_inputs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_inputs)
    {
        activations[index] = 1.0 / (1.0 + exp(-input[index]));
    }
}

__global__ void relu_kernel(float *activations, float *input, int num_inputs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_inputs)
    {
        activations[index] = max(0.0f, input[index]);
    }
}

__global__ void sigmoid_backward_kernel(float* input_gradient, float* output_gradient, float* activations, int num_inputs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_inputs) {
        input_gradient[index] = output_gradient[index] * activations[index] * (1 - activations[index]);
    }
}

__global__ void relu_backward_kernel(float* input_gradient, float* output_gradient, float* activations, int num_inputs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_inputs) {
        input_gradient[index] = output_gradient[index] * (activations[index] > 0 ? 1 : 0);
    }
}

void ActivationLayer::forward(float *input)
{
    if (activations == nullptr)
    {
        throw std::runtime_error("Activations array is not allocated/defined");
    }

    int threads_per_block = 256;
    int blocks = (num_inputs + threads_per_block - 1) / threads_per_block;
    switch (activation_type)
    {
    case SIGMOID:
        sigmoid_kernel<<<blocks, threads_per_block>>>(activations, input, num_inputs);
        break;
    case RELU:
        relu_kernel<<<blocks, threads_per_block>>>(activations, input, num_inputs);
        break;
    }

    cudaDeviceSynchronize();
}

void ActivationLayer::backward(float* output_gradient) {
    if (input_gradient == nullptr) {
        throw std::runtime_error("Input gradient array is not allocated/defined");
    }

    int threads_per_block = 256;
    int blocks = (num_inputs + threads_per_block - 1) / threads_per_block;
    switch (activation_type) {
        case SIGMOID:
            sigmoid_backward_kernel<<<blocks, threads_per_block>>>(input_gradient, output_gradient, activations, num_inputs);
            break;
        case RELU:
            relu_backward_kernel<<<blocks, threads_per_block>>>(input_gradient, output_gradient, activations, num_inputs);
            break;
    }

    cudaDeviceSynchronize();    

    // check for any errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("Error in activation_layer_backward: " + std::string(cudaGetErrorString(error)));
    }
}

void ActivationLayer::step(float learning_rate) {
    // do nothing
    return;
}