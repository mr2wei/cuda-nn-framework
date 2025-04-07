#include "ActivationLayer.hpp"
#include <stdexcept>

__global__ void sigmoid_kernel(float *activations, float *input, int num_inputs, int batch_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_inputs * batch_size) {
        int batch_index = index / num_inputs;
        int input_index = index % num_inputs;

        activations[index] = 1.0 / (1.0 + exp(-input[batch_index * num_inputs + input_index]));
    }
}

__global__ void relu_kernel(float *activations, float *input, int num_inputs, int batch_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_inputs * batch_size) {
        int batch_index = index / num_inputs;
        int input_index = index % num_inputs;

        activations[index] = max(0.0f, input[batch_index * num_inputs + input_index]);
    }
}

__global__ void sigmoid_backward_kernel(float* input_gradient, float* output_gradient, float* activations, int num_inputs, int batch_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_inputs * batch_size) {
        int batch_index = index / num_inputs;
        int input_index = index % num_inputs;

        input_gradient[index] = output_gradient[batch_index * num_inputs + input_index] * activations[batch_index * num_inputs + input_index] * (1 - activations[batch_index * num_inputs + input_index]);
    }
}

__global__ void relu_backward_kernel(float* input_gradient, float* output_gradient, float* activations, int num_inputs, int batch_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_inputs * batch_size) {
        int batch_index = index / num_inputs;
        int input_index = index % num_inputs;

        input_gradient[index] = output_gradient[batch_index * num_inputs + input_index] * (activations[batch_index * num_inputs + input_index] > 0 ? 1 : 0);
    }
}

void ActivationLayer::forward(float *input, int batch_size)
{
    if (activations == nullptr)
    {
        throw std::runtime_error("Activations array is not allocated/defined");
    }

    int threads_per_block = 256;
    int blocks = (num_inputs * batch_size + threads_per_block - 1) / threads_per_block;
    switch (activation_type)
    {
    case SIGMOID:
        sigmoid_kernel<<<blocks, threads_per_block>>>(activations, input, num_inputs, batch_size);
        break;
    case RELU:
        relu_kernel<<<blocks, threads_per_block>>>(activations, input, num_inputs, batch_size);
        break;
    }

    cudaDeviceSynchronize();
}

void ActivationLayer::backward(float* output_gradient, float* input_gradient, float* weights_gradient, float* biases_gradient, int batch_size) {
    if (input_gradient == nullptr) {
        throw std::runtime_error("Input gradient array is not allocated/defined");
    }

    int threads_per_block = 256;
    int blocks = (num_inputs * batch_size + threads_per_block - 1) / threads_per_block;
    switch (activation_type) {
        case SIGMOID:
            sigmoid_backward_kernel<<<blocks, threads_per_block>>>(input_gradient, output_gradient, activations, num_inputs, batch_size);
            break;
        case RELU:
            relu_backward_kernel<<<blocks, threads_per_block>>>(input_gradient, output_gradient, activations, num_inputs, batch_size);
            break;
    }

    cudaDeviceSynchronize();    

    // check for any errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("Error in activation_layer_backward: " + std::string(cudaGetErrorString(error)));
    }
}

