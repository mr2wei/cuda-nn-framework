#include "LinearLayer.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdexcept>
// #include <iostream>

#define TILE_SIZE 32

__global__ void linear_layer_kernal(
    float* weight_matrix, float* biases, 
    float* x_inputs, float* z_values,
    int num_output_neurons, int num_input_neurons,
    int batch_size
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_output_neurons * batch_size)
    {
        int batch_index = index / num_output_neurons;
        int neuron_index = index % num_output_neurons;

        z_values[index] = biases[neuron_index];
        for (int i = 0; i < num_input_neurons; i++)
        {
            z_values[index] += weight_matrix[neuron_index * num_input_neurons + i] * 
                               x_inputs[batch_index * num_input_neurons + i];
        }
    }
}

__global__ void transpose(float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load data into shared memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();

    // Transpose and write back to global memory
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void linear_layer_backward_weights_kernal(
    float* weights_gradient, float* output_gradient,
    float* prev_input, int num_output_neurons, 
    int num_input_neurons, int batch_size
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_output_neurons * num_input_neurons) {
        int output_idx = index / num_input_neurons;
        int input_idx = index % num_input_neurons;

        for (int i = 0; i < batch_size; i++) {
            weights_gradient[index] += output_gradient[i * num_output_neurons + output_idx] * prev_input[i * num_input_neurons + input_idx];
        }
        weights_gradient[index] /= batch_size;
    }
}

__global__ void linear_layer_backward_inputs_kernal(
    float* input_gradient, float* output_gradient,
    float* weights_transposed, int num_output_neurons, 
    int num_input_neurons, int batch_size
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_input_neurons * batch_size) {
        int batch_index = index / num_input_neurons;
        int input_index = index % num_input_neurons;

        input_gradient[index] = 0;

        for (int i = 0; i < num_output_neurons; i++) {
            input_gradient[index] += weights_transposed[input_index * num_output_neurons + i] * output_gradient[batch_index * num_output_neurons + i];
        }
    }
}

__global__ void linear_layer_backward_biases_kernal(
    float* biases_gradient, float* output_gradient,
    int num_output_neurons, int batch_size
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_output_neurons) {
        for (int i = 0; i < batch_size; i++) {
            biases_gradient[index] += output_gradient[i * num_output_neurons + index];
        }
        biases_gradient[index] /= batch_size;
    }
}

void LinearLayer::forward(float* input, int batch_size) {
    // if the arrays are not allocated/defined, raise an error
    if (weights == nullptr || biases == nullptr || z_values == nullptr || activations == nullptr) {
        throw std::runtime_error("Arrays are not allocated/defined");
    }
    // prev_input = input;
    cudaMemcpy(prev_input, input, num_inputs * sizeof(float) * batch_size, cudaMemcpyDeviceToDevice);
    linear_layer_kernal<<< (num_outputs * batch_size + 255) / 256, 256 >>>(weights, biases, input, z_values, num_outputs, num_inputs, batch_size);
    cudaDeviceSynchronize();
}

void LinearLayer::backward(float* output_gradient, float* input_gradient, float* weights_gradient, float* biases_gradient, int batch_size) {
    // if the arrays are not allocated/defined, raise an error
    if (weights_gradient == nullptr || biases_gradient == nullptr || input_gradient == nullptr) {
        throw std::runtime_error("Arrays are not allocated/defined");
    }

    int threads_per_block = 256;
    int blocks = (num_outputs + threads_per_block - 1) / threads_per_block;

    linear_layer_backward_biases_kernal<<<blocks, threads_per_block>>>(biases_gradient, output_gradient, num_outputs, batch_size);
    cudaDeviceSynchronize();
    // error = cudaGetLastError();
    // if (error != cudaSuccess) {
    //     throw std::runtime_error("Error in backward biases kernel: " + std::string(cudaGetErrorString(error)));
    // }


    // bias gradient is just the output gradient
    // cudaError_t error = cudaMemcpy(biases_gradient, output_gradient, num_outputs * sizeof(float), cudaMemcpyDeviceToDevice); // this only works for single item
    
    // TODO: add batching current implementation only works for single item

    // if (error != cudaSuccess) {
    //     throw std::runtime_error("Error copying biases gradient: " + std::string(cudaGetErrorString(error)));
    // }

    // transpose weights
    float* weights_transposed;
    error = cudaMalloc(&weights_transposed, num_outputs * num_inputs * sizeof(float));
    // if (error != cudaSuccess) {
    //     throw std::runtime_error("Error allocating weights_transposed: " + std::string(cudaGetErrorString(error)));
    // }

    // got the transpose kernel from somewhere online
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((num_inputs + TILE_SIZE - 1) / TILE_SIZE, (num_outputs + TILE_SIZE - 1) / TILE_SIZE);
    transpose<<<gridSize, blockSize>>>(weights, weights_transposed, num_inputs, num_outputs);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaFree(weights_transposed);
        throw std::runtime_error("Error in transpose kernel: " + std::string(cudaGetErrorString(error)));
    }

    // Print prev_input values
    float* host_prev_input = new float[num_inputs];
    cudaMemcpy(host_prev_input, prev_input, num_inputs * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "prev_input: ";
    // for (int i = 0; i < num_inputs; i++) {
    //     std::cout << host_prev_input[i] << ", ";
    // }
    // std::cout << std::endl;
    // delete[] host_prev_input;

    blocks = (num_outputs * num_inputs + threads_per_block - 1) / threads_per_block;

    linear_layer_backward_weights_kernal<<<blocks, threads_per_block>>>(weights_gradient, output_gradient, prev_input, num_outputs, num_inputs, batch_size);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaFree(weights_transposed);
        throw std::runtime_error("Error in backward weights kernel: " + std::string(cudaGetErrorString(error)));
    }

    blocks = (num_inputs * batch_size + threads_per_block - 1) / threads_per_block;

    linear_layer_backward_inputs_kernal<<<blocks, threads_per_block>>>(input_gradient, output_gradient, weights_transposed, num_outputs, num_inputs, batch_size);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaFree(weights_transposed);
        throw std::runtime_error("Error in backward inputs kernel: " + std::string(cudaGetErrorString(error)));
    }

    error = cudaFree(weights_transposed);
    if (error != cudaSuccess) {
        throw std::runtime_error("Error freeing weights_transposed: " + std::string(cudaGetErrorString(error)));
    }
}
