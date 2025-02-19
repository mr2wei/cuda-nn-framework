#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>
#include "NeuralNetwork.hpp"
#include "NNLayer.hpp"
#include "ActivationLayer.hpp"  
#include "LinearLayer.hpp"

// Function to read CSV data into a vector of floats
std::vector<std::vector<float>> read_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<float>> data;
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }
    
    // Skip the header row
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }
        data.push_back(row);
    }

    return data;
}

int main() {
    // Read data from CSV
    std::vector<std::vector<float>> data = read_csv("cook_county_data/cook_county_train_val_cleaned.csv");

    // Modified to exclude Sale Price column (assuming it's the first column)
    std::vector<std::vector<float>> inputs;
    std::vector<float> targets;
    for (const auto& row : data) {
        // Skip the Sale Price column (first column) and use it as target
        targets.push_back(row[0]);
        std::vector<float> input_row(row.begin() + 1, row.end());
        inputs.push_back(input_row);
    }

    // split data into training and validation
    std::vector<std::vector<float>> training_inputs, validation_inputs;
    std::vector<float> training_targets, validation_targets;
    for (int i = 0; i < inputs.size(); i++) {
        if (i % 5 == 0) {
            validation_inputs.push_back(inputs[i]);
            validation_targets.push_back(targets[i]);
        } else {
            training_inputs.push_back(inputs[i]);
            training_targets.push_back(targets[i]);
        }
    }

    // Create layers for the neural network
    std::vector<NNLayer*> layers = {
        new LinearLayer(3, 1)
    };

    // Initialize the neural network
    NeuralNetwork nn(layers);

    std::cout << "Initial Run:" << std::endl;
    float average_loss = 0;
    for (int i = 0; i < validation_inputs.size(); i++) {
        nn.forward(validation_inputs[i].data());
        average_loss += nn.get_loss(validation_targets[i]);
    }
    average_loss /= validation_inputs.size();
    std::cout << "Average loss: " << average_loss << std::endl;

    // optimise 
    std::cout << "Optimising..." << std::endl;
    std::cout << "Training inputs size: " << training_inputs.size() << std::endl;
    int num_epochs = 10;
    int rand_index = rand() % training_inputs.size();
    for (int i = 0; i < num_epochs; i++) {
        for (int j = 0; j < training_inputs.size(); j++) {
            // std::cout << "Iteration " << j << " with index " << rand_index << std::endl;
            nn.forward(training_inputs[rand_index].data());
            nn.backward(training_targets[rand_index]);
            nn.step(0.00001);
            nn.zero_gradients();
            rand_index = rand() % training_inputs.size();
            float result = nn.get_results()[0];
            if (std::isnan(result)) {
                std::cout << "NaN detected on iteration " << i << " with index " << rand_index << std::endl;
                return 0;
            } else if (std::isinf(result)) {
                std::cout << "Inf detected on iteration " << i << " with index " << rand_index << std::endl;
                return 0;
            }
        }
    }

    // re run the full data to get new average loss
    std::cout << "New Run:" << std::endl;
    average_loss = 0;
    for (int i = 0; i < validation_inputs.size(); i++) {
        nn.forward(validation_inputs[i].data());
        average_loss += nn.get_loss(validation_targets[i]);
    }
    average_loss /= validation_inputs.size();
    std::cout << "New average loss: " << average_loss << std::endl;

    // save model
    nn.save_model("model.txt");

    // Export validation predictions and targets to CSV
    std::ofstream outfile("validation_results.csv");
    outfile << "Predicted,Actual\n";
    
    for (int i = 0; i < validation_inputs.size(); i++) {
        nn.forward(validation_inputs[i].data());
        std::vector<float> prediction = nn.get_results();
        outfile << prediction[0] << "," << validation_targets[i] << "\n";
    }
    
    outfile.close();
    std::cout << "Validation results saved to validation_results.csv" << std::endl;

    // Clean up layers
    for (NNLayer* layer : layers) {
        delete layer;
    }

    return 0;
}


