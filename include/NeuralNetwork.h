#pragma once
#include <vector>

class NeuralNetwork {
public:
    virtual ~NeuralNetwork() = default;

    // Initializing Network
    virtual void initialize(int inputSize, int hiddenSize, int outputSize, float bias) = 0;

    // Forward propagation
    virtual std::vector<float> forward(const std::vector<float> &input) = 0;

    // Back propagation + update weights
    virtual void train(const std::vector<float> &input, uint8_t target, float learning_rate) = 0;
};