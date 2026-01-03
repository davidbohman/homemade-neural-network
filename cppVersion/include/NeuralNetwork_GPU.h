#pragma once
#include "NeuralNetwork.h"

class NeuralNetwork_GPU : public NeuralNetwork {
public:
    void initialize(int inputSize, int hiddenSize, int outputSize, float bias) override;
    std::vector<float> forward(const std::vector<float> &input) override;
    void train(const std::vector<float> &input, const std::vector<float> &target, float learning_rate) override;
};