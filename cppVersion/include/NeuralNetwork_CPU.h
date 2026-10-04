#pragma once
#include "NeuralNetwork.h"


extern double forwarding_time;
extern double back_prop_time; 
extern double matmul_time;

class NeuralNetwork_CPU : public NeuralNetwork {
public:
    void initialize(int inputSize, int hiddenSize, int outputSize, float bias) override;
    std::vector<float> forward(const std::vector<float> &input) override;
    void train(const std::vector<float> &input, uint8_t target, float learning_rate) override;

private:
    // Variables
    int inputSize;
    int hiddenSize;
    int outputSize;

    std::vector<float> weights_hidden;
    std::vector<float> weights_output;

    std::vector<float> bias_hidden;
    std::vector<float> bias_output;

    std::vector<float> layer_0_output;
    std::vector<float> layer_0_raw_output;

    //Activation function
    float(*activation_func)(float);
    float(*activation_derivate_func)(float);

    //



};