#pragma once
#include <vector>

//Time measurements
extern double forwarding_time;
extern double back_prop_time; 
extern double matmul_time;



class NeuralNetwork {
public:
    ~NeuralNetwork() = default;

    // Initializing Network
    void initialize(int inputSize, int hiddenSize, int outputSize, float bias);

    // Forward propagation
    std::vector<float> forward(const std::vector<float> &input);

    // Back propagation + update weights
    void train(const std::vector<float> &input, uint8_t target, float learning_rate);

private:
    // Variables
    int inputSize;
    int hiddenSize;
    int outputSize;

    std::vector<float> weights_hidden;
    std::vector<float> weights_output;

    std::vector<float> bias_hidden;
    std::vector<float> bias_output;

    std::vector<float> hidden_act;
    std::vector<float> hidden_pre;

    //Activation function
    float(*activation_func)(float);
    float(*activation_derivative_func)(float);

    //


};