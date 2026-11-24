#pragma once
#include <vector>

//Time measurements
extern double forwarding_time;
extern double back_prop_time; 
extern double matmul_time;
extern double transpose_time;
extern double softMax_time;


class NeuralNetwork {
public:
    ~NeuralNetwork() = default;

    // Initializing Network
    void initialize(std::vector<int>layers, float bias);

    // Forward propagation
    std::vector<float> forward(const std::vector<float> &input);

    // Batched forwarding + Back propagation + update weights
    void train(const std::vector<float>& X, const std::vector<uint8_t>& targets, size_t batchSize, float learingRate);

private:
    // Variables
    uint inputSize;

    struct Layer{
        uint layer_size;
        std::vector<float> weights;
        std::vector<float> bias;
        std::vector<float> activated;
        std::vector<float> pre;
        std::vector<float> error;
    };

    std::vector<Layer>hiddenLayers;

    //Activation function
    float(*activationFunc)(float);
    float(*activationDerivativeFunc)(float);  
};