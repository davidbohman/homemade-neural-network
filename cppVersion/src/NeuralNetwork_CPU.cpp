#include "../include/NeuralNetwork_CPU.h"
#include <cassert>
#include <random>


float randomWeight(float scale){
        static std::mt19937 rng(std::random_device{}());
        static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(rng) * scale;
}

// -------- Linear Algebra ----------
std::vector<float> matrix_multiplication(const std::vector<float> &m1, int m1_rows, int m1_cols,
                                         const std::vector<float> &m2, int m2_rows, int m2_cols){
     // mxn * nxm = mxm
    assert(m1_cols == m2_rows && "Matrix dimensions do not match for multiplication");

    std::vector<float> result(m1_rows * m2_cols, 0.0f);

    /*
    [1, 2] * [5] = [(1*5 + 2*6)] =  [17]
    [4, 3]   [6]   [(4*5 + 3*6)]    [38]
    */

    for(int i = 0; i < m1_rows; i++){
        for(int j = 0; j < m2_cols; j++){
            float sum = 0.0f;
            //Calculate dot product
            for(int k = 0; k < m1_cols; k++){
                sum += m1[i * m1_cols + k] * m2[k * m2_cols + j];
            }
            result[i * m2_cols + j] = sum;
        }
    }
    return result;
}

//Used for finding hidden layer error term
std::vector<float> matmul_transpose_left(const std::vector<float>& m, int rows, int cols, const std::vector<float>& v){
    std::vector<float> result(cols, 0.0f);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[j] += m[i * cols + j] * v[i];
    return result;
}

void matrix_addition(std::vector<float> &m1, const std::vector<float> &m2, int rows, int cols){
    assert(m1.size() == m2.size() && "Matrix dimensions do not match for addition");
    assert(m1.size() == rows * cols && "m1 size does not match rows*cols");

    for(int i = 0; i < rows*cols; i++) m1[i] += m2[i];
}

void matrix_subtraction(std::vector<float> &m1, const std::vector<float> &m2, int rows, int cols){
    assert(m1.size() == m2.size() && "Matrix dimensions do not match for subtraction");
    assert(m1.size() == rows * cols && "m1 size does not match rows*cols");

    for(int i = 0; i < rows * cols; i++) m1[i] -= m2[i];
}

void multiply_with_constant(std::vector<float> &m1, float k){
    for(auto &e : m1) e = e * k;
}

std::vector<float> elementwise_multiply(std::vector<float> &m1, std::vector<float> &m2){

    assert(m1.size() == m2.size() && "Matrix dimensions do not match for elementwise multiplication");

    std::vector<float> result(m1.size(), 0.0f);

    for(size_t i = 0; i < m1.size(); i++) result[i] = m1[i]*m2[i]; 

    return result;
}

void elementwise_multiply_inplace(std::vector<float> &m1, const std::vector<float> &m2) {
    assert(m1.size() == m2.size() && "Matrix dimensions do not match for elementwise multiplication");
    for(size_t i = 0; i < m1.size(); i++)
        m1[i] *= m2[i];
}

//------- Activation functions ---------

float relu(float x){
    return x < 0 ? 0 : x;
}

float relu_derivate(float x){
    return x > 0 ? 1.0f : 0.0f;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float sigmoid_derivate(float x){
    float s = sigmoid(x);
    return s * (1.0f - s);
}

std::vector<float> softmax(const std::vector<float> &v){
        std::vector<float> result(v.size());

    float max_val = *std::max_element(v.begin(), v.end());

    // Compute exp(v_i - max)
    float sum = 0.0f;
    for (size_t i = 0; i < v.size(); i++) {
        result[i] = std::exp(v[i] - max_val);
        sum += result[i];
    }

    // Normalize
    for (size_t i = 0; i < v.size(); i++) {
        result[i] /= sum;
    }

    return result;
}

// --------- Cost Gradiant ------------

struct Partials{
    std::vector<float> weight_partial;
    std::vector<float> bias_partial;
};

Partials cost_gradient(const std::vector<float> &error_term, int error_rows, const std::vector<float> &prev_layer_output, int plo_rows){
    Partials result;
    result.weight_partial = 
        matrix_multiplication(error_term, error_rows, 1, prev_layer_output, 1, plo_rows);

    result.bias_partial = error_term; 

    return result;
}

// --------- Class functions ----------

void NeuralNetwork_CPU::initialize(int inputSize, int hiddenSize, int outputSize, float bias) {
    this -> inputSize = inputSize;
    this -> hiddenSize = hiddenSize;
    this -> outputSize = outputSize;

    //Weight matrixes (stored as a long vector)
    weights_hidden.resize(hiddenSize * inputSize);
    weights_output.resize(outputSize * hiddenSize);

    //Assign random values at start:
    float scale = sqrt(1.0f / inputSize); //Normalising values
    for (auto &w : weights_hidden) w = randomWeight(scale);
    for (auto &w : weights_output) w = randomWeight(scale);

    bias_hidden.resize(hiddenSize);
    bias_output.resize(outputSize); 
    for(auto &b : bias_hidden) b = bias;
    for(auto &b : bias_output) b = bias;

    //Need to save for backpropagation
    std::vector<float> layer_0_raw_output;
    std::vector<float> layer_0_output;

    //Set activation function (Can swap to relu)
    activation_func = sigmoid;
    activation_derivate_func = sigmoid_derivate;
}

std::vector<float> NeuralNetwork_CPU::forward(const std::vector<float> &input) { 

    // Creating hidden layer

    //weight0 @ input
    layer_0_output = 
        matrix_multiplication(weights_hidden, hiddenSize, inputSize,input, inputSize, 1); // input is 784x1 vector
    //(weight0 @ input) - bias
    matrix_subtraction(layer_0_output, bias_hidden, hiddenSize, 1);

    //Copy used later for backpropagation
    layer_0_raw_output = layer_0_output;

    //Activation
    for(auto &e : layer_0_output) e = activation_func(e);

    // Creating output layer
    std::vector<float> layer_1_output = 
        matrix_multiplication(weights_output, outputSize, hiddenSize, layer_0_output, hiddenSize, 1);

    matrix_subtraction(layer_1_output, bias_output, 10, 1);

    //Output activation
    return softmax(layer_1_output); 

}
void NeuralNetwork_CPU::train(const std::vector<float> &input, uint8_t target, float learning_rate) {
    
    std::vector<float> target_vector(outputSize, 0);
    target_vector[target] = 1.0f;

     //Getting error term for output vector (10x1) vector
     std::vector<float> output_error = forward(input);
     matrix_subtraction(output_error, target_vector, outputSize, 1);
    
     // Error term for hidden layer (128 x 1)


    //Transposes weights_output matrix by swaping rows <-> cols, to match and enable multiplication with output_error
    std::vector<float> hidden_error =
        matmul_transpose_left(weights_output, outputSize, hiddenSize, output_error);
        

    for(auto &e : layer_0_raw_output) e = activation_derivate_func(e);
    elementwise_multiply_inplace(hidden_error, layer_0_raw_output);
    // layer_0_raw_output is the output from the first layer without activation
    //hidden_error now contains (W(n+1).T @ output_error) * activation_derivate(layer_0_raw)

    Partials w_and_b_0 = cost_gradient(hidden_error, hiddenSize, input, inputSize);
    Partials w_and_b_1 = cost_gradient(output_error, outputSize, layer_0_output, hiddenSize);
    
    // --- Update weights and bias ---

    //Hidden layer weight update
    multiply_with_constant(w_and_b_0.weight_partial, learning_rate);
    matrix_subtraction(weights_hidden, w_and_b_0.weight_partial, hiddenSize, inputSize);
    //Output layer weight update
    multiply_with_constant(w_and_b_1.weight_partial, learning_rate);
    matrix_subtraction(weights_output, w_and_b_1.weight_partial, outputSize, hiddenSize);
    
    //hidden bias update
    multiply_with_constant(w_and_b_0.bias_partial, learning_rate);
    matrix_subtraction(bias_hidden, w_and_b_0.bias_partial, hiddenSize, 1);
    //Output bias update
    multiply_with_constant(w_and_b_1.bias_partial, learning_rate);
    matrix_subtraction(bias_output, w_and_b_1.bias_partial, outputSize, 1);
}