#include "../include/NeuralNetwork.h"
#include <cassert>
#include <chrono>
#include <random>

#define RELU

double forwarding_time = 0.0;
double back_prop_time = 0.0;
double matmul_time = 0.0;

// -------- UTILS -----------------

struct ScopedTimer {
    double& acc;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    explicit ScopedTimer(double& a) : acc(a) {}
    ~ScopedTimer() { acc += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count(); }
};

float randomWeight(float scale){
        static std::mt19937 rng(std::random_device{}());
        static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        return dist(rng) * scale;
}

// -------- Linear Algebra ----------


std::vector<float> matVecMultiplication(const std::vector<float>& W, int rows, int cols, 
                                        const std::vector<float>& x) 
{
    assert(x.size() == static_cast<size_t>(cols));
    ScopedTimer t(matmul_time);
    std::vector<float> y(rows);
    for (int i = 0; i < rows; ++i) {
        const float* row = &W[i * cols];
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (int k = 0; k < cols; ++k) sum += row[k] * x[k];
        y[i] = sum;
    }
    return y;
}

// W -= lr * a * b^T   (outer-product gradient step, no temporaries)
void apply_outer_update(std::vector<float>& W, const std::vector<float>& a,
                        const std::vector<float>& b, float lr) {
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            W[i * b.size() + j] -= lr * a[i] * b[j];
}

// y += alpha * x
void axpy(std::vector<float>& y, const std::vector<float>& x, float alpha) {
    assert(y.size() == x.size());
    for (size_t i = 0; i < y.size(); ++i) y[i] += alpha * x[i];
}

//Used for finding hidden layer error term
std::vector<float> matvec_transposed(const std::vector<float>& m, int rows, int cols, const std::vector<float>& v){
    std::vector<float> result(cols, 0.0f);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[j] += m[i * cols + j] * v[i];
    return result;
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


// --------- Class functions ----------

void NeuralNetwork::initialize(int inputSize, int hiddenSize, int outputSize, float bias) {
    this -> inputSize = inputSize;
    this -> hiddenSize = hiddenSize;
    this -> outputSize = outputSize;

    //Weight matrixes (stored as a long vector)
    weights_hidden.resize(hiddenSize * inputSize);
    weights_output.resize(outputSize * hiddenSize);

    //Assign random values at start:
    float scale_hidden = sqrt(1.0f / inputSize); //Normalising values
    float scale_output = sqrt(1.0f / hiddenSize);
    for (auto &w : weights_hidden) w = randomWeight(scale_hidden);
    for (auto &w : weights_output) w = randomWeight(scale_output);

    bias_hidden.resize(hiddenSize);
    bias_output.resize(outputSize); 
    for(auto &b : bias_hidden) b = bias;
    for(auto &b : bias_output) b = bias;

    //Need to save for backpropagation
    std::vector<float> layer_0_raw_output;
    std::vector<float> layer_0_output;

    //Set activation function (Can swap to relu)

#ifdef RELU
    activation_func = relu;
    activation_derivative_func = relu_derivate;
#else
    activation_func = sigmoid;
    activation_derivative_func = sigmoid_derivate;
#endif
}

std::vector<float> NeuralNetwork::forward(const std::vector<float> &input) { 

    //Adds measurment when "ScopedTimer" object runs out of scope
    ScopedTimer t(forwarding_time); 


    hidden_pre = matVecMultiplication(weights_hidden, hiddenSize, inputSize, input);
    axpy(hidden_pre, bias_hidden, 1.0f);          // add bias

    hidden_act = hidden_pre;                      // copy, then activate in place
    for (auto& e : hidden_act) e = activation_func(e);

    std::vector<float> logits = matVecMultiplication(weights_output, outputSize, hiddenSize, hidden_act);
    axpy(logits, bias_output, 1.0f);

    return softmax(logits);
}

void NeuralNetwork::train(const std::vector<float> &input, uint8_t target, float learning_rate) {

    assert(target < outputSize);

    std::vector<float> out_err = forward(input);

    ScopedTimer t(back_prop_time);

    out_err[target] -= 1.0f;   // softmax + cross-entropy gradient: p - onehot

    // Compute hidden error BEFORE updating weights_output
    std::vector<float> hidden_err = matvec_transposed(weights_output, outputSize, hiddenSize, out_err);
    for (int i = 0; i < hiddenSize; ++i)
        hidden_err[i] *= activation_derivative_func(hidden_pre[i]);

    apply_outer_update(weights_output, out_err,    hidden_act, learning_rate);
    apply_outer_update(weights_hidden, hidden_err, input,      learning_rate);
    axpy(bias_output, out_err,    -learning_rate);
    axpy(bias_hidden, hidden_err, -learning_rate);

}