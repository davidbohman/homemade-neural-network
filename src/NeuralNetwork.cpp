#include "../include/NeuralNetwork.h"
#include <cassert>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

#define RELU

using namespace std;

double forwarding_time = 0.0;
double back_prop_time = 0.0;
double matmul_time = 0.0;
double transpose_time = 0.0;
double softMax_time = 0.0;


// -------- UTILS -----------------

struct ScopedTimer {
    double& acc;
    chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
    explicit ScopedTimer(double& a) : acc(a) {}
    ~ScopedTimer() { acc += chrono::duration<double>(chrono::high_resolution_clock::now() - start).count(); }
};

float randomWeight(float scale){
        static mt19937 rng(random_device{}());
        static uniform_real_distribution<float> dist(-1.0f, 1.0f);
        return dist(rng) * scale;
}

// -------- Linear Algebra ----------


// A x B = C
void matmul(const vector<float>& A, size_t aRows, const vector<float>& B, size_t bRows, vector<float>& result){

    ScopedTimer t(matmul_time);
    assert(A.size() % aRows == 0);
    assert(B.size() % bRows == 0);

    size_t aCols = A.size() / aRows;
    size_t bCols = B.size() / bRows;

    //Make sure result doens't contain garbage
    std::fill(result.begin(), result.end(), 0.0f);

    assert(aCols == bRows);
    assert(result.size() == aRows * bCols);
    //Should be optimized!
    #pragma omp parallel for
    for (size_t i = 0; i < aRows; i++) {          // each row of A
        for (size_t k = 0; k < bRows; k++) {        //each row of B
            float a = A[i * aCols + k];
            #pragma omp simd
            for (size_t j = 0; j < bCols; j++) {    //each col of B
                result[i * bCols + j] += a * B[k * bCols + j]; 
            }
        }
    }
}

// out = Mᵀ, where M is rows x cols (row-major)
void transpose(const vector<float>& M, size_t rows, size_t cols, vector<float>& out) {
    ScopedTimer t(transpose_time);
    out.resize(static_cast<size_t>(rows) * cols);
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            out[j * rows + i] = M[i * cols + j];
}


// y += alpha * x
void axpy(vector<float>& y, const vector<float>& x, float alpha) {
    assert(y.size() == x.size());
    for (size_t i = 0; i < y.size(); ++i) y[i] += alpha * x[i];
}

//------- Activation functions ---------

float relu(float x){
    return x < 0 ? 0 : x;
}

float relu_derivate(float x){
    return x > 0 ? 1.0f : 0.0f;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float sigmoid_derivate(float x){
    float s = sigmoid(x);
    return s * (1.0f - s);
}

vector<float> softmax(const vector<float> &v){
    vector<float> result(v.size());

    float max_val = *max_element(v.begin(), v.end());

    // Compute exp(v_i - max)
    float sum = 0.0f;
    for (size_t i = 0; i < v.size(); i++) {
        result[i] = exp(v[i] - max_val);
        sum += result[i];
    }

    // Normalize
    for (size_t i = 0; i < v.size(); i++) {
        result[i] /= sum;
    }

    return result;
}

// Softmax over each column of an (rows x cols) row-major matrix, in place
void softmaxColumns(vector<float>& M, size_t rows, size_t cols) {
    ScopedTimer t(softMax_time);
    for (size_t s = 0; s < cols; s++) {
        float maxVal = M[s];
        for (size_t r = 1; r < rows; r++) maxVal = max(maxVal, M[r * cols + s]);
        float sum = 0.0f;
        for (size_t r = 0; r < rows; r++) {
            float& e = M[r * cols + s];
            e = exp(e - maxVal);
            sum += e;
        }
        for (size_t r = 0; r < rows; r++) M[r * cols + s] /= sum;
    }
}

// --------- Class functions ----------

//TODO:: Make it possible to have multiple hidden layers

void NeuralNetwork::initialize(vector<int>layers, float bias) {

    //atleast an input, a hidden and an output layer
    assert(layers.size() >= 3);

    this -> inputSize = layers[0];
 
    //First hidden layer

    //Allocate and assign sizes to each layer excluding input layer
    hiddenLayers.resize(layers.size()-1);

        
    //Remianing layers
    for(size_t i = 0; i < hiddenLayers.size(); i++){
            //layers vector got one more element, layers[0] = inputSize
            int prevSize = layers[i]; 
            hiddenLayers[i].layer_size = layers[i+1];
            //Size dependent on previous layer
            hiddenLayers[i].weights.resize(hiddenLayers[i].layer_size * prevSize);
            //Scale for randomization

        #ifdef RELU
            float scale = sqrt(2.0f / (float)prevSize);
        #else 
            float scale = sqrt(1.0f / (float)prevSize);
        #endif
            //randomizing weights
            for(auto &w : hiddenLayers[i].weights) w = randomWeight(scale);
            //bias Allocation & assign
            hiddenLayers[i].bias.assign(hiddenLayers[i].layer_size, bias); 
    }

#ifdef RELU
    activationFunc = relu;
    activationDerivativeFunc = relu_derivate;
#else
    activationFunc = sigmoid;
    activationDerivativeFunc = sigmoid_derivate;
#endif
}

vector<float> NeuralNetwork::forward(const vector<float> &input) { 

    //Adds measurment when "ScopedTimer" object runs out of scope
    ScopedTimer t(forwarding_time);
    
    //First layer
    Layer& firstHidden = hiddenLayers[0];

    firstHidden.pre.resize(firstHidden.layer_size * 1);

    matmul(     firstHidden.weights,
            firstHidden.layer_size,
                input, 
            inputSize, 
          firstHidden.pre);

    axpy(hiddenLayers[0].pre, hiddenLayers[0].bias, 1.0f);
    hiddenLayers[0].activated = hiddenLayers[0].pre;
    for(auto& e : hiddenLayers[0].activated) e = activationFunc(e);


    for(size_t i = 1; i < hiddenLayers.size(); i++){

        Layer& currentLayer = hiddenLayers[i];
        Layer& prevLayer = hiddenLayers[i-1];

        //A vector                                            
        currentLayer.pre.resize(currentLayer.layer_size * 1);
        
        matmul(     currentLayer.weights,
                currentLayer.layer_size,
                    prevLayer.activated,
                prevLayer.layer_size,
              currentLayer.pre);
                                            
        axpy(currentLayer.pre, currentLayer.bias, 1.0f);
        
        //Don't want to activate output layer
        if(i + 1 == hiddenLayers.size()) break;

        currentLayer.activated = currentLayer.pre;
        for(auto& e : currentLayer.activated) e = activationFunc(e);
    }
    
    //Softmax on output layer
    return softmax(hiddenLayers.back().pre);
}

void NeuralNetwork::train(const std::vector<float>& X, const std::vector<uint8_t>& targets,
                                 size_t batchSize, float learingRate){
    
    const int L = hiddenLayers.size();
    assert(L >= 2);
    assert(X.size() == inputSize * batchSize);
    assert(targets.size() == batchSize);

    // ---------- FORWARD ----------
    {
        ScopedTimer t(forwarding_time);

        for (int i = 0; i < L; i++) {
            Layer& layer = hiddenLayers[i];
            const vector<float>& in = (i == 0) ? X : hiddenLayers[i - 1].activated;
            size_t inSize = (i == 0) ? inputSize : hiddenLayers[i - 1].layer_size;
            size_t n = layer.layer_size;

            // (n x inSize) * (inSize x b) = (n x b)
            layer.pre.resize(n * batchSize);
            matmul(layer.weights, n, in, inSize, layer.pre);   // matmul must zero result

            // add bias to every column
            for (size_t r = 0; r < n; r++)
                for (size_t s = 0; s < batchSize; s++)
                    layer.pre[r * batchSize + s] += layer.bias[r];

            // activate (not on the output layer)
            if (i + 1 < L) {
                layer.activated = layer.pre;
                for (auto& e : layer.activated) e = activationFunc(e);
            }
        }
    }

    ScopedTimer t(back_prop_time);

    // ---------- OUTPUT ERROR: softmax per sample (column), minus one-hot ----------

    Layer& out = hiddenLayers.back();
    out.error = out.pre;                         // copy
    softmaxColumns(out.error, out.layer_size, batchSize);
    for (size_t s = 0; s < batchSize; s++) 
        out.error[targets[s] * batchSize + s] -= 1.0f;

    // ---------- BACKPROPAGATE ERROR (all of it, BEFORE touching weights) ----------
    vector<float> Wt; //Allocate transposed weight matrix

    for (int i = static_cast<int>(L) - 2; i >= 0; i--) {
        Layer& cur  = hiddenLayers[i];
        Layer& next = hiddenLayers[i + 1];
        int  nCur = cur.layer_size;
        int nNext = next.layer_size;

        // cur.error = next.weightsᵀ * next.error   ((nCur x nNext) * (nNext x b))
        Wt.assign(nCur * nNext, 0.0f);
        transpose(next.weights, nNext, nCur, Wt);          // Wt is nCur x nNext
        cur.error.resize(nCur * batchSize);
        matmul(Wt, nCur, next.error, nNext, cur.error);

        
        // multiply by activation derivative
        for (size_t idx = 0; idx < nCur * batchSize; idx++)
            cur.error[idx] *= activationDerivativeFunc(cur.pre[idx]);
    }

    // ---------- UPDATE WEIGHTS: W -= lr/b * E * Aᵀ ----------

    const float scale = learingRate / static_cast<float>(batchSize);

    vector<float> At, grad, biasGrad;
    vector<float> ones(batchSize, 1.0f);

    for (int i = 0; i < L; i++) {
        Layer& layer = hiddenLayers[i];
        const vector<float>& A = (i == 0) ? X : hiddenLayers[i - 1].activated;
        int prev = (i == 0) ? inputSize : hiddenLayers[i - 1].layer_size;
        int n = layer.layer_size;

        // grad = E * Aᵀ        (n x batch) * (batch x prev) = (n x prev)
        transpose(A, prev, batchSize, At);              // A is (prev x batch), At is (batch x prev)
        grad.resize(n * prev);
        matmul(layer.error, n, At, batchSize, grad);
        axpy(layer.weights, grad, -scale);

        // bias grad = E * ones  (n x batch) * (batch x 1) = (n x 1)
        biasGrad.resize(n);
        matmul(layer.error, n, ones, batchSize, biasGrad);
        axpy(layer.bias, biasGrad, -scale);
    }
}
