#include <zlib.h>
#include <vector>
#include <iostream>
#include "../include/NeuralNetwork.h"
#include "../include/NeuralNetwork_CPU.h"





struct MNISTImage {
    std::vector<float> pixels;  // 28*28 = 784
    uint8_t label;
};

//Changes from big endian to little endian (MSB -> LSB first)
uint32_t readBigEndianUint32(gzFile &file) {
    unsigned char bytes[4];
    gzread(file, bytes, 4);
    return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8) | uint32_t(bytes[3]);
}

std::vector<MNISTImage> loadMNIST(const std::string &image_path, const std::string &label_path) {
    gzFile imgFile = gzopen(image_path.c_str(), "rb");
    gzFile lblFile = gzopen(label_path.c_str(), "rb");
    if (!imgFile || !lblFile) throw std::runtime_error("Cannot open MNIST files.");

    // Read headers
    uint32_t magicImages = readBigEndianUint32(imgFile);
    uint32_t numImages = readBigEndianUint32(imgFile);
    uint32_t rows = readBigEndianUint32(imgFile);
    uint32_t cols = readBigEndianUint32(imgFile);

    uint32_t magicLabels = readBigEndianUint32(lblFile);
    uint32_t numLabels = readBigEndianUint32(lblFile);

    if (numImages != numLabels) throw std::runtime_error("Number of images and labels mismatch.");

    std::vector<MNISTImage> dataset(numImages);

    for (uint32_t i = 0; i < numImages; ++i) {
        MNISTImage &img = dataset[i];

        // Read raw bytes
        std::vector<unsigned char> buffer(rows * cols);
        gzread(imgFile, buffer.data(), rows*cols);
        
        // Convert to floats
        img.pixels.resize(rows * cols);
        for(size_t j = 0; j < buffer.size(); ++j){
            img.pixels[j] = buffer[j] / 255.0f;
        }

        unsigned char label;
        gzread(lblFile, &label, 1);
        img.label = label;
    }

    gzclose(imgFile);
    gzclose(lblFile);

    return dataset;
}

bool isHit(std::vector<float> &output, uint8_t target){

    float highest = 0;
    int highest_index = 0;
    for(int i = 0; i < output.size(); i++){
        if(highest < output[i]){
            highest = output[i];
            highest_index = i;
        }
    }

    return highest_index == target;

}

int main() {
    try {
        auto trainData = loadMNIST("../MNIST/MNIST/raw/train-images-idx3-ubyte.gz", "../MNIST/MNIST/raw/train-labels-idx1-ubyte.gz");
        std::cout << "Loaded " << trainData.size() << " images." << std::endl;

        auto testData = loadMNIST("../MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz", "../MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz");

        std::unique_ptr<NeuralNetwork> nn;

        /* ANVÄND SEN NÄR GPU HAR IMPLEMENTERATS OCKSÅ
        bool useGPU = true; // eller sätt via argument / config
        if (useGPU) {
            nn = std::make_unique<NeuralNetwork_GPU>();
        } else {
            nn = std::make_unique<NeuralNetwork_CPU>();
        }
        */
        nn = std::make_unique<NeuralNetwork_CPU>();  //Ta bort denna rad när GPU implementerats

        // Training variables
        float LEARNING_RATE = 0.01;
        int EPOCHS = 1;


        nn->initialize(784, 128, 10, 0.1f);

        std::cout << "Training..." << std::endl;
        auto start_training = std::chrono::high_resolution_clock::now(); // Time messurment

        for(int i = 0; i < EPOCHS; i++){
            auto start_epoch = std::chrono::high_resolution_clock::now(); // Time messurment

            for (auto &data : trainData) {
                nn->train(data.pixels, data.label, LEARNING_RATE);
            }

            auto end_epoch = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_epoch - start_epoch;
            std::cout << "Epoch " << i << " complete. Elapsed time: " << elapsed.count() << " seconds" << std::endl;;
        }
        
        auto end_training = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_training - start_training;

        std::cout <<"Training complete after: " << elapsed.count() << "seconds\nstarting testing..." << std::endl;

        //Testing
        std::vector<float> output;
        int hits = 0;
        int tries = 0;
        for(auto &data : testData){
            output = nn->forward(data.pixels);
            if(isHit(output, data.label)) hits++;
            tries++;
        }

        float hit_rate = (float) hits / (float) tries;
        std::cout <<"Testing finished, hit rate: " << hit_rate << std::endl; 

        
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
}


