#include <algorithm>
#include <cstddef>
#include <zlib.h>
#include <vector>
#include <iostream>
#include "../include/NeuralNetwork.h"
#include "../include/NeuralNetwork_CPU.h"
#include <thread>
#include <string>
#include <raylib.h>

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

int training_progress = 0;
bool training_complete = false;
float hit_rate = 0.0f;

void train_nn(NeuralNetwork_CPU & nn, float LEARNING_RATE, int EPOCHS){
    try{
        auto trainData = loadMNIST("../MNIST/MNIST/raw/train-images-idx3-ubyte.gz", "../MNIST/MNIST/raw/train-labels-idx1-ubyte.gz");
        std::cout << "Loaded " << trainData.size() << " images." << std::endl;

        auto testData = loadMNIST("../MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz", "../MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz");


        std::cout << "Training..." << std::endl;
        // Time messurment
        auto start_training = std::chrono::high_resolution_clock::now(); 
        forwarding_time = 0.0;
        back_prop_time = 0.0;
        matmul_time = 0.0;

        for(int i = 0; i < EPOCHS; i++){
            auto start_epoch = std::chrono::high_resolution_clock::now(); // Time messurment
            int iteration = 0;
            for (auto &data : trainData) {
                nn.train(data.pixels, data.label, LEARNING_RATE);
                if(iteration % 500 == 0) training_progress++;
                iteration++;
            }
            
            auto end_epoch = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_epoch - start_epoch;

            std::cout << "Epoch " << i << " complete. Elapsed time: " << elapsed.count() << " seconds" << std::endl;
            std::cout << "Time spent on forwarding: " << forwarding_time << "seconds (" << (forwarding_time/elapsed.count())*100 << "%)" << std::endl;
            std::cout << "Time spent on back propagation: " << back_prop_time << "seconds (" << (back_prop_time/elapsed.count())*100 << "%)"<< std::endl;
            std::cout << "Time spent on matmul: " << matmul_time << "seconds (" << (matmul_time/elapsed.count())*100 << "%)"<< std::endl;
        }
        
        auto end_training = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_training - start_training;

        std::cout <<"Training complete after: " << elapsed.count() << "seconds\nstarting testing..." << std::endl;

        //Testing
        std::vector<float> output;
        int hits = 0;
        int tries = 0;
        for(auto &data : testData){
            output = nn.forward(data.pixels);
            if(isHit(output, data.label)) hits++;
            tries++;
            hit_rate = (float) hits / (float) tries;
        }
        training_complete = true;
        std::cout <<"Testing finished, hit rate: " << hit_rate << std::endl; 

    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
    return;
}

struct Drawing_pixel{
    int x_coor;
    int y_coor;
    int x_pos;
    int y_pos;
    int width;
    int height;
    float val = 0.0;
    Color color;
};
//Vector that holdes drawing made by user
std::vector<Drawing_pixel> user_img(28*28);
std::vector<float> user_img_data(28*28);
bool data_changed = false;
int current_guess = 0;


std::atomic<bool> exit_application{false};
void nn_guess(NeuralNetwork_CPU & nn){
    while(!exit_application){
        if(training_complete && data_changed){
            for(const Drawing_pixel pixel : user_img){
                user_img_data[pixel.y_coor + pixel.x_coor*28] = pixel.val;
            } 
            std::vector<float> output = nn.forward(user_img_data);
            float highest = 0;
            int highest_index = 0;
            for(int i = 0; i < (int) output.size(); i++){
                if(highest < output[i]){
                    highest = output[i];
                    highest_index = i;
                }
            }
            current_guess = highest_index;
            data_changed = false;
        }
    }
}


struct Dir_elem{
    int x;
    int y;
    float val;
};

//Creates directional matrix, used for pixel painting
std::vector<Dir_elem> make_dir_offset(int radius) {
    float fade = 1.0f / (float) radius;
    radius--;
    std::vector<Dir_elem> dirs;
    for (int dr = -radius; dr <= radius; dr++) {
        for (int dc = -radius; dc <= radius; dc++) {
            if (dr == 0 && dc == 0) continue;   // skip the center cell
            //if (dr*dr + dc*dc > radius*radius) continue; //THIS IS IF CIRCLE IS WANTED INSTEAD
            if (abs(dr) + abs(dc) > radius) continue; // THIS IS IF DIAMOND IS WANTED INSTEAD
            float val =1.0f - (std::max(dr, dc) * fade);
            dirs.push_back({dc, dr, val});
        }
    }
    return dirs;
}
std::vector<Dir_elem> dir_offset = make_dir_offset(2);  

void paint_pixel(Drawing_pixel & pixel, int radius){


    pixel.val = 1.0f;
    pixel.color = WHITE;
    if(radius <= 1) return;

    for(Dir_elem dir : dir_offset){
        int nx = pixel.x_coor + dir.x;
        int ny = pixel.y_coor + dir.y;
        if (nx < 0 || nx >= 28 || ny < 0 || ny >= 28) continue;

        Drawing_pixel& p = user_img[ny * 28 + nx];
        p.val = std::min(1.0f ,p.val + dir.val);         
        p.color = WHITE;
        p.color.a = (unsigned char)(p.val * 255.0f);
    }
}

struct Button{
    int x_pos;
    int y_pos = 410; //Will be same for all buttons
    int width;
    int height = 35; //Will be same for all buttons
    std::string text;
    Color color = GRAY;
}; 

//TODO: Make the draw function (Think about the brush option)

bool within_space(int mouse_x, int mouse_y, Drawing_pixel pixel){
    bool horisontal = (pixel.x_pos < mouse_x) && (mouse_x < (pixel.x_pos+pixel.width));
    bool vertical = (pixel.y_pos < mouse_y) && (mouse_y < (pixel.y_pos+pixel.height));
    return horisontal && vertical;
}

bool within_space(int mouse_x, int mouse_y, Button btn){
    bool horisontal = (btn.x_pos < mouse_x) && (mouse_x < (btn.x_pos+btn.width));
    bool vertical = (btn.y_pos < mouse_y) && (mouse_y < (btn.y_pos+btn.height));
    return horisontal && vertical;
}

int main() {
    try {

        //NN setup
        NeuralNetwork_CPU nn; 
        nn.initialize(784, 128, 10, 0.1f);

        //Helper threads
        std::thread training_thread(train_nn, std::ref(nn), 0.01, 1);
        std::thread guess_thread(nn_guess, std::ref(nn));


        //Drawing
        int window_width = 800;
        int window_height = 450;
        int text_start_coor = 10+ window_width/2;
        int user_pixel_size = 13;
        
        Color text_color = DARKGRAY;
        
        //Pixel grid
        for(int i = 0; i < 28; i++)
            {
                for(int j = 0; j < 28; j++){
                    user_img[i*28 + j].x_coor = j;
                    user_img[i*28 + j].y_coor = i;
                    user_img[i*28 + j].x_pos = 9+i*(user_pixel_size+1);
                    user_img[i*28 + j].y_pos = 10+j*(user_pixel_size+1);
                    user_img[i*28 + j].width = user_pixel_size;
                    user_img[i*28 + j].height = user_pixel_size;
                    user_img[i*28 + j].val = 0.0f;
                    user_img[i*28 + j].color = DARKBROWN;
            }
        }

        //Buttons will always be at the bottom
        Button reset_btn;
        reset_btn.x_pos = 4;
        reset_btn.width = 398;
        reset_btn.text = "RESET";
        Button correct_btn;
        correct_btn.x_pos = 405;
        correct_btn.width = 390/2 - 2;
        correct_btn.text = "CORRECT";
        Button epoch_btn;
        epoch_btn.x_pos = correct_btn.x_pos + correct_btn.width + 3;
        epoch_btn.width = correct_btn.width;
        epoch_btn.text = "NEW EPOCH";

        std::vector<Button> btns({reset_btn, correct_btn, epoch_btn});
        
        InitWindow(window_width, window_height, "raylib test");
        SetTargetFPS(60);


        while (!WindowShouldClose()) {
            BeginDrawing();
            ClearBackground(DARKGRAY);

            //DRAWING AREA
            DrawRectangle(4, 5, 400, 400, BLACK);

            for(Drawing_pixel &pixel : user_img){
                DrawRectangle(pixel.x_pos, pixel.y_pos, pixel.width, pixel.height, pixel.color);
                if((IsMouseButtonPressed(0) || IsMouseButtonDown(0)) && within_space(GetMouseX(), GetMouseY(), pixel))
                {
                    paint_pixel(pixel, 2);
                    data_changed = true;
                }
            }
            



            //INFO PANEL
            DrawRectangle(405, 5, 390, 400, LIGHTGRAY);

            //Training status
            DrawText("TRAINING", text_start_coor, 10, 30, text_color);
            for(int i = 0; i < training_progress; i++){
                DrawRectangle(text_start_coor + i*3, 40, 20, 20, text_color);
            }
            if(training_complete){
                DrawText("- COMPLETE", text_start_coor + 18*9 + 9, 10, 30, text_color);
            }

            //NN Status
            DrawText((std::string("HIT RATE : ")+ (std::to_string(hit_rate*100)+"%")).data(), text_start_coor, 70, 20, text_color);
            
            if(training_complete){
                DrawText((std::string("MACHINE GUESS : ")+ (std::to_string(current_guess))).data(), text_start_coor, 100,20 , text_color); 
            }

            //BUTTONS
            for(Button btn : btns){
                DrawRectangle(btn.x_pos, btn.y_pos, btn.width, btn.height, btn.color);
                DrawText(btn.text.data(), btn.x_pos + (btn.width/2)-6*btn.text.length(), btn.y_pos + (btn.height/2) -7, 20, BLACK);
            } 
            if(IsMouseButtonPressed(0) && within_space(GetMouseX(), GetMouseY(), reset_btn))
            {
                for(Drawing_pixel & pixel: user_img){
                    pixel.val = 0.0f;
                    pixel.color = DARKBROWN;
                }
            }
            if(IsMouseButtonPressed(0) && within_space(GetMouseX(), GetMouseY(), correct_btn))
            {   
                //Add logic
                std::cout << "Correct answer!!" << std::endl;
            }
            if(IsMouseButtonPressed(0) && within_space(GetMouseX(), GetMouseY(), epoch_btn))
            {
                if(training_complete){
                    training_complete = false;
                    if(training_thread.joinable()) training_thread.join();
                    training_progress = 0;
                    training_thread = std::thread(train_nn, std::ref(nn), 0.01, 1);
                }
            }

            EndDrawing();
            exit_application = WindowShouldClose();
        }
        CloseWindow();

        if(training_thread.joinable()) training_thread.join();
        if(guess_thread.joinable()) guess_thread.join();
        
        return 0;
        
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}


