#include <algorithm>
#include <cstddef>
#include <zlib.h>
#include <vector>
#include <iostream>
#include "../include/NeuralNetwork_CPU.h"
#include <thread>
#include <string>
#include <raylib.h>

// ---- READING MNIST DATASET -----

struct MNISTImage {
    std::vector<float> pixels;  // 28*28 = 784
    uint8_t label;
};

//Changes from big endian to little endian (MSB -> LSB first)
uint32_t read_big_endian_uint32(gzFile &file) {
    unsigned char bytes[4];
    gzread(file, bytes, 4);
    return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8) | uint32_t(bytes[3]);
}

std::vector<MNISTImage> load_MNIST(const std::string &image_path, const std::string &label_path) {
    gzFile imgFile = gzopen(image_path.c_str(), "rb");
    gzFile lblFile = gzopen(label_path.c_str(), "rb");
    if (!imgFile || !lblFile) throw std::runtime_error("Cannot open MNIST files.");

    // Read headers
    uint32_t magicImages = read_big_endian_uint32(imgFile);
    uint32_t numImages = read_big_endian_uint32(imgFile);
    uint32_t rows = read_big_endian_uint32(imgFile);
    uint32_t cols = read_big_endian_uint32(imgFile);

    uint32_t magicLabels = read_big_endian_uint32(lblFile);
    uint32_t numLabels = read_big_endian_uint32(lblFile);

    if (magicImages != 2051) throw std::runtime_error("Bad magic number in image file.");
    if (magicLabels != 2049) throw std::runtime_error("Bad magic number in label file.");
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

// ---- DRAWING FUNCTIONS ----

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

// ---- NN FUNCTIONS ----

bool is_hit(std::vector<float> &output, uint8_t target){

    float highest = 0;
    uint8_t highest_index = 0;
    for(size_t i = 0; i < output.size(); i++){
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
    auto trainData = load_MNIST("MNIST/MNIST/raw/train-images-idx3-ubyte.gz", "MNIST/MNIST/raw/train-labels-idx1-ubyte.gz");
    std::cout << "Loaded " << trainData.size() << " images." << std::endl;
    auto testData = load_MNIST("MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz", "MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz");
    std::cout << "Training..." << std::endl;
    // Time messurment
    auto start_training = std::chrono::high_resolution_clock::now();
    forwarding_time = 0.0;
    back_prop_time = 0.0;
    matmul_time = 0.0;

    for (int i = 0; i < EPOCHS; i++) {
        auto start_epoch =
        std::chrono::high_resolution_clock::now(); // Time messurment
        int iteration = 0;
        for (auto &data : trainData) {
            nn.train(data.pixels, data.label, LEARNING_RATE);
            if (iteration % 500 == 0) training_progress++;
            iteration++;
        }

        auto end_epoch = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_epoch - start_epoch;

        std::cout << "Epoch complete. Elapsed time: " << elapsed.count() << " seconds" << std::endl;
        std::cout << "Time spent on forwarding: " << forwarding_time << "seconds (" << (forwarding_time / elapsed.count()) * 100 << "%)" << std::endl;
        std::cout << "Time spent on back propagation: " << back_prop_time << "seconds (" << (back_prop_time / elapsed.count()) * 100 << "%)" << std::endl;
        std::cout << "Time spent on matrix multiplication: " << matmul_time << "seconds (" << (matmul_time / elapsed.count()) * 100 << "%)" << std::endl;
    }

    auto end_training = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_training - start_training;

    std::cout << "Training complete after: " << elapsed.count()
              << "seconds\nstarting testing..." << std::endl;

    // Testing the NN - Generating hit rate
    std::vector<float> output;
    int hits = 0;
    int tries = 0;
    for (auto &data : testData) {
        output = nn.forward(data.pixels);
        if (is_hit(output, data.label)) hits++;
        tries++;
        hit_rate = (float)hits / (float)tries;
    }
    training_complete = true;
    std::cout << "Testing finished, hit rate: " << hit_rate << std::endl;

    return;
}

std::atomic<bool> exit_application{false};
//One thread will always be running this function
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

// ---- BUTTONS ---- 

struct Button{
    int x_pos = 4;
    int y_pos; 
    int width;
    int height = 35; //Will be same for all buttons
    std::string text;
    Color color = GRAY;
}; 

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

    // NN setup
    NeuralNetwork_CPU nn;
    nn.initialize(784,128, 10, 0.2f);

    // Helper threads
    std::thread training_thread(train_nn, std::ref(nn), 0.01, 1);
    std::thread guess_thread(nn_guess, std::ref(nn));

    // Drawing
    int window_width = 408;
    int window_height = 610;


    int text_start_coor = 10;

    //Info Panel
    int info_width = 400;
    int info_height = 120;

    //Drawing
    int drawing_area_side = 400; // is a square
    int user_pixel_size = 13;

    Color text_color = DARKGRAY;

    // Pixel grid
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
        user_img[i * 28 + j].x_coor = j;
        user_img[i * 28 + j].y_coor = i;
        user_img[i * 28 + j].x_pos = 8 + i * (user_pixel_size + 1);
        user_img[i * 28 + j].y_pos = info_height+13 + j * (user_pixel_size + 1);
        user_img[i * 28 + j].width = user_pixel_size;
        user_img[i * 28 + j].height = user_pixel_size;
        user_img[i * 28 + j].val = 0.0f;
        user_img[i * 28 + j].color = DARKBROWN;
        }
    }  

    int tmp_btn_width = 398;
    
    // Buttons will always be at the bottom
    Button reset_btn;
    reset_btn.y_pos = drawing_area_side + info_height + 12;
    reset_btn.width = tmp_btn_width;
    reset_btn.text = "RESET";
    Button epoch_btn;
    epoch_btn.y_pos = epoch_btn.height + reset_btn.y_pos + 4;
    epoch_btn.width = tmp_btn_width;
    epoch_btn.text = "NEW EPOCH";

    std::vector<Button> btns({reset_btn, epoch_btn});

    SetTraceLogLevel(LOG_NONE);
    InitWindow(window_width, window_height, "NEURAL NETWORK TESTER");
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(DARKGRAY);

        // DRAWING AREA
        DrawRectangle(4, info_height+8, drawing_area_side, drawing_area_side, BLACK);

        for (Drawing_pixel &pixel : user_img) {
            DrawRectangle(pixel.x_pos, pixel.y_pos, pixel.width, pixel.height,pixel.color);
            if ((IsMouseButtonPressed(0) || IsMouseButtonDown(0)) && within_space(GetMouseX(), GetMouseY(), pixel)) {
                paint_pixel(pixel, 2);
                data_changed = true;
            }
        }

        // INFO PANEL
        DrawRectangle(4, 4, info_width, info_height, LIGHTGRAY);

        // Training status
        DrawText("TRAINING", text_start_coor, 10, 30, text_color);
        for (int i = 0; i < training_progress; i++) {
            DrawRectangle(text_start_coor + i * 3, 40, 30, 20, text_color);
        }
        if (training_complete) {
            DrawText("-", window_width/2 - 15, 10, 30, text_color);
            DrawText("COMPLETE", text_start_coor + 20 * 10 + 18, 10, 30, text_color);
        }

        // NN Status
        DrawText((std::string("HIT RATE : ") + (std::to_string(hit_rate * 100) + "%")).data(),text_start_coor, 65, 20, text_color);

        if (training_complete) {
            DrawText((std::string("MACHINE GUESS : ") + (std::to_string(current_guess))).data(), text_start_coor, 90, 20, text_color);
        }

        // BUTTONS
        for (Button btn : btns) {
            DrawRectangle(btn.x_pos, btn.y_pos, btn.width, btn.height, btn.color);
            DrawText(btn.text.data(), btn.x_pos + (btn.width / 2) - 6 * btn.text.length(), btn.y_pos + (btn.height / 2) - 7, 20, BLACK);
        }
        if (IsMouseButtonPressed(0) && within_space(GetMouseX(), GetMouseY(), reset_btn)) {
            for (Drawing_pixel &pixel : user_img) {
            pixel.val = 0.0f;
            pixel.color = DARKBROWN;
        }
        }
        if (IsMouseButtonPressed(0) && within_space(GetMouseX(), GetMouseY(), epoch_btn)) {
            if (training_complete) {
                training_complete = false;
                if (training_thread.joinable()) training_thread.join();
                training_progress = 0;
                training_thread = std::thread(train_nn, std::ref(nn), 0.01, 1);
            }
        }

        EndDrawing();
        exit_application = WindowShouldClose(); //ATOMIC FLAG
    }
    CloseWindow();

    if (training_thread.joinable())
        training_thread.join();
    if (guess_thread.joinable())
        guess_thread.join();

    return 0;
}


