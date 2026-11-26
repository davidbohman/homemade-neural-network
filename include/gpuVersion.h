#pragma once



// device-resident network
void gpuNetInit(const int* sizes, int numSizes, int maxBatch,
                const float* const* hW, const float* const* hB);

void gpuNetTrain(const float* hX, const unsigned char* hLabels, int B, float lr);

void gpuNetForward(const float* hX, int B, float* hOut);

int  gpuNetCountHits(const float* hX, const unsigned char* hLabels, int B);

void gpuNetDownloadWeights(float* const* hW, float* const* hB);

void gpuNetFree();




