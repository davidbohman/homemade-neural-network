#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "../include/gpuVersion.h"

#define TILE 32

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                   \
                    cudaGetErrorString(err_), __FILE__, __LINE__);         \
            exit(1);                                                       \
        }                                                                  \
    } while (0)


__global__ void cudaMatMul(float* A, int aRows, int aCols, float*B, int bRows, int bCols, float*result){
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    int numTiles = (aCols + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < aRows && aCol < aCols) ? A[row * aCols + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < bRows && col < bCols) ? B[bRow * bCols + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < aRows && col < bCols) {
        result[row * bCols + col] = sum;
    }
}

// A[r][s] = act(A[r][s] + bias[r]).  mode 0 = none, 1 = ReLU
__global__ void addBiasAct(float* A, const float* bias, int rows, int B, int mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * B) return;
    float v = A[idx] + bias[idx / B];
    if (mode == 1) v = v > 0.f ? v : 0.f;
    A[idx] = v;
}

// Softmax down each column (one thread per sample). Reads are coalesced across s.
__global__ void softmaxCols(float* A, int rows, int B) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= B) return;
    float m = A[s];
    for (int r = 1; r < rows; ++r) m = fmaxf(m, A[r * B + s]);
    float sum = 0.f;
    for (int r = 0; r < rows; ++r) {
        float e = expf(A[r * B + s] - m);
        A[r * B + s] = e;
        sum += e;
    }
    float inv = 1.f / sum;
    for (int r = 0; r < rows; ++r) A[r * B + s] *= inv;
}

// Softmax + cross-entropy gradient: delta = output - onehot(label)
__global__ void outputDelta(float* D, const float* A, const unsigned char* labels, int rows, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * B) return;
    int r = idx / B, s = idx % B;
    D[idx] = A[idx] - (labels[s] == r ? 1.f : 0.f);
}

// delta *= ReLU'(z), using the post-ReLU activation (a > 0  <=>  z > 0)
__global__ void reluMask(float* D, const float* A, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) D[i] = A[i] > 0.f ? D[i] : 0.f;
}

// out (C x R) = transpose of in (R x C). Launch: block(32,8), grid(ceil(C/32), ceil(R/32))
__global__ void transposeKernel(const float* in, float* out, int R, int C) {
    __shared__ float t[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8)
        if (x < C && y + j < R) t[threadIdx.y + j][threadIdx.x] = in[(size_t)(y + j) * C + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8)
        if (x < R && y + j < C) out[(size_t)(y + j) * R + x] = t[threadIdx.x][threadIdx.y + j];
}

// W -= scale * dW
__global__ void sgdUpdate(float* W, const float* dW, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) W[i] -= scale * dW[i];
}

// b[r] -= scale * sum_s D[r][s].  One block per row, 256 threads.
__global__ void biasUpdate(float* bias, const float* D, int B, float scale) {
    __shared__ float red[256];
    int r = blockIdx.x;
    float s = 0.f;
    for (int i = threadIdx.x; i < B; i += blockDim.x) s += D[r * B + i];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) bias[r] -= scale * red[0];
}

// Counts samples whose argmax matches the label
__global__ void countHits(const float* A, const unsigned char* labels, int rows, int B, int* hits) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= B) return;
    int best = 0;
    float bv = A[s];
    for (int r = 1; r < rows; ++r) {
        float v = A[r * B + s];
        if (v > bv) { bv = v; best = r; }
    }
    if (best == labels[s]) atomicAdd(hits, 1);
}


namespace {

struct GpuNet {
    bool ready = false;
    int L = 0;                         // number of weight layers
    int maxB = 0;
    std::vector<int> sizes;            // sizes[0] = input, sizes[L] = output
    // Index l = 1..L (index 0 unused except A[0], the input)
    std::vector<float*> W, bias, dW, A, D;
    float* WT = nullptr;               // scratch: transposed weights
    float* AT = nullptr;               // scratch: transposed activations
    unsigned char* dLabels = nullptr;
    int* dHits = nullptr;
} g;

void devMatMul(float* A, int M, int K, float* B, int N, float* C) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    cudaMatMul<<<grid, block>>>(A, M, K, B, K, N, C);   // C(MxN) = A(MxK) * B(KxN)
    CUDA_CHECK(cudaGetLastError());
}

void devTranspose(const float* in, float* out, int R, int C) {
    dim3 block(32, 8);
    dim3 grid((C + 31) / 32, (R + 31) / 32);
    transposeKernel<<<grid, block>>>(in, out, R, C);
    CUDA_CHECK(cudaGetLastError());
}

inline int blocks(int n, int t = 256) { return (n + t - 1) / t; }

void forwardDevice(int B) {
    for (int l = 1; l <= g.L; ++l) {
        devMatMul(g.W[l], g.sizes[l], g.sizes[l - 1], g.A[l - 1], B, g.A[l]);
        int n = g.sizes[l] * B;
        addBiasAct<<<blocks(n), 256>>>(g.A[l], g.bias[l], g.sizes[l], B, l < g.L ? 1 : 0);
        CUDA_CHECK(cudaGetLastError());
    }
    softmaxCols<<<blocks(B, 128), 128>>>(g.A[g.L], g.sizes[g.L], B);
    CUDA_CHECK(cudaGetLastError());
}

void checkReady(int B) {
    if (!g.ready) { fprintf(stderr, "gpuNet: not initialised\n"); exit(1); }
    if (B > g.maxB) { fprintf(stderr, "gpuNet: batch %d > maxBatch %d\n", B, g.maxB); exit(1); }
}

} // namespace


// sizes: {784, 512, 256, 128, 10}.  hW[l-1] = out x in row-major (layer l = 1..L).
// hB[l-1] = out biases, or pass hB == nullptr for zero biases.
void gpuNetInit(const int* sizes, int numSizes, int maxBatch,
                const float* const* hW, const float* const* hB) {
    if (g.ready) gpuNetFree();
    g.L = numSizes - 1;
    g.maxB = maxBatch;
    g.sizes.assign(sizes, sizes + numSizes);
    g.W.assign(g.L + 1, nullptr);   g.bias.assign(g.L + 1, nullptr);
    g.dW.assign(g.L + 1, nullptr);  g.A.assign(g.L + 1, nullptr);
    g.D.assign(g.L + 1, nullptr);

    size_t maxW = 0, maxAct = 0;
    for (int l = 0; l <= g.L; ++l) {
        CUDA_CHECK(cudaMalloc(&g.A[l], (size_t)g.sizes[l] * maxBatch * sizeof(float)));
        if (l < g.L) maxAct = std::max(maxAct, (size_t)g.sizes[l] * maxBatch);
    }
    for (int l = 1; l <= g.L; ++l) {
        size_t wN = (size_t)g.sizes[l] * g.sizes[l - 1];
        maxW = std::max(maxW, wN);
        CUDA_CHECK(cudaMalloc(&g.W[l],    wN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g.dW[l],   wN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g.bias[l], g.sizes[l] * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g.D[l],    (size_t)g.sizes[l] * maxBatch * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(g.W[l], hW[l - 1], wN * sizeof(float), cudaMemcpyHostToDevice));
        if (hB) CUDA_CHECK(cudaMemcpy(g.bias[l], hB[l - 1], g.sizes[l] * sizeof(float), cudaMemcpyHostToDevice));
        else    CUDA_CHECK(cudaMemset(g.bias[l], 0, g.sizes[l] * sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&g.WT, maxW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.AT, maxAct * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.dLabels, maxBatch));
    CUDA_CHECK(cudaMalloc(&g.dHits, sizeof(int)));
    g.ready = true;
}

// One SGD step on a batch. hX is features x B (sample s = column s), exactly
// the layout your X vector already has. lr is divided by B internally
// (gradient averaged over the batch).
void gpuNetTrain(const float* hX, const unsigned char* hLabels, int B, float lr) {
    checkReady(B);
    CUDA_CHECK(cudaMemcpy(g.A[0], hX, (size_t)g.sizes[0] * B * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.dLabels, hLabels, B, cudaMemcpyHostToDevice));

    forwardDevice(B);

    const int L = g.L;
    const float scale = lr / (float)B;

    int n = g.sizes[L] * B;
    outputDelta<<<blocks(n), 256>>>(g.D[L], g.A[L], g.dLabels, g.sizes[L], B);
    CUDA_CHECK(cudaGetLastError());

    for (int l = L; l >= 1; --l) {
        int out = g.sizes[l], in = g.sizes[l - 1];

        // dW = D[l] (out x B) * A[l-1]^T (B x in)
        devTranspose(g.A[l - 1], g.AT, in, B);              // AT: B x in
        devMatMul(g.D[l], out, B, g.AT, in, g.dW[l]);       // dW: out x in

        // delta for the previous layer uses the OLD weights, so do it before the update
        if (l > 1) {
            devTranspose(g.W[l], g.WT, out, in);             // WT: in x out
            devMatMul(g.WT, in, out, g.D[l], B, g.D[l - 1]); // in x B
            int m = in * B;
            reluMask<<<blocks(m), 256>>>(g.D[l - 1], g.A[l - 1], m);
            CUDA_CHECK(cudaGetLastError());
        }

        int wN = out * in;
        sgdUpdate<<<blocks(wN), 256>>>(g.W[l], g.dW[l], wN, scale);
        biasUpdate<<<out, 256>>>(g.bias[l], g.D[l], B, scale);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Forward pass for B samples (features x B). hOut receives sizes[L] x B floats.
void gpuNetForward(const float* hX, int B, float* hOut) {
    checkReady(B);
    CUDA_CHECK(cudaMemcpy(g.A[0], hX, (size_t)g.sizes[0] * B * sizeof(float), cudaMemcpyHostToDevice));
    forwardDevice(B);
    CUDA_CHECK(cudaMemcpy(hOut, g.A[g.L], (size_t)g.sizes[g.L] * B * sizeof(float), cudaMemcpyDeviceToHost));
}

// Returns how many of the B samples are classified correctly (for fast testing).
int gpuNetCountHits(const float* hX, const unsigned char* hLabels, int B) {
    checkReady(B);
    CUDA_CHECK(cudaMemcpy(g.A[0], hX, (size_t)g.sizes[0] * B * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.dLabels, hLabels, B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(g.dHits, 0, sizeof(int)));
    forwardDevice(B);
    countHits<<<blocks(B, 128), 128>>>(g.A[g.L], g.dLabels, g.sizes[g.L], B, g.dHits);
    CUDA_CHECK(cudaGetLastError());
    int hits = 0;
    CUDA_CHECK(cudaMemcpy(&hits, g.dHits, sizeof(int), cudaMemcpyDeviceToHost));
    return hits;
}

// Copy weights/biases back so the CPU forward() (single-sample, drawing UI) sees them.
void gpuNetDownloadWeights(float* const* hW, float* const* hB) {
    for (int l = 1; l <= g.L; ++l) {
        CUDA_CHECK(cudaMemcpy(hW[l - 1], g.W[l], (size_t)g.sizes[l] * g.sizes[l - 1] * sizeof(float), cudaMemcpyDeviceToHost));
        if (hB) CUDA_CHECK(cudaMemcpy(hB[l - 1], g.bias[l], g.sizes[l] * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

void gpuNetFree() {
    if (!g.ready) return;
    for (int l = 0; l <= g.L; ++l) {
        cudaFree(g.A[l]);
        if (l >= 1) { cudaFree(g.W[l]); cudaFree(g.dW[l]); cudaFree(g.bias[l]); cudaFree(g.D[l]); }
    }
    cudaFree(g.WT); cudaFree(g.AT); cudaFree(g.dLabels); cudaFree(g.dHits);
    g = GpuNet();
}

