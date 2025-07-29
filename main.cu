#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <locale>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <numeric> 
#include <cublas_v2.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        float sum = 0.f;
        for (int i = 0; i < N; ++i)
            sum += A[row * N + i] * B[i * K + col];
        C[row * K + col] = sum;
    }
}

void matmul_cuda(const std::vector<std::vector<float>> &A,
                 const std::vector<std::vector<float>> &B,
                 std::vector<std::vector<float>> &C)
{
    int M = A.size();
    int N = A[0].size();
    int K = B[0].size();
    std::vector<float> A_flat(M * N), B_flat(N * K), C_flat(M * K);

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            A_flat[i * N + j] = A[i][j];

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < K; ++j)
            B_flat[i * K + j] = B[i][j];

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));
    cudaMemcpy(d_A, A_flat.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_flat.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(C_flat.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    C.resize(M, std::vector<float>(K));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            C[i][j] = C_flat[i * K + j];

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}



__global__ void add_bias_kernel(float* y, const float* b, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // muestra
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // dimensión
    if (row < M && col < N) {
        y[row * N + col] += b[col];
    }
}

__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // fila original
    int j = blockIdx.x * blockDim.x + threadIdx.x; // columna original

    if (i < rows && j < cols)
        output[j * rows + i] = input[i * cols + j];
}

__global__ void softmax_kernel(float* mat, int rows, int cols) {
    extern __shared__ float shdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows || tid >= cols) return;

    float* row_data = mat + row * cols;

    // Paso 1: hallar el máximo de la fila
    float maxval = -1e20f;
    for (int i = tid; i < cols; i += blockDim.x)
        maxval = fmaxf(maxval, row_data[i]);
    shdata[tid] = maxval;
    __syncthreads();

    // Reducimos para hallar el verdadero máximo
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            shdata[tid] = fmaxf(shdata[tid], shdata[tid + offset]);
        __syncthreads();
    }
    maxval = shdata[0];
    __syncthreads();

    // Paso 2: calcular suma de exponenciales
    float sum = 0.f;
    for (int i = tid; i < cols; i += blockDim.x)
        sum += expf(row_data[i] - maxval);
    shdata[tid] = sum;
    __syncthreads();

    // Reducción para suma total
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            shdata[tid] += shdata[tid + offset];
        __syncthreads();
    }
    sum = shdata[0];
    __syncthreads();

    // Paso 3: normalizar
    for (int i = tid; i < cols; i += blockDim.x)
        row_data[i] = expf(row_data[i] - maxval) / sum;
}


__global__ void layernorm_kernel(
    float* x, float* y, const float* gamma, const float* beta,
    int T, int D, float eps)
{
    int t = blockIdx.x;      // fila
    int j = threadIdx.x;     // dimensión

    extern __shared__ float stats[];
    float* mu = stats;
    float* var = stats + T;

    if (j >= D || t >= T) return;

    float* row = x + t * D;

    // Paso 1: calcular media (por hilo con reducción implícita)
    float sum = 0.f;
    for (int k = 0; k < D; ++k)
        sum += row[k];
    float mean = sum / D;

    // Paso 2: calcular varianza
    float sqsum = 0.f;
    for (int k = 0; k < D; ++k)
        sqsum += (row[k] - mean) * (row[k] - mean);
    float variance = sqsum / D;

    float std_inv = rsqrtf(variance + eps);
    float xhat = (row[j] - mean) * std_inv;
    y[t * D + j] = gamma[j] * xhat + beta[j];
}

__global__ void relu_kernel(float* mat, float* mask, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        int idx = i * N + j;
        float val = mat[idx];
        if (val > 0) {
            mask[idx] = 1.f;
        } else {
            mat[idx] = 0.f;
            mask[idx] = 0.f;
        }
    }
}


__global__ void dropout_kernel(float* mat, float* mask, int M, int N, float rate, unsigned long seed) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        int idx = i * N + j;

        curandState state;
        curand_init(seed, idx, 0, &state);
        float rand_val = curand_uniform(&state);

        float keep = (rand_val > rate) ? 1.f : 0.f;
        mask[idx] = keep;
        mat[idx] *= keep / (1.f - rate);
    }
}

__global__ void avg_pool_kernel(const float* input, float* output, int T, int D) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;

    float sum = 0.f;
    for (int t = 0; t < T; ++t) {
        sum += input[t * D + j];
    }
    output[j] = sum / T;
}

///////////////////                  CUDA                ////////////////////////////

void matmul_cuda_head(const std::vector<std::vector<float>> &A,
                      const std::vector<std::vector<float>> &B,
                      std::vector<std::vector<float>> &C)
{
    int M = A.size();     // T
    int N = A[0].size();  // d_k
    int K = B[0].size();  // d_k o T

    std::vector<float> A_flat(M * N), B_flat(N * K), C_flat(M * K);

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            A_flat[i * N + j] = A[i][j];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < K; ++j)
            B_flat[i * K + j] = B[i][j];

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));
    cudaMemcpy(d_A, A_flat.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_flat.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16), grid((K + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C_flat.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    C.resize(M, std::vector<float>(K));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            C[i][j] = C_flat[i * K + j];

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void average_pool_cuda(const std::vector<std::vector<float>> &input,
                       std::vector<float> &output)
{
    int T = input.size();     // número de pasos
    int D = input[0].size();  // dimensión del modelo

    std::vector<float> input_flat(T * D);
    output.resize(D);

    for (int t = 0; t < T; ++t)
        for (int j = 0; j < D; ++j)
            input_flat[t * D + j] = input[t][j];

    float *d_input, *d_output;
    cudaMalloc(&d_input, T * D * sizeof(float));
    cudaMalloc(&d_output, D * sizeof(float));
    cudaMemcpy(d_input, input_flat.data(), T * D * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (D + block - 1) / block;
    avg_pool_kernel<<<grid, block>>>(d_input, d_output, T, D);

    cudaMemcpy(output.data(), d_output, D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input); cudaFree(d_output);
}


void dropout_cuda(std::vector<std::vector<float>> &mat,
                  std::vector<std::vector<float>> &mask,
                  float rate, bool train=true)
{
    if (!train || rate <= 0.f) {
        mask = std::vector<std::vector<float>>(mat.size(), std::vector<float>(mat[0].size(), 1.f));
        return;
    }

    int M = mat.size(), N = mat[0].size();
    std::vector<float> mat_flat(M * N), mask_flat(M * N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            mat_flat[i * N + j] = mat[i][j];

    float *d_mat, *d_mask;
    cudaMalloc(&d_mat, M * N * sizeof(float));
    cudaMalloc(&d_mask, M * N * sizeof(float));
    cudaMemcpy(d_mat, mat_flat.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
    dropout_kernel<<<grid, block>>>(d_mat, d_mask, M, N, rate, time(NULL));

    cudaMemcpy(mat_flat.data(), d_mat, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mask_flat.data(), d_mask, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    mat.resize(M, std::vector<float>(N));
    mask.resize(M, std::vector<float>(N));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            mat[i][j] = mat_flat[i * N + j];
            mask[i][j] = mask_flat[i * N + j];
        }

    cudaFree(d_mat); cudaFree(d_mask);
}

void relu_cuda(std::vector<std::vector<float>> &mat,
               std::vector<std::vector<float>> &mask)
{
    int M = mat.size(), N = mat[0].size();
    std::vector<float> mat_flat(M * N), mask_flat(M * N);

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            mat_flat[i * N + j] = mat[i][j];

    float *d_mat, *d_mask;
    cudaMalloc(&d_mat, M * N * sizeof(float));
    cudaMalloc(&d_mask, M * N * sizeof(float));

    cudaMemcpy(d_mat, mat_flat.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    relu_kernel<<<grid, block>>>(d_mat, d_mask, M, N);

    cudaMemcpy(mat_flat.data(), d_mat, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mask_flat.data(), d_mask, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    mat.resize(M, std::vector<float>(N));
    mask.resize(M, std::vector<float>(N));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            mat[i][j] = mat_flat[i * N + j];
            mask[i][j] = mask_flat[i * N + j];
        }

    cudaFree(d_mat); cudaFree(d_mask);
}



void layernorm_cuda(std::vector<std::vector<float>>& x,
                    const std::vector<float>& gamma,
                    const std::vector<float>& beta,
                    std::vector<std::vector<float>>& y,
                    float eps = 1e-6f)
{
    int T = x.size();     // número de filas
    int D = x[0].size();  // dimensión

    std::vector<float> x_flat(T * D), y_flat(T * D);
    for (int i = 0; i < T; ++i)
        for (int j = 0; j < D; ++j)
            x_flat[i * D + j] = x[i][j];

    float *d_x, *d_y, *d_gamma, *d_beta;
    cudaMalloc(&d_x, T * D * sizeof(float));
    cudaMalloc(&d_y, T * D * sizeof(float));
    cudaMalloc(&d_gamma, D * sizeof(float));
    cudaMalloc(&d_beta, D * sizeof(float));

    cudaMemcpy(d_x, x_flat.data(), T * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma.data(), D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta.data(), D * sizeof(float), cudaMemcpyHostToDevice);

    layernorm_kernel<<<T, D>>>(d_x, d_y, d_gamma, d_beta, T, D, eps);
    cudaMemcpy(y_flat.data(), d_y, T * D * sizeof(float), cudaMemcpyDeviceToHost);

    y.resize(T, std::vector<float>(D));
    for (int i = 0; i < T; ++i)
        for (int j = 0; j < D; ++j)
            y[i][j] = y_flat[i * D + j];

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma); cudaFree(d_beta);
}


void softmax_cuda(std::vector<std::vector<float>> &mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    std::vector<float> flat(rows * cols);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            flat[i * cols + j] = mat[i][j];

    float* d_mat;
    cudaMalloc(&d_mat, rows * cols * sizeof(float));
    cudaMemcpy(d_mat, flat.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    size_t shmem = threads * sizeof(float);
    softmax_kernel<<<rows, threads, shmem>>>(d_mat, rows, cols);

    cudaMemcpy(flat.data(), d_mat, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_mat);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat[i][j] = flat[i * cols + j];
}

void transpose_cuda(const std::vector<std::vector<float>> &A,
                    std::vector<std::vector<float>> &B)
{
    int rows = A.size();
    int cols = A[0].size();
    std::vector<float> A_flat(rows * cols), B_flat(cols * rows);

    // Aplanar matriz A
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A_flat[i * cols + j] = A[i][j];

    float *d_in, *d_out;
    cudaMalloc(&d_in, rows * cols * sizeof(float));
    cudaMalloc(&d_out, cols * rows * sizeof(float));
    cudaMemcpy(d_in, A_flat.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    transpose_kernel<<<grid, block>>>(d_in, d_out, rows, cols);

    cudaMemcpy(B_flat.data(), d_out, cols * rows * sizeof(float), cudaMemcpyDeviceToHost);
    B.resize(cols, std::vector<float>(rows));
    for (int i = 0; i < cols; ++i)
        for (int j = 0; j < rows; ++j)
            B[i][j] = B_flat[i * rows + j];

    cudaFree(d_in);
    cudaFree(d_out);
}



void add_bias_cuda(std::vector<std::vector<float>>& Y, const std::vector<float>& B) {
    int M = Y.size();
    int N = Y[0].size();

    std::vector<float> Y_flat(M * N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            Y_flat[i * N + j] = Y[i][j];

    float* d_Y;
    float* d_B;
    cudaMalloc(&d_Y, M * N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMemcpy(d_Y, Y_flat.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    add_bias_kernel<<<grid, block>>>(d_Y, d_B, M, N);

    cudaMemcpy(Y_flat.data(), d_Y, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            Y[i][j] = Y_flat[i * N + j];

    cudaFree(d_Y);
    cudaFree(d_B);
}



// Aleatoriedad reproducible
std::mt19937 rng(42);
auto randf = [](float a, float b) {
    return std::uniform_real_distribution<float>(a, b)(rng);
};

// Softmax
void softmax(std::vector<float> &logits) {
    float maxv = *std::max_element(logits.begin(), logits.end());
    float denom = 0.f;
    for (float &v : logits) {
        v = std::exp(v - maxv);
        denom += v;
    }
    for (float &v : logits)
        v /= denom;
}

// Carga MNIST desde CSV
struct MuestraMnist {
    std::vector<std::vector<float>> imagen; // 28×28
    int etiqueta;
};

// Layer Normalization
struct CapaNorm {
    float eps;
    std::vector<float> gamma, beta, grad_gamma, grad_beta;
    std::vector<std::vector<float>> cache_xhat;
    std::vector<float> m_g, v_g, m_b, v_b;
    int tstep = 0;
    CapaNorm(int dim, float eps_ = 1e-6f) : eps(eps_) {
        gamma.assign(dim, 1.f); beta.assign(dim, 0.f);
        grad_gamma.assign(dim, 0.f); grad_beta.assign(dim, 0.f);
    }
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>> &x) {
        int T = x.size(), D = x[0].size();
        cache_xhat.assign(T, std::vector<float>(D));

        std::vector<std::vector<float>> y;
        layernorm_cuda(const_cast<std::vector<std::vector<float>>&>(x), gamma, beta, y, eps);
        cache_xhat = y; 
        return y;
    }
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>> &grad_out) {
        int T = grad_out.size(), D = grad_out[0].size();
        std::vector<std::vector<float>> grad_in(T, std::vector<float>(D));
        for (int j = 0; j < D; ++j) {
            float sum_gg = 0, sum_gb = 0;
            for (int t = 0; t < T; ++t) {
                sum_gg += grad_out[t][j]*cache_xhat[t][j];
                sum_gb += grad_out[t][j];
            }
            grad_gamma[j] += sum_gg;
            grad_beta[j]  += sum_gb;
        }
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < D; ++j)
                grad_in[t][j] = grad_out[t][j]*gamma[j];
        return grad_in;
    }
    void step_adam(float lr, int batch, float beta1, float beta2, float eps_opt) {
        if (m_g.empty()) {
            m_g.assign(gamma.size(),0.f); v_g.assign(gamma.size(),0.f);
            m_b.assign(beta.size(),0.f); v_b.assign(beta.size(),0.f);
        }
        tstep++;
        float lr_t = lr*std::sqrt(1 - std::pow(beta2, tstep))/(1 - std::pow(beta1, tstep));
        for (size_t i = 0; i < gamma.size(); ++i) {
            float g = grad_gamma[i]/batch;
            m_g[i] = beta1*m_g[i] + (1-beta1)*g;
            v_g[i] = beta2*v_g[i] + (1-beta2)*g*g;
            gamma[i] -= lr_t*m_g[i]/(std::sqrt(v_g[i]) + eps_opt);
            grad_gamma[i] = 0.f;
            float gb = grad_beta[i]/batch;
            m_b[i] = beta1*m_b[i] + (1-beta1)*gb;
            v_b[i] = beta2*v_b[i] + (1-beta2)*gb*gb;
            beta[i]  -= lr_t*m_b[i]/(std::sqrt(v_b[i]) + eps_opt);
            grad_beta[i]  = 0.f;
        }
    }
};

// Dropout
struct Dropout {
    float rate;
    std::vector<std::vector<float>> mask;
    Dropout(float r=0.1f): rate(r) {}
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>> &x, bool train) {
        if (!train || rate<=0) return x;
        int T=x.size(), D=x[0].size();
        mask.assign(T, std::vector<float>(D));
        auto y = x;
        for(int t=0;t<T;++t)
            for(int j=0;j<D;++j) {
                float keep = randf(0.f,1.f)>rate?1.f:0.f;
                mask[t][j] = keep;
                y[t][j] *= keep/(1.f-rate);
            }
        return y;
    }
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>> &g_out) {
        if(mask.empty()) return g_out;
        int T=g_out.size(), D=g_out[0].size();
        auto g = g_out;
        for(int t=0;t<T;++t)
            for(int j=0;j<D;++j)
                g[t][j] *= mask[t][j]/(1.f-rate);
        return g;
    }
};

// Capa lineal + Adam
struct Lineal {
    std::vector<std::vector<float>> w, m_w, v_w, grad_w;
    std::vector<float> b, m_b, v_b, grad_b;
    std::vector<std::vector<float>> cache_x;
    float beta1=0.9f, beta2=0.999f, eps_opt=1e-8f;
    int tstep=0;
    Lineal(int dim_in,int dim_out) {
        float lim = std::sqrt(6.f/(dim_in+dim_out));
        w.resize(dim_out, std::vector<float>(dim_in));
        grad_w = w; m_w = w; v_w = w;
        for(auto &fila: w)
            for(float &val: fila)
                val = randf(-lim, lim);
        b.resize(dim_out); grad_b.assign(dim_out,0.f);
        m_b.assign(dim_out,0.f); v_b.assign(dim_out,0.f);
        for(float &val: b) val = randf(-lim, lim);
    }
    std::vector<std::vector<float>> transpuesta(const std::vector<std::vector<float>> &mat) {
        int filas = mat.size(), cols = mat[0].size();
        std::vector<std::vector<float>> res(cols, std::vector<float>(filas));
        for (int i = 0; i < filas; ++i)
            for (int j = 0; j < cols; ++j)
                res[j][i] = mat[i][j];
        return res;
    }
    std::vector<float> forward(const std::vector<float> &x) {
        cache_x = {x};
        std::vector<float> y(w.size());
        for(size_t i=0;i<w.size();++i) {
            float s = b[i];
            for(size_t j=0;j<w[i].size();++j)
                s += w[i][j]*x[j];
            y[i] = s;
        }
        return y;
    }
    //Forward por lotes usando CUDA
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>> &x) {
        cache_x = x;
        int num_muestras = x.size();
        int dim_salida   = w.size();
        int dim_entrada  = w[0].size();
        std::vector<std::vector<float>> y(num_muestras, std::vector<float>(dim_salida));
        std::vector<std::vector<float>> w_T;
        transpose_cuda(w, w_T);
        matmul_cuda(x, w_T, y);
        add_bias_cuda(y, b);
        return y;
    }
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>> &g_out) {
        int N=g_out.size(), D_in=w[0].size();
        std::vector<std::vector<float>> grad_in(N, std::vector<float>(D_in,0.f));
        for(int n=0;n<N;++n) {
            auto &x = cache_x[n];
            auto &go = g_out[n];
            for(size_t i=0;i<w.size();++i) {
                grad_b[i] += go[i];
                for(size_t j=0;j<w[i].size();++j) {
                    grad_w[i][j] += go[i]*x[j];
                    grad_in[n][j] += w[i][j]*go[i];
                }
            }
        }
        return grad_in;
    }
    std::vector<float> backward(const std::vector<float> &g_out) {
        auto gm = backward(std::vector<std::vector<float>>{g_out});
        return gm[0];
    }
    void step_adam(float lr,int batch) {
        tstep++;
        float lr_t = lr*std::sqrt(1 - std::pow(beta2,tstep))/(1 - std::pow(beta1,tstep));
        for(size_t i=0;i<w.size();++i) {
            for(size_t j=0;j<w[i].size();++j) {
                float g = grad_w[i][j]/batch;
                m_w[i][j] = beta1*m_w[i][j] + (1-beta1)*g;
                v_w[i][j] = beta2*v_w[i][j] + (1-beta2)*g*g;
                w[i][j] -= lr_t*m_w[i][j]/(std::sqrt(v_w[i][j])+eps_opt);
                grad_w[i][j] = 0;
            }
        }
        for(size_t i=0;i<b.size();++i) {
            float g = grad_b[i]/batch;
            m_b[i] = beta1*m_b[i] + (1-beta1)*g;
            v_b[i] = beta2*v_b[i] + (1-beta2)*g*g;
            b[i] -= lr_t*m_b[i]/(std::sqrt(v_b[i])+eps_opt);
            grad_b[i] = 0;
        }
    }
};

// Multi-Head Attention con forward corregido y backward propagando a w_o
struct MultiCabezaAtencion {
    int d_model,h,d_k;
    Lineal w_q,w_k,w_v,w_o;
    // caches
    std::vector<std::vector<std::vector<float>>> Qh,Kh,Vh,attn,ctx;
    std::vector<std::vector<float>> x_cache;

    MultiCabezaAtencion(int dm,int nh): d_model(dm), h(nh), d_k(dm/nh),
        w_q(dm,dm), w_k(dm,dm), w_v(dm,dm), w_o(dm,dm)
    {
        assert(dm%nh==0);
    }

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>> &x) {
        x_cache = x;
        int T = x.size();
        auto Q = w_q.forward(x);
        auto K = w_k.forward(x);
        auto V = w_v.forward(x);
        // dividir en cabezas
        Qh.assign(h, std::vector<std::vector<float>>(T, std::vector<float>(d_k)));
        Kh.assign(h, std::vector<std::vector<float>>(T, std::vector<float>(d_k)));
        Vh.assign(h, std::vector<std::vector<float>>(T, std::vector<float>(d_k)));
        for(int head=0; head<h; ++head) {
            for(int t=0; t<T; ++t) {
                Qh[head][t] = {Q[t].begin() + head*d_k, Q[t].begin() + (head+1)*d_k};
                Kh[head][t] = {K[t].begin() + head*d_k, K[t].begin() + (head+1)*d_k};
                Vh[head][t] = {V[t].begin() + head*d_k, V[t].begin() + (head+1)*d_k};
            }
        }
        attn.assign(h, std::vector<std::vector<float>>(T, std::vector<float>(T)));
        ctx.assign(h,  std::vector<std::vector<float>>(T, std::vector<float>(d_k,0.f)));
        std::vector<std::vector<float>> concat(T, std::vector<float>(d_model));
        float escala = 1.f/std::sqrt((float)d_k);
        for(int head=0; head<h; ++head) {
            // scores + softmax + contexto
            for(int i=0;i<T;++i){
                for(int j=0;j<T;++j){
                    float s=0;
                    for(int k=0;k<d_k;++k)
                        s += Qh[head][i][k]*Kh[head][j][k];
                    attn[head][i][j] = s*escala;
                }
                float mx = *std::max_element(attn[head][i].begin(), attn[head][i].end());
                float den = 0;
                for(float &v: attn[head][i]) { v = std::exp(v-mx); den += v; }
                for(float &v: attn[head][i]) v /= den;
                for(int j=0;j<T;++j)
                    for(int k=0;k<d_k;++k)
                        ctx[head][i][k] += attn[head][i][j]*Vh[head][j][k];
            }
            for(int i=0;i<T;++i)
                for(int k=0;k<d_k;++k)
                    concat[i][head*d_k+k] = ctx[head][i][k];
        }
        return w_o.forward(concat);
    }

    // Propagar al menos el gradiente a w_o
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>> &grad_out) {
        auto grad_concat = w_o.backward(grad_out);
        return grad_concat;
    }

    void step_adam(float lr,int batch) {
        w_q.step_adam(lr,batch);
        w_k.step_adam(lr,batch);
        w_v.step_adam(lr,batch);
        w_o.step_adam(lr,batch);
    }
};

// FeedForward
struct FeedForward {
    Lineal l1,l2;
    Dropout drop;
    std::vector<std::vector<float>> mask_relu;
    FeedForward(int dm,int dff): l1(dm,dff), l2(dff,dm), drop(0.1f) {}
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>> &x,bool train) {
        auto h1 = l1.forward(x);
        // ReLU
        mask_relu.resize(h1.size(), std::vector<float>(h1[0].size()));
        for(size_t i=0;i<h1.size();++i)
            for(size_t j=0;j<h1[i].size();++j) {
                mask_relu[i][j] = h1[i][j]>0?1.f:0.f;
                if(mask_relu[i][j]==0) h1[i][j]=0;
            }
        auto hdrop = drop.forward(h1, train);
        return l2.forward(hdrop);
    }
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>> &g,bool train) {
        auto g2    = l2.backward(g);
        auto gdrop = drop.backward(g2);
        for(size_t i=0;i<gdrop.size();++i)
            for(size_t j=0;j<gdrop[0].size();++j)
                gdrop[i][j] *= mask_relu[i][j];
        return l1.backward(gdrop);
    }
    void step_adam(float lr,int batch) {
        l1.step_adam(lr,batch);
        l2.step_adam(lr,batch);
    }
};

// Bloque Encoder con dropouts separados
struct BloqueEncoder {
    MultiCabezaAtencion mha;
    FeedForward ff;
    CapaNorm norm1,norm2;
    Dropout dropout_atencion, dropout_ff;

    BloqueEncoder(int dm,int nh,int dff)
      : mha(dm,nh), ff(dm,dff), norm1(dm), norm2(dm),
        dropout_atencion(0.1f), dropout_ff(0.1f) {}

    std::vector<std::vector<float>> forward(std::vector<std::vector<float>> &x,bool train) {
        // Sublayer 1
        auto x1         = norm1.forward(x);
        auto attn_out   = mha.forward(x1);
        auto attn_drop  = dropout_atencion.forward(attn_out, train);
        for(size_t t=0;t<x.size();++t)
            for(size_t j=0;j<x[0].size();++j)
                x[t][j] += attn_drop[t][j];

        // Sublayer 2
        auto x2       = norm2.forward(x);
        auto ff_out   = ff.forward(x2, train);
        auto ff_drop  = dropout_ff.forward(ff_out, train);
        for(size_t t=0;t<x.size();++t)
            for(size_t j=0;j<x[0].size();++j)
                x[t][j] += ff_drop[t][j];

        return x;
    }

    std::vector<std::vector<float>> backward(std::vector<std::vector<float>> &g,bool train) {
        // FF branch
        auto g_ff      = ff.backward(dropout_ff.backward(g), train);
        auto g_norm2   = norm2.backward(g_ff);
        for(size_t t=0;t<g.size();++t)
            for(size_t j=0;j<g[0].size();++j)
                g[t][j] += g_norm2[t][j];

        // Attention branch
        auto g_attn    = mha.backward(dropout_atencion.backward(g));
        auto g_norm1   = norm1.backward(g_attn);
        for(size_t t=0;t<g.size();++t)
            for(size_t j=0;j<g[0].size();++j)
                g[t][j] += g_norm1[t][j];

        return g;
    }

    void step_adam(float lr,int batch) {
        mha.step_adam(lr,batch);
        ff.step_adam(lr,batch);
        norm1.step_adam(lr,batch,0.9f,0.999f,1e-8f);
        norm2.step_adam(lr,batch,0.9f,0.999f,1e-8f);
    }
};

// Encoder completo
struct Encoder {
    std::vector<BloqueEncoder> bloques;
    CapaNorm norm_final;
    Encoder(int N,int dm,int nh,int dff): norm_final(dm) {
        for(int i=0;i<N;++i)
            bloques.emplace_back(dm,nh,dff);
    }
    std::vector<std::vector<float>> forward(std::vector<std::vector<float>> &x,bool train) {
        for(auto &b: bloques) b.forward(x,train);
        return norm_final.forward(x);
    }
    std::vector<std::vector<float>> backward(std::vector<std::vector<float>> &g,bool train) {
        auto gr = norm_final.backward(g);
        for(int i=bloques.size()-1;i>=0;--i)
            gr = bloques[i].backward(gr,train);
        return gr;
    }
    void step_adam(float lr,int batch) {
        norm_final.step_adam(lr,batch,0.9f,0.999f,1e-8f);
        for(auto &b: bloques) b.step_adam(lr,batch);
    }
};

// Clasificador
struct ClasificadorMNIST {
    Lineal proy_pix;
    Encoder encoder;
    Lineal capa_salida;
    std::vector<std::vector<float>> pos_enc;
    int seq_len, d_model;
    ClasificadorMNIST(int sl=28,int dm=128,int N=2,int nh=4,int dff=256)
      : proy_pix(sl,dm), encoder(N,dm,nh,dff), capa_salida(dm,10),
        seq_len(sl), d_model(dm)
    {
        pos_enc.assign(sl, std::vector<float>(dm));
        for(int pos=0; pos<sl; ++pos)
            for(int i=0; i<dm; i+=2) {
                float div = std::exp(-std::log(10000.f)*i/dm);
                pos_enc[pos][i]   = std::sin(pos*div);
                if(i+1<dm)
                    pos_enc[pos][i+1] = std::cos(pos*div);
            }
    }
    std::tuple<std::vector<std::vector<float>>, std::vector<float>> forward_cache(
        std::vector<std::vector<float>> img, bool train)
    {
        auto x = proy_pix.forward(img);
        for(int t=0;t<seq_len;++t)
            for(int j=0;j<d_model;++j)
                x[t][j] += pos_enc[t][j];
        auto enc_out = encoder.forward(x,train);

        std::vector<float> pooled;
        average_pool_cuda(enc_out, pooled);
        auto logits = capa_salida.forward(pooled);
        return std::make_tuple(enc_out, logits);
    }
    void backward_batch(const std::vector<float> &d_logits) {
        auto d_pooled = capa_salida.backward(d_logits);
        std::vector<std::vector<float>> d_enc(seq_len, std::vector<float>(d_model));
        for(int t=0;t<seq_len;++t)
            for(int j=0;j<d_model;++j)
                d_enc[t][j] = d_pooled[j]/seq_len;
        auto g_enc = encoder.backward(d_enc, true);
        proy_pix.backward(g_enc);
    }
    void step_adam_batch(float lr, int batch) {
        capa_salida.step_adam(lr,batch);
        encoder.step_adam(lr,batch);
        proy_pix.step_adam(lr,batch);
    }
};

// Pérdida Cross-Entropy
float perdida_cross_entropy(const std::vector<float> &probs, int y) {
    constexpr float eps = 1e-8f;
    return -std::log(probs[y] + eps);
}

// argmax
int argmax(const std::vector<float> &v) {
    return std::max_element(v.begin(), v.end()) - v.begin();
}

std::vector<MuestraMnist> cargar_mnist_csv(const std::string &ruta) {
    std::vector<MuestraMnist> datos;
    std::ifstream f(ruta);
    if (!f.is_open()) {
        std::cerr << "No puedo abrir " << ruta << "\n";
        return datos;
    }
    std::string linea;
    // Detectar encabezado
    if (std::getline(f, linea)) {
        std::stringstream ss(linea);
        std::string tok;
        std::getline(ss, tok, ',');
        bool enc = false;
        try { static_cast<void>(std::stoi(tok)); }
        catch (...) { enc = true; }
        if (!enc) f.seekg(0);
    }
    while (std::getline(f, linea)) {
        if (linea.empty()) continue;
        std::stringstream ss(linea);
        std::string tok;
        std::getline(ss, tok, ',');
        int label = std::stoi(tok);
        std::vector<float> pix;
        while (std::getline(ss, tok, ','))
            pix.push_back(std::stof(tok) / 255.f);
        if (pix.size() != 28*28) continue;
        std::vector<std::vector<float>> img(28, std::vector<float>(28));
        for (int i = 0; i < 28; ++i)
            for (int j = 0; j < 28; ++j)
                img[i][j] = pix[i*28 + j];
        datos.push_back({img, label});
    }
    return datos;
}


int main() {
    auto dataset = cargar_mnist_csv("mnist_train.csv");
    auto test = cargar_mnist_csv("mnist_test.csv");

    verificar_gpu();


    int n_train = int(dataset.size() * 0.9);
    std::vector<MuestraMnist> train(dataset.begin(), dataset.begin() + n_train);
    std::vector<MuestraMnist> val(dataset.begin() + n_train, dataset.end());

    std::cout << "Train: " << train.size()
              << ", Val: " << val.size()
              << ", Test: " << test.size() << " muestras\n";

    ClasificadorMNIST modelo;
    float lr = 3e-4f;
    int epochs = 10;
    int batch_size = 128;

    float best_val_acc = 0.0f;
    float best_test_acc = 0.0f;
    int patience = 3;
    int wait = 0;
    int lr_wait = 0;

    for (int e = 0; e < epochs; ++e) {
        std::shuffle(train.begin(), train.end(), rng);
        float loss_acum = 0.f;
        int cont = 0;
        std::vector<std::vector<std::vector<float>>> imgs_batch;
        std::vector<int> labels_batch;

        for (size_t idx = 0; idx < train.size(); ++idx) {
            auto &m = train[idx];
            imgs_batch.push_back(m.imagen);
            labels_batch.push_back(m.etiqueta);

            if (imgs_batch.size() == batch_size || idx + 1 == train.size()) {
                for (size_t b = 0; b < imgs_batch.size(); ++b) {
                    auto tuple_out = modelo.forward_cache(imgs_batch[b], true);
                    auto &logits = std::get<1>(tuple_out);
                    std::vector<std::vector<float>> logits_mat = {logits};
                    softmax_cuda(logits_mat);
                    auto &probs = logits_mat[0];
                    
                    loss_acum += perdida_cross_entropy(probs, labels_batch[b]);
                    std::vector<float> dlog(10);
                    for (int k = 0; k < 10; ++k)
                        dlog[k] = probs[k] - (k == labels_batch[b] ? 1.f : 0.f);
                    modelo.backward_batch(dlog);
                }
                modelo.step_adam_batch(lr, imgs_batch.size());
                imgs_batch.clear();
                labels_batch.clear();
            }
            if (++cont % 2000 == 0)
                std::cout << "  " << cont << " muestras procesadas...\n";
        }

        std::cout << "Epoch " << e+1 << " - pérdida media: "
                  << (loss_acum / train.size()) << "\n";

        int aciertos_train = 0;
        for (auto &m : train) {
            auto tuple_out = modelo.forward_cache(m.imagen, false);
            auto &logits = std::get<1>(tuple_out);
            std::vector<std::vector<float>> logits_mat = {logits};
            softmax_cuda(logits_mat);
            auto &probs = logits_mat[0];
            if (argmax(probs) == m.etiqueta)
                ++aciertos_train;
        }
        float acc_train = 100.f * aciertos_train / train.size();
        std::cout << "Exactitud en entrenamiento: " << acc_train << "%\n";

        int aciertos_val = 0;
        for (auto &m : val) {
            auto tuple_out = modelo.forward_cache(m.imagen, false);
            auto &logits = std::get<1>(tuple_out);
            std::vector<std::vector<float>> logits_mat = {logits};
            softmax_cuda(logits_mat);
            auto &probs = logits_mat[0];

            if (argmax(probs) == m.etiqueta)
                ++aciertos_val;
        }
        float acc_val = 100.f * aciertos_val / val.size();
        std::cout << "Exactitud en validación: " << acc_val << "%\n";
        std::cout << "------------------------------------------\n";

        // Early stopping & LR scheduling
        if (acc_val > best_val_acc) {
            best_val_acc = acc_val;
            wait = 0;
            lr_wait = 0;
    std::vector<int> y_true, y_pred;
    float loss_test = 0.f;

    for (auto &m : test) {
        auto tuple_out = modelo.forward_cache(m.imagen, false);
        auto &logits = std::get<1>(tuple_out);
        std::vector<std::vector<float>> logits_mat = {logits};
        softmax_cuda(logits_mat);
        auto &probs = logits_mat[0];

        y_true.push_back(m.etiqueta);
        y_pred.push_back(argmax(probs));
        loss_test += perdida_cross_entropy(probs, m.etiqueta);
    }

    float acc_test = precision_micro(y_true, y_pred);  // = accuracy
    float prec = precision_micro(y_true, y_pred);
    float rec = recall_micro(y_true, y_pred);
    float f1 = f1_micro(prec, rec);
    float avg_loss = loss_test / test.size();

    if (acc_test > best_test_acc)
        best_test_acc = acc_test;

    std::cout << "  (Mejoró)\n";
    std::cout << "  >> Exactitud en test: " << acc_test << "%\n";
    std::cout << "  >> Precision (micro): " << prec << "%\n";
    std::cout << "  >> Recall    (micro): " << rec << "%\n";
    std::cout << "  >> F1 Score  (micro): " << f1 << "%\n";
    std::cout << "  >> Pérdida promedio: " << avg_loss << "\n";


        } else {
            ++wait;
            ++lr_wait;
            if (lr_wait >= 2) {
                lr *= 0.5f;
                std::cout << "Reduciendo learning rate a: " << lr << "\n";
                lr_wait = 0;
            }
            if (wait >= patience) {
                std::cout << "Early stopping activado (sin mejora en val por " << patience << " épocas)\n";
                break;
            }
        }
    }
    std::cout << "Mejor exactitud en test alcanzada: " << best_test_acc << "%\n";
    return 0;
}
    
