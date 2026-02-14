#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define BLOCK_SIZE 256

__device__ double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

__global__ void compute_gradient(
    double *X, double *y, double *w,
    double *gradient,
    int N, int d)
{
    __shared__ double local_grad[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double grad_sum = 0.0;

    if (idx < N) {
        double dot = 0.0;
        for (int j = 0; j < d; j++)
            dot += X[idx*d + j] * w[j];

        double pred = sigmoid(dot);
        grad_sum = pred - y[idx];
    }

    local_grad[tid] = grad_sum;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (tid < stride)
            local_grad[tid] += local_grad[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        gradient[blockIdx.x] = local_grad[0];
}

int main() {

    int N = 1000000;   // samples
    int d = 10;        // features
    double lr = 0.01;
    int epochs = 50;

    size_t sizeX = N * d * sizeof(double);
    size_t sizeY = N * sizeof(double);

    double *h_X = (double*)malloc(sizeX);
    double *h_y = (double*)malloc(sizeY);
    double *h_w = (double*)malloc(d * sizeof(double));

    for(int i=0;i<N*d;i++)
        h_X[i] = rand()/(double)RAND_MAX;

    for(int i=0;i<N;i++)
        h_y[i] = rand()%2;

    for(int j=0;j<d;j++)
        h_w[j] = 0.0;

    double *d_X, *d_y, *d_w, *d_grad;

    cudaMalloc(&d_X, sizeX);
    cudaMalloc(&d_y, sizeY);
    cudaMalloc(&d_w, d*sizeof(double));
    cudaMalloc(&d_grad, (N/BLOCK_SIZE+1)*sizeof(double));

    cudaMemcpy(d_X, h_X, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sizeY, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, d*sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for(int e=0; e<epochs; e++) {

        compute_gradient<<<grid, block>>>(d_X, d_y, d_w, d_grad, N, d);
        cudaDeviceSynchronize();

        double *h_grad = (double*)malloc(grid.x*sizeof(double));
        cudaMemcpy(h_grad, d_grad, grid.x*sizeof(double),
                   cudaMemcpyDeviceToHost);

        double total_grad = 0.0;
        for(int i=0;i<grid.x;i++)
            total_grad += h_grad[i];

        total_grad /= N;

        for(int j=0;j<d;j++)
            h_w[j] -= lr * total_grad;

        cudaMemcpy(d_w, h_w, d*sizeof(double),
                   cudaMemcpyHostToDevice);

        free(h_grad);
    }

    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_w);
    cudaFree(d_grad);

    free(h_X);
    free(h_y);
    free(h_w);

    printf("Training completed.\n");
    return 0;
}
