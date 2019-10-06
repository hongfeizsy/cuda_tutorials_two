#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define get_idx() (threadIdx.x)

__global__ void sum(float *x) {
	int idx = get_idx();
	x[idx] += 1;
}

int main() {
	int N = 32;
	int nbytes = N * sizeof(float);

	float *dx = NULL, *hx = NULL;

	/* allocate GPU memory */
	cudaMalloc((void **)&dx, nbytes);
	if (dx == NULL) {
		printf("couldn't allocate GPU memory");
		return -1;
	}

	/* allocate CPU memory */
	hx = (float*) malloc(nbytes);
	//cudaMalloc((void **)&hx, nbytes);

	if (hx == NULL) {
		printf("couldn't allocate CPU memory");
		return -2;
	}

	/* init */
	printf("hx original: \n");
	for (int i = 0; i < N; i++) {
		hx[i] = i;
		printf("%g\n", hx[i]);
	}

	/* copy data to GPU */
	cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);

	/* call GPU */
	sum <<<1, N>>> (dx);

	/* let GPU finish */
	cudaThreadSynchronize();

	/* copy data from GPU */
	cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		printf("%g\n", hx[i]);
	}
	
	return 0;
}





