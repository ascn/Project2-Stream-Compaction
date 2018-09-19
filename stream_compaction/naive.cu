#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
	namespace Naive {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer() {
			static PerformanceTimer timer;
			return timer;
		}

		__global__
		void kernNaiveScanIteration(int n, int d, int *o, const int *i) {
			int k = threadIdx.x + (blockIdx.x * blockDim.x);
			if (k >= n) { return; }
			int offset = 1 << (d - 1);
			o[k] = k >= offset ? i[k - offset] + i[k] : i[k];
		}
		
		__global__
		void kernShiftRight(int n, int *o, int *i) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) { return; }
			o[index] = index == 0 ? 0 : i[index - 1];
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int blockSize = 128;
			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
			dim3 threadsPerBlock(blockSize);

			int *dv_idata, *dv_odata;
			cudaMalloc((void **) &dv_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dv_idata failed!");

			cudaMalloc((void **) &dv_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dv_odata failed!");

			cudaMemcpy(dv_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy to dv_idata failed!");

			timer().startGpuTimer();

			for (int d = 1; d <= ilog2ceil(n); ++d) {
				kernNaiveScanIteration << <blocksPerGrid, threadsPerBlock >> > (n, d, dv_odata, dv_idata);
				std::swap(dv_idata, dv_odata);
			}
			kernShiftRight << <blocksPerGrid, threadsPerBlock >> > (n, dv_odata, dv_idata);

			timer().endGpuTimer();

			cudaMemcpy(odata, dv_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dv_idata);
			cudaFree(dv_odata);
        }
    }
}
