#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		
		__global__
		void kernUpSweep(int n, int d, int *data, int offset_1, int offset_2) {
			int k = threadIdx.x + (blockIdx.x * blockDim.x);
			if (k % offset_1 != 0) { return; }
			if (k > n) { return; }
			data[k + offset_1 - 1] += data[k + offset_2 - 1];
			if (k == n - 1) { data[k] = 0; }
		}

		__global__
		void kernDownSweep(int n, int d, int *data, int offset_1, int offset_2) {
			int k = threadIdx.x + (blockIdx.x * blockDim.x);
			if (k % offset_1 != 0) { return; }
			if (k > n) { return; }
			int t = data[k + offset_2 - 1];
			data[k + offset_2 - 1] = data[k + offset_1 - 1];
			data[k + offset_1 - 1] += t;
		}

		__global__
			void kernZeroCorrect(int n, int *data) {
			int k = threadIdx.x + (blockIdx.x * blockDim.x);
			if (k > n) { return; }
			data[k] -= data[0];
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int paddedSize = 1 << ilog2ceil(n);

			int *idataPadded = new int[paddedSize];
			for (int i = 0; i < paddedSize; ++i) {
				idataPadded[i] = i < n ? idata[i] : 0;
			}

			int blockSize = 128;
			dim3 blocksPerGrid((paddedSize + blockSize - 1) / blockSize);
			dim3 threadsPerBlock(blockSize);

			int *dv_data;
			cudaMalloc((void **) &dv_data, paddedSize * sizeof(int));
			checkCUDAError("cudaMalloc dv_data failed!");

			cudaMemcpy(dv_data, idataPadded, paddedSize * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy to dv_data failed!");

			bool end = true;
			try {
				timer().startGpuTimer();
			} catch (std::exception &) {
				end = false;
			}

			for (int i = 0; i < ilog2ceil(n); ++i) {
				kernUpSweep << <blocksPerGrid, threadsPerBlock >> > (paddedSize, i, dv_data, 1 << (i + 1), 1 << i);
			}

			// set root to 0
			int z = 0;
			cudaMemcpy(dv_data + n - 1, &z, sizeof(int), cudaMemcpyHostToDevice);

			for (int i = ilog2ceil(n) - 1; i >= 0; i--) {
				kernDownSweep << <blocksPerGrid, threadsPerBlock >> > (paddedSize, i, dv_data, 1 << (i + 1), 1 << i);
			}

			if (end) { timer().endGpuTimer(); }
			kernZeroCorrect << <blocksPerGrid, threadsPerBlock >> > (paddedSize, dv_data);
			cudaMemcpy(odata, dv_data, n * sizeof(int), cudaMemcpyDeviceToHost);

			delete idataPadded;
			cudaFree(dv_data);
        }

		__global__
		void kernMapToBoolean(int n, int *odata, int *idata) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx > n) { return; }
			odata[idx] = idata[idx] == 0 ? 0 : 1;
		}

		__global__
		void kernScatter(int n, int *odata, int *bdata, int *scandata, int *idata) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx > n) { return; }
			if (bdata[idx] == 1) {
				odata[scandata[idx]] = idata[idx];
			}
		}

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

			int blockSize = 128;
			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
			dim3 threadsPerBlock(blockSize);

			int *dv_bdata, *dv_scandata, *dv_idata, *dv_data;
			cudaMalloc((void **) &dv_bdata, n * sizeof(int));
			checkCUDAError("cudaMalloc dv_bdata failed!");

			cudaMalloc((void **) &dv_scandata, n * sizeof(int));
			checkCUDAError("cudaMalloc dv_scandata failed!");

			cudaMalloc((void **) &dv_data, n * sizeof(int));
			checkCUDAError("cudaMalloc dv_data failed!");

			cudaMalloc((void **) &dv_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dv_idata failed!");

			cudaMemcpy(dv_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy to dv_idata failed!");

			int *cpu_bdata, *cpu_scandata;
			cpu_bdata = new int[n];
			cpu_scandata = new int[n];

            timer().startGpuTimer();

			kernMapToBoolean << <blocksPerGrid, threadsPerBlock >> > (n, dv_bdata, dv_idata);

			cudaMemcpy(cpu_bdata, dv_bdata, n * sizeof(int), cudaMemcpyDeviceToHost);

			int count = 0;
			for (int i = 0; i < n; ++i) {
				if (cpu_bdata[i] == 1) { count++; }
			}

			scan(n, cpu_scandata, cpu_bdata);

			cudaMemcpy(dv_scandata, cpu_scandata, n * sizeof(int), cudaMemcpyHostToDevice);

			kernScatter<<<blocksPerGrid, threadsPerBlock>>>(n, dv_data, dv_bdata, dv_scandata, dv_idata);

            timer().endGpuTimer();

			cudaMemcpy(odata, dv_data, count * sizeof(int), cudaMemcpyDeviceToHost);

			delete(cpu_bdata);
			delete(cpu_scandata);
			cudaFree(dv_bdata);
			cudaFree(dv_scandata);
			cudaFree(dv_idata);
			cudaFree(dv_data);

            return count;
        }
    }
}
