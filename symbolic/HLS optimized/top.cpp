extern "C" {
    typedef float data_t;
    const int BATCH = 16;
    const int K = 4;
    const int L = 256;

    // Function 1: Circular padding of input A
    void pad_input(data_t A[K][L], data_t padded_A[2*L-1]) {
        #pragma HLS INLINE off
        pad_loop: for (int p = 0; p < 2*L-1; p++) {
            #pragma HLS PIPELINE II=1
            padded_A[p] = A[0][p % L];  // Assume j=0 (will be called per j)
        }
    }

    // Function 2: Kernel preparation (flip and shift)
    void prepare_kernel(data_t B[K][L], data_t kernel[L]) {
        #pragma HLS INLINE off
        kernel_prep: for (int m = 0; m < L; m++) {
            #pragma HLS PIPELINE II=1
            kernel[m] = B[0][(L - 1 - m + 1) % L];  // Assume j=0 (will be called per j)
        }
    }

    // Function 3: Systolic convolution (MAC operations)
    void compute_convolution(
        data_t padded_A[2*L-1],
        data_t kernel[L],
        data_t C[K][L],
        int j
    ) {
        #pragma HLS INLINE off
        conv_loop: for (int n = 0; n < L; n++) {
            #pragma HLS PIPELINE II=1
            data_t acc = 0;
            systolic_mac: for (int k = 0; k < L; k++) {
                acc += padded_A[n + k] * kernel[k];
            }
            C[j][n] = acc;
        }
    }

    // Top-level function with DATAFLOW parallelism
    void circular_convolution_3d(
        data_t A[BATCH][K][L],
        data_t B[BATCH][K][L],
        data_t C[BATCH][K][L]
    ) {
        #pragma HLS DATAFLOW

        for (int i = 0; i < BATCH; i++) {
            #pragma HLS PERFORMANCE target_ti=1

            for (int j = 0; j < K; j++) {
                #pragma HLS PERFORMANCE target_ti=1

                // Local buffers (partitioned for parallelism)
                data_t padded_A[2*L-1];
                #pragma HLS ARRAY_PARTITION variable=padded_A cyclic factor=4 dim=1

                data_t kernel[L];
                #pragma HLS ARRAY_PARTITION variable=kernel cyclic factor=4 dim=1

                // Execute all stages in parallel per (i,j)
                #pragma HLS DATAFLOW
                pad_input(A[i], padded_A);
                prepare_kernel(B[i], kernel);
                compute_convolution(padded_A, kernel, C[i], j);
            }
        }
    }
}
