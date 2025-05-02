extern "C" {
    typedef float data_t;
    const int BATCH = 16;
    const int K = 4;
    const int L = 256;

    void circular_convolution_3d(data_t A[BATCH][K][L],
                               data_t B[BATCH][K][L],
                               data_t C[BATCH][K][L]) {
        // Temporary buffer for padded input (L + L-1 elements)
        data_t padded_A[2*L-1];
        
        for (int i = 0; i < BATCH; i++) {
            for (int j = 0; j < K; j++) {
                // 1. Create circularly padded A (matches F.pad(A, (0,L-1), mode='circular'))
                for (int p = 0; p < 2*L-1; p++) {
                    padded_A[p] = A[i][j][p % L];
                }
                
                // 2. Prepare kernel 
                data_t kernel[L];
                for (int m = 0; m < L; m++) {
                    // Equivalent to flip + roll by 1
                    kernel[m] = B[i][j][(L - 1 - m + 1) % L];
                }
                
                // 3. Perform the convolution
                for (int n = 0; n < L; n++) {
                    data_t acc = 0;
                    for (int k = 0; k < L; k++) {
                        acc += padded_A[n + k] * kernel[k];
                    }
                    C[i][j][n] = acc;
                }
            }
        }
    }
}