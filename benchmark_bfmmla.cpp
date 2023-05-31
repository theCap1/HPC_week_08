#include <iostream>
#include <chrono>
#include <cstdint>
#include <arm_bf16.h>
#include <arm_neon.h>

#define DTYPE16 bfloat16_t
#define DTYPE32 float32_t

extern "C" {
    void bfmmla_kernel( DTYPE16 *i_a,
                        DTYPE16 *i_b,
                        DTYPE32 *io_c);
}

int main(){
    int64_t i_m = 4;
    int64_t i_n = 4;
    int64_t i_k = 4;
    
    DTYPE16 *i_a = (DTYPE16*) malloc(i_m*i_k*sizeof(DTYPE16));
    DTYPE16 *i_b = (DTYPE16*) malloc(i_k*i_n*sizeof(DTYPE16));
    DTYPE32 *io_c = (DTYPE32*) malloc(i_m*i_n/2*sizeof(DTYPE32));
    DTYPE32 *c_old = (DTYPE32*) malloc(i_m*i_n*sizeof(DTYPE32));

    DTYPE32 a = 0;
    for (int i = 0; i < i_k*i_m; i++)
    {
        a = i;
        *(i_a+i) = vcvth_bf16_f32(a);
    }

    for (int i = 0; i < i_k*i_n; i++)
    {
        a = (i_k*i_n-i);
        *(i_b+i) = vcvth_bf16_f32(a);
    }

    for (int i = 0; i < i_m*i_n; i++)
    {
        *(io_c+i) = i;
        *(c_old+i) = i;
    }

    std::cout << "running SVE GEMM microbenchmarks" << std::endl;
    std::chrono::steady_clock::time_point l_tp0, l_tp1;
    std::chrono::duration< double > l_dur;
    double l_g_flops = 0;
    int l_n_threads = 1;
    uint64_t l_n_repetitions = 1;
    l_n_repetitions *= 10000000;

    bfmmla_kernel(  i_a,
                    i_b,
                    io_c ); // dry run

    l_tp0 = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < l_n_repetitions; i++)
    {
        bfmmla_kernel(  i_a,
                        i_b,
                        io_c );
    }
    l_tp1 = std::chrono::steady_clock::now();

    l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );

    std::cout << "  # of executions: " << l_n_repetitions << std::endl;
    std::cout << "  duration: " << l_dur.count() << " seconds" << std::endl;
    std::cout << "  average duration: " << l_dur.count()/l_n_repetitions << " seconds" << std::endl;
    l_g_flops = 2*i_m*i_n*i_k;
    l_g_flops *= l_n_threads;
    l_g_flops *= l_n_repetitions;
    l_g_flops *= 1.0E-9;
    l_g_flops /= l_dur.count();
    std::cout << "  GFLOPS: " << l_g_flops << std::endl;

    free(i_a);
    free(i_b);
    free(io_c);
}