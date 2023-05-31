#include <iostream>  
#include <chrono>
#include <cstdint>
#include <arm_bf16.h>
#include <arm_neon.h>

#define DTYPE16 bfloat16_t
#define DTYPE32 float32_t

// extern "C" {
//     void bfmmla_kernel( DTYPE16 *i_a,
//                         DTYPE16 *i_b,
//                         DTYPE32 *io_c);
// }

void convert_a_to_bfmmla( uint64_t           i_m,
                          uint64_t           i_n,
                          uint64_t           i_ld,
                          bfloat16_t const * i_a_col_major,
                          bfloat16_t       * o_a_fmmla ){
    
    for (int i = 0; i < i_m*i_n/8; i++)     // loop through blocks
    {
        for (int j = 0; j < 2; i++)         // loop through columns
        {
            for (int k = 0; k < 4; i++)     // loop through rows
            {
                *(o_a_fmmla+i*2+j+k*i_ld) = *(i_a_col_major+i*8+j*4+k);
            }
        }
    }
    
}

void convert_b_to_bfmmla( uint64_t           i_m,
                          uint64_t           i_n,
                          uint64_t           i_ld,
                          bfloat16_t const * i_b_col_major,
                          bfloat16_t       * o_b_fmmla ){
    
    for (int i = 0; i < i_m*i_n/8; i++)     // loop through blocks
    {
        for (int j = 0; j < 2; i++)         // loop through rows
        {
            for (int k = 0; k < 4; i++)     // loop through columns
            {
                *(o_b_fmmla+i*4+k+j*i_ld) = *(i_b_col_major+i*8+j*4+k);
            }
        }
    }
    
}

void convert_c_to_bfmmla( uint64_t         i_m,
                          uint64_t         i_n,
                          uint64_t         i_ld,
                          float    const * i_c_col_major,
                          float          * o_c_fmmla ){
    
    for (int i = 0; i < i_m*i_n/4; i++)     // loop through blocks
    {
        for (int j = 0; j < 2; i++)         // loop through rows
        {
            for (int k = 0; k < 2; i++)     // loop through columns
            {
                *(o_c_fmmla+i*2+k+j*i_ld) = *(i_c_col_major+i*8+j*2+k);
            }
        }
    }
    
}

void convert_c_from_bfmmla( uint64_t         i_m,
                            uint64_t         i_n,
                            uint64_t         i_ld,
                            float    const * i_c_fmmla,
                            float          * o_c_col_major ){
    
    for (int i = 0; i < i_m*i_n/4; i++)     // loop through blocks
    {
        for (int j = 0; j < 2; i++)         // loop through rows
        {
            for (int k = 0; k < 2; i++)     // loop through columns
            {
                *(o_c_col_major+i*2+k+j*i_ld) = *(i_c_fmmla+i*8+j*2+k);
            }
        }
    }

}

void main(){

    uint64_t i_m = 8;
    uint64_t i_n = 8;
    uint64_t i_k = 2;
    uint64_t i_lda = i_m;
    uint64_t i_ldb = i_k;
    uint64_t i_ldc = i_m;

    DTYPE16 *i_a_col_major = (DTYPE16*) malloc(i_m*i_k*sizeof(DTYPE16));
    DTYPE16 *o_a_fmmla = (DTYPE16*) malloc(i_m*i_k*sizeof(DTYPE16));
    DTYPE16 *i_b_col_major = (DTYPE16*) malloc(i_k*i_n*sizeof(DTYPE16));
    DTYPE16 *o_b_fmmla = (DTYPE16*) malloc(i_k*i_n*sizeof(DTYPE16));
    DTYPE32 *i_c_col_major = (DTYPE32*) malloc(i_m*i_n/2*sizeof(DTYPE32));
    DTYPE32 *o_c_fmmla = (DTYPE32*) malloc(i_m*i_n/2*sizeof(DTYPE32));

    DTYPE32 a = 0;
    for (int i = 0; i < i_k*i_m; i++)
    {
        a = i;
        *(i_a_col_major+i) = vcvth_bf16_f32(a);
        *(o_a_fmmla+i) = vcvth_bf16_f32(a);
    }

    for (int i = 0; i < i_k*i_n; i++)
    {
        a = (i_k*i_n-i);
        *(i_b_col_major+i) = vcvth_bf16_f32(a);
        *(o_b_fmmla+i) = vcvth_bf16_f32(a);
    }

    for (int i = 0; i < i_m*i_n; i++)
    {
        *(i_c_col_major+i) = i;
        *(o_c_fmmla+i) = i;
    }

    convert_a_to_bfmmla(i_m,
                        i_n,
                        i_lda,
                        i_a_col_major,
                        o_a_fmmla )

    std::cout << "running SVE GEMM microbenchmarks" << std::endl;
    std::chrono::steady_clock::time_point l_tp0, l_tp1;
    std::chrono::duration< double > l_dur;
    double l_g_flops = 0;
    int l_n_threads = 1;
    uint64_t l_n_repetitions = 1;
    l_n_repetitions *= 10000000;

    bfmmla_kernel(  o_a_fmmla,
                    o_a_fmmla,
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

    free(i_a_col_major);
    free(o_a_fmmla);
    free(i_b_col_major);
    free(o_b_fmmla);
    free(i_c_col_major);
    free(o_c_fmmla);
}