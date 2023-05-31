#include <iostream>  
#include <arm_bf16.h>
#include <arm_neon.h>

#define DTYPE16 bfloat16_t
#define DTYPE32 float32_t


// extern "C" {
//     void bfmmla_kernel( DTYPE16 *i_a,
//                         DTYPE16 *i_b,
//                         DTYPE32 *io_c);
// }

int main(){//   int i_argc,  
            // char const * i_argv[] ) {
                
    // if( i_argc != 2 ) {
    //     std::cerr << "error, run as ./sve_examples EXAMPLE_ID" << std::endl;
    // }

    int64_t i_m = 8;
    int64_t i_n = 8;
    int64_t i_k = 2;

    DTYPE16 *i_a = (DTYPE16*) malloc(i_m*i_k*sizeof(DTYPE16));
    DTYPE16 *i_b = (DTYPE16*) malloc(i_k*i_n*sizeof(DTYPE16));
    DTYPE32 *io_c = (DTYPE32*) malloc(i_m*i_n/2*sizeof(DTYPE32));
    DTYPE32 *c_old = (DTYPE32*) malloc(i_m*i_n/2*sizeof(DTYPE32));

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

    // bfmmla_kernel(  i_a,
    //                 i_b,
    //                 io_c );

    for(int c=0; c<i_k; c++)
    {
        std::cout << "\t\t\t";
        for(int r=0; r<i_n; r++)
        {
            std::cout << vcvtah_f32_bf16(*(i_b +r*i_k + c)) << "\t";
        }
        printf("\n");
    }

    for(int i=0; i<i_m; ++i)
    {
        for(int j=0; j<i_k; ++j)
        {
            std::cout << vcvtah_f32_bf16(*(i_a +j*i_m + i)) << "\t";
        }
        std::cout << "\t";
        for(int r=0; r<i_n; r++) // i_n = 6
        {
            std::cout << *(io_c +r*i_m + i) << "\t";
        }
        std::cout << "\t";
        for(int r=0; r<i_n; r++) // i_n = 6
        {
            std::cout << *(c_old +r*i_m + i) << "\t";
        }
        std::cout << "\t";
        printf("\n");
    }

    free(i_a);
    free(i_b);
    free(io_c);
    free(c_old);
    
    
    return EXIT_SUCCESS;
}
