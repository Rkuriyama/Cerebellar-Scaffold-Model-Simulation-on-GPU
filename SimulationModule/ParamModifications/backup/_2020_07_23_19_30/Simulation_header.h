#ifndef _SIM_HEADER_H_
#define _SIM_HEADER_H_

#include "struct_enum_def.h"
#include <curand_kernel.h>

#define generate_noise_current(I,m,std,N) curandGenerateNormal(gen, I, N, m, std)



#define CUDA_SAFE_CALL(func) \
do { \
     cudaError_t err = (func); \
     if (err != cudaSuccess) { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(err); \
     } \
} while(0)



__global__ void Philox_setup_kernel(unsigned long long seed, curandStatePhilox4_32_10_t *state, unsigned int N);
__global__ void Philox_generate_normal(CTYPE *a, CTYPE mean, CTYPE std_div, curandStatePhilox4_32_10_t *state, unsigned int N);
__global__ void Philox_generate_uniform(CTYPE *a, curandStatePhilox4_32_10_t *state, unsigned int N);
__global__ void Philox_generate_uniform4(CTYPE *a, curandStatePhilox4_32_10_t *state, unsigned int N);

//__device__ int step_input( const CTYPE, const float );

/*
__global__ void InputStimulation( const int n, int *spike,
                                  curandStatePhilox4_32_10_t *state,
                                  const int num, const int base_id,
                                  const unsigned int *rptr, const unsigned int *cindices, const CTYPE *weight,
                                  const int target_row, const int total_nn );
*/
/*
__global__ void InputStimulationMT( const int n, int *spike,
                                  float *state,
                                  const int num, const int base_id,
                                  const unsigned int *rptr, const unsigned int *cindices, const CTYPE *weight,
                                  const int target_row, const int total_nn );
*/
//__global__ void calculate_current_diff( const int, const int, const int, const int, CTYPE*, const int*, const unsigned int*, const unsigned int*, const CTYPE* ,const int*, const int,  const int );

__global__ void calculate_current_diff(const int ,const int,const int,const int, CTYPE *,const int *, const unsigned int *, const unsigned int *,  const CTYPE *, const CTYPE, const char *,const int,  const int);
__global__ void calculate_current_diff_arrange(const int ,const int,const int,const int, CTYPE *,const int *, const unsigned int *, const unsigned int *,  const CTYPE *, const CTYPE, const char *,const int,  const int, const unsigned int);


__global__ void update(CTYPE*, CTYPE*, CTYPE*, CTYPE*, CTYPE*, CTYPE*, char*, int*, const Neuron*, const char*, const int, const int);
//__global__ void update_RK2(CTYPE*, CTYPE*, CTYPE*, CTYPE*, CTYPE*, CTYPE*, int*, int*, const Neuron*, const int*, const int, const int);


//__global__ void init_in_for_reduction( CTYPE*,CTYPE*, const int*, const int*,const unsigned int*, const unsigned int*, const CTYPE*, const int, const int, const unsigned int );
//__device__ void warpReduce(volatile CTYPE*, int);
//__global__ void reduce5( CTYPE*, CTYPE*, int );
//__global__ void add_tmp_to_dg( CTYPE*, CTYPE*, const int*, const int, const int );

__host__ void calc_current_diff_PR(CTYPE*, CTYPE*, const int,const int,const int,const int, CTYPE*,const int* ,const unsigned int*,const unsigned int*, const CTYPE*, const CTYPE, const char*,const int,  const int, cudaStream_t);



__host__ void host_calculate_current_diff(const int preType, const int postNum, const int post_base_id, CTYPE *dg,const int *refractory_time_left , const unsigned int *rptr, const unsigned int *cindices,  const CTYPE *weight, const int target_row, const int target_block,  const Sim_cond_lif_exp *Dev);
__host__ void host_update(CTYPE *u, CTYPE *g_exc, CTYPE *dg_exc, CTYPE *g_inh, CTYPE *dg_inh,CTYPE *Inoise, char *spike,int *refractory_time_left ,const Neuron *Neurons,const char *type_array,const int target_row, const int total_nn, const float t, FILE *fp, int *NeuronTypeID);

#endif
