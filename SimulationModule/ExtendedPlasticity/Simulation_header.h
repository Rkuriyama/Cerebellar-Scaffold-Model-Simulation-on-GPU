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

__global__ void spike_propagation(const int post_base_id, const int postNum, CTYPE *dg, const int max_conv, const int *cindices,  const CTYPE *weight, const CTYPE w_bar, const char *spike, const int base);
__global__ void spike_propagation_mThreads(const int post_base_id, const int postNum, CTYPE *dg, const int max_conv, const int *cindices,  const CTYPE *weight, const CTYPE w_bar, const char *spike, const int base, const unsigned int threadsPerNeuron);
__host__   void spike_propagation_PR(CTYPE *out, CTYPE *tmp, const int max_conv, const int pre_base_id,const int postNum,const int post_base_id, CTYPE *dg, const int *cindices, const CTYPE *weight, const CTYPE w_bar, const char *spike,const int row,  const int total_nn, cudaStream_t stream);

__global__ void update(CTYPE*, CTYPE*, CTYPE*, CTYPE*, CTYPE*, CTYPE*, char*, int*, const Neuron*, const char*, const int, const int);
__global__ void update_lif(CTYPE*, CTYPE*, CTYPE*, CTYPE*, CTYPE*, CTYPE*, char*, int*, const Neuron*, const char*, const int, const int, const int);

__host__ void host_spike_propagation(const int pre_type, const int postNum, const int post_base_id, CTYPE *dg, const int max_conv, const int *cindices,  const CTYPE *weight, const CTYPE w_bar, const int target_row, const int target_block,  const Sim_cond_lif_exp *Dev);

__host__ void host_update(CTYPE *u, CTYPE *g_exc, CTYPE *dg_exc, CTYPE *g_inh, CTYPE *dg_inh,CTYPE *Inoise, char *spike,int *refractory_time_left ,const Neuron *Neurons,const char *type_array,const int target_row, const int total_nn, const float t, FILE *fp, int *NeuronTypeID);

#endif
