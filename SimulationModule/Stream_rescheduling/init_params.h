#ifndef __INIT_PARAMS_FOR_SIM_H_
#define __INIT_PARAMS_FOR_SIM_H_

#include "struct_enum_def.h"
#include <curand_kernel.h>

#define CUDA_SAFE_CALL(func) \
do { \
     cudaError_t err = (func); \
     if (err != cudaSuccess) { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(err); \
     } \
} while(0)



//int set_neuron_params( Neuron* ,enum NeuronType, const char*, int, int, CTYPE, CTYPE, CTYPE, CTYPE, CTYPE, CTYPE, CTYPE, CTYPE, CTYPE );
//void set_connectivity_params(Connectivity*, Neuron*, enum ConnectionType,const char*, int, int, int, int, CTYPE, CTYPE, int );
void init_neurons_params( Neuron*, int* );
void init_connectivity_params( Connectivity*, Neuron*, int*, int* );


int LoadConnectivityFile(const char *,unsigned int **,  unsigned int **, unsigned int **, CTYPE **, CTYPE, int , int );
void Initialize_Weight_matrixes( Connectivity*, int**, const Neuron*);

__global__ void check_consistency( unsigned int *csr_rptr, unsigned int *csr_cindices, CTYPE *csr_val, int *ell_cindices, CTYPE *ell_val, unsigned int max_conv, int post_num );

int set_base_id( Neuron* );

__global__ void InitParams( CTYPE*,  CTYPE*, CTYPE*, CTYPE*, CTYPE*, int*, char*, Neuron*, char*, curandStatePhilox4_32_10_t*, const int);
__host__ void Host_InitParams( CTYPE *u, CTYPE *g_exc, CTYPE *dg_exc, CTYPE *g_inh, CTYPE *dg_inh, int *refractory_time_left, char *spike , Neuron *Neurons ,char *type, const int total_nn);

#endif
