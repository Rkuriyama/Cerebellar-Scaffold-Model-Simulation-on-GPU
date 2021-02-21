#ifndef _SYN_PLAS_H_
#define _SYN_PLAS_H_

#include "struct_enum_def.h"

__global__ void PF_PC_LTD_LTP_ELL( char *spike, int max_conv, int *cindices, CTYPE *val, int teacher_max_conv, int *teacher_cindices,
                                int target_row, int tail,
                                int post_num, int pre_base, int teacher_base, int delay_pre, int delay_teacher, int total_nn );

void Init_Plasticity( STDP_PLASTICITY **p, int *ConnectivityTypeID );

__host__ void invoke_stdp_plasticity( char *spike, Neuron *d_neurons,  Connectivity *d_connections, STDP_PLASTICITY *p, int target_row, int tail, int total_nn, cudaStream_t *streams );

/*
__global__ void HebbianRule( STDP *plasticity, char *spike, unsigned int *rptr, unsigned int *cindices, CTYPE *val,
                             int target_row, int tail,
                             int post_num, int post_base, int pre_base, int total_nn);

__global__ void PerceptronRule( STDP plasticity, char *spike, unsigned int *rptr, unsigned int *cindices, CTYPE *val, unsigned int *teacher_rptr, unsigned int *teacher_cindices,
                                int target_row, int tail,
                                int post_num, int pre_base, int teacher_base, int total_nn );
*/

#endif
