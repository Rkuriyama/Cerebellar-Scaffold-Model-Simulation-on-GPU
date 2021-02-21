#ifndef _SYN_PLAS_H_
#define _SYN_PLAS_H_

#include "struct_enum_def.h"

void Init_Plasticity( STDP_PLASTICITY **p, int *ConnectivityTypeID );

__host__ void invoke_stdp_plasticity( char *spike, Neuron *d_neurons,  Connectivity *d_connections, STDP_PLASTICITY *p, int target_row, int tail, int total_nn, cudaStream_t *streams );

#endif
