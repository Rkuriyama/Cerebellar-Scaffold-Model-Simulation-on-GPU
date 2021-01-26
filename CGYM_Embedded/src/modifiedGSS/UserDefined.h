#ifndef __GSS_USER_DEFINED_H_
#define __GSS_USER_DEFINED_H_
#define INPUT_STIM_NUM 3
#define DEV_NUM 1

#include "struct_enum_def.h"
void init_neurons_params( Neuron*, int* );
void init_connectivity_params( Connectivity*, Neuron*, int*, int* );

__host__ void InitInputStimulation( InputFunctionsStruct *, Neuron*, int *);

void Init_Plasticity( STDP_PLASTICITY **p, int *ConnectivityTypeID );
#endif
