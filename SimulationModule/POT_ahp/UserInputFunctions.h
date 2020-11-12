#ifndef USER_INPUT_FUNCTION_H
#define USER_INPUT_FUNCTION_H


#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "struct_enum_def.h"
#include "option.h"
#include "Simulation_header.h"

__device__ int GlomTargetInput(const float, const CTYPE);
__host__ void InitInputStimulation( InputFunctionsStruct *, Neuron*, int *);
__global__ void InputStimulation( const int n, char *spike,
                                  curandStatePhilox4_32_10_t *state,
                                  const int num, const int base_id,
                                  const unsigned int *IdList,
                                  const int target_row, const int total_nn,
	       			  int func_id);
#define INPUT_STIM_NUM 2
#endif 
