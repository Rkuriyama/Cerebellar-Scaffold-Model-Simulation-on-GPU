#ifndef __GPU_SCAFFOLD_SIMULATION_H__
#define __GPU_SCAFFOLD_SIMULATION_H__

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>

#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <omp.h>

#include <string>

#include "init_params.h"
#include "Simulation_header.h"
#include "option.h"

#include "UserInputFunctions.h"
#include "SynapticPlasticity.h"



#define DEBUG 0
#define PROGRESS 0
#define DEBUG_T 500
#define DEBUG_LOG "dlog.log"
#define MEASURE 1

#define PRINT 1
#define N_PRINT_THREDS 2

#define DEV_NUM 1


////////////// prototype declaration
void CreateNetworkEnv(  NetworkEnvironment *env, int num_of_dev, int num_of_neuron_types, int num_of_connectivity_types, int num_of_inputs, int num_of_plasticities);
void InitializeNetworkEnv( NetworkEnvironment *env, int T_print, int T_cpu_sim, std::string dir );
void ResetNetworkEnv(NetworkEnvironment *env, int trial);
void FinalizeNetworkEnv( NetworkEnvironment *env );
void loop_n_steps( NetworkEnvironment *env, int start_step, int n_steps, int trial, char **output_spike_train);
void SetInputFreq(NetworkEnvironment *env, const float *freq_list);
////////////////////////////////////

#endif
