#include "UserInputFunctions.h"

__device__ char PoissonProcess (const CTYPE r, const CTYPE time, const CTYPE freq, const CTYPE start){
	return ( r < freq/sec*DT );
}

__device__ char PeriodicFiring (const float r, const CTYPE time, const CTYPE freq, const CTYPE start){
	return ( ((int)((time - start - DT)*freq/sec)) != ((int)((time - start)*freq/sec)) );
}

__device__ CTYPE SinusoidalOscillation (const float max, const float mean, const float cycle, const float shift, const CTYPE time){
    return max/2 * (1 - cosf( 2*M_PI*cycle*(time/sec) + shift ));
}



{noise_func}

__host__ void InitInputStimulation( InputFunctionsStruct *List, Neuron *host_neurons, int *NeuronTypeID){
	FILE *fp;
	char str[256] = {'\0'};
	unsigned int *host_Id_list;
	int i;

    {stim_init}

    return;
}

{func_pointer}

__global__ void InputStimulation( const int n, char *spike,
                                  curandStatePhilox4_32_10_t *state,
                                  const int num, const int base_id,
                                  const unsigned int *IdList,
                                  const int target_row, const int total_nn,
	       			  int func_id){
	// cindicesはここでは使わない pallot neuron
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	if( i < num ){
		//Glomのnoise発火
		int global_id = base_id + IdList[i];
		float r[STEP_MAX];
		float4 tmp_r;

		for(int step = 0; step < STEP_MAX; step++){
			switch(step%4){
				case 0:
					tmp_r = curand_uniform4(&state[i]);
					r[step] = tmp_r.x;
					break;
				case 1: r[step] = tmp_r.y; break;
				case 2: r[step] = tmp_r.z; break;
				case 3: r[step] = tmp_r.w; break;
			}
		}

		char spike_ = 0;
		for(int step = 0; step < STEP_MAX;step++){
			spike_ += d_pInputFunctions[func_id](r[step], (CTYPE)n +((CTYPE)step)*DT);
		}
		spike[target_row + global_id] = (spike_)? 1 : 0;
	}
	return;
}
