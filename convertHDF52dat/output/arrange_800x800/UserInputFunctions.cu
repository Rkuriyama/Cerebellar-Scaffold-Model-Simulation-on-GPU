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



__device__ char background_noise(const float r, const CTYPE time){
	char flag = 0;
	flag = ((0.0 <= time) && PoissonProcess( r, time, 1.0, 0.0 ) );
	return (flag)?1:0;
}

__device__ char tone_stim(const float r, const CTYPE time){
	char flag = 0;
	flag = ((300.0 <= time && time < 350.0) && PoissonProcess( r, time, 150.0, 300.0 ) );
	return (flag)?1:0;
}

__device__ char puff_stim(const float r, const CTYPE time){
	char flag = 0;
	flag = ((340 <= time && time <= 350) && PeriodicFiring( r, time, 500.0, 340 ) );
	return (flag)?1:0;
}


__host__ void InitInputStimulation( InputFunctionsStruct *List, Neuron *host_neurons, int *NeuronTypeID){
	FILE *fp;
	char str[256] = {'\0'};
	unsigned int *host_Id_list;
	int i;

    	if ( (fp = fopen( "background_noise.dat" , "r")) == NULL){
		fprintf(stderr, "cannot open file: background_noise.dat\n");
		exit(1);
	}

	List[0].type = NeuronTypeID[glomerulus];
	List[0].base_id = host_neurons[NeuronTypeID[glomerulus]].base_id;
	List[0].num = 28798;
	List[0].func_id = 0;
	i=0;
	List[0].IdList = (unsigned int*)malloc(sizeof(unsigned int)*List[0].num);
	while( fgets(str, 256, fp) != NULL ){
		sscanf(str, "%u", &List[0].IdList[i]);
		i++;
	}
	fclose(fp);

//
//	if ( (fp = fopen( "tone_stim.dat" , "r")) == NULL){
//		fprintf(stderr, "cannot open file: tone_stim.dat\n");
//		exit(1);
//	}
//
//	List[1].type = NeuronTypeID[glomerulus];
//	List[1].base_id = host_neurons[NeuronTypeID[glomerulus]].base_id;
//	List[1].num = 18618;
//	List[1].func_id = 1;
//	i=0;
//	List[1].IdList = (unsigned int*)malloc(sizeof(unsigned int)*List[1].num);
//	while( fgets(str, 256, fp) != NULL ){
//		sscanf(str, "%u", &List[1].IdList[i]);
//		i++;
//	}
//	fclose(fp);
//
//
//	if ( (fp = fopen( "puff_stim.dat" , "r")) == NULL){
//		fprintf(stderr, "cannot open file: puff_stim.dat\n");
//		exit(1);
//	}
//
//	List[2].type = NeuronTypeID[io_cell];
//	List[2].base_id = host_neurons[NeuronTypeID[io_cell]].base_id;
//	List[2].num = 47;
//	List[2].func_id = 2;
//	i=0;
//	List[2].IdList = (unsigned int*)malloc(sizeof(unsigned int)*List[2].num);
//	while( fgets(str, 256, fp) != NULL ){
//		sscanf(str, "%u", &List[2].IdList[i]);
//		i++;
//	}
//	fclose(fp);




    return;
}

__device__ pointFunction_t d_pInputFunctions[] = {background_noise,tone_stim,puff_stim};



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
