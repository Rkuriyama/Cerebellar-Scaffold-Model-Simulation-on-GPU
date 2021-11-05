#include "UserInputFunctions.h"
#include <vector>

__device__ char PoissonProcess (const CTYPE r, const CTYPE time, const CTYPE freq, const CTYPE start){
	return ( r < freq/sec*DT );
}

__device__ char PeriodicFiring (const float r, const CTYPE time, const CTYPE freq, const CTYPE start){
	return ( ((int)((time - start - DT)*freq/sec)) != ((int)((time - start)*freq/sec)) );
}

__device__ CTYPE SinusoidalOscillation (const float max, const float mean, const float osci, const float shift, const CTYPE time){
    return max/2 * (1 - cosf( 2*M_PI*(time/sec)/osci + shift ));
}




__device__ char background_noise(const float r, const CTYPE time){
	char flag = 0;
	flag = ( (0.0 <= time ) && PoissonProcess( r, time, 1.0, 0.0 ) );
	return (flag)?1:0;
}

__device__ char tone_stim(const float r, const CTYPE time){
	char flag = 0;
	flag = ((300.0 <= time) && (time < 351) && PeriodicFiring( r, time, 140.0, 292.8 ) ) || PoissonProcess(r, time, 1.0, 0.0);
	return (flag)?1:0;
}



__device__ char puff_stim(const float r, const CTYPE time){
	char flag = 0;
    CTYPE fleq;
    fleq = SinusoidalOscillation( 3.f, 1.5f, 2, 0, time );
	flag = ((0.0 <= time ) && PoissonProcess( r, time, fleq, 0 ) );
	return (flag)?1:0;
}

__host__ void setInputStim( InputFunctionsStruct *List,  Neuron*host_neurons, int *NeuronTypeID, enum NeuronType type, const std::string file_name, int func_id ){
    FILE *fp;

    if( (fp = fopen( file_name.c_str(), "r")) == NULL){
		fprintf(stderr, "cannot open file: %s\n", file_name.c_str());
		exit(1);
    }


    std::vector<unsigned int> v;
    char str[256]={'\0'};

    while( fgets(str, 256, fp) != NULL ){
        unsigned int id;
		sscanf(str, "%u", &id);
        v.push_back(id);
    }
    
    List->type = NeuronTypeID[type];
    List->base_id = host_neurons[ NeuronTypeID[type] ].base_id;
    List->num = v.size();
    List->IdList = (unsigned int *)malloc( sizeof(unsigned int)*List->num );
    memcpy( List->IdList, &(v[0]), sizeof(unsigned int)*List->num );

    List->func_id = func_id;

    fclose(fp);

    return;
}

__host__ void InitInputStimulation( InputFunctionsStruct *List, Neuron *host_neurons, int *NeuronTypeID, std::string dir){
    int stim_id = 0;

    setInputStim( &(List[stim_id]), host_neurons, NeuronTypeID, glomerulus, dir+"background_noise.dat", stim_id);
    stim_id++;

    setInputStim( &(List[stim_id]), host_neurons, NeuronTypeID, glomerulus, dir+"tone_stim.dat", stim_id);
    stim_id++;

    return;
}
__device__ pointFunction_t d_pInputFunctions[] = {background_noise, tone_stim};
//__device__ pointFunction_t d_pInputFunctions[] = {tone_stim, tone_stim, puff_stim};

__global__ void InputStimulation( const int n, char *spike,
                                  curandStatePhilox4_32_10_t *state,
                                  const float freq,
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
            //spike_ += PoissonProcess(r[step], (CTYPE)n +((CTYPE)step)*DT, freq, 0);
			spike_ += d_pInputFunctions[func_id](r[step], (CTYPE)n +((CTYPE)step)*DT);
		}
		spike[target_row + global_id] = (spike_)? 1 : 0;
	}
	return;
}
