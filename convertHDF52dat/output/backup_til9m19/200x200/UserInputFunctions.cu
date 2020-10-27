#include "UserInputFunctions.h"

__device__ int PoissonProcess (const CTYPE r, const CTYPE time, const CTYPE freq, const CTYPE start){
	return ( r < freq/sec*DT );
}

__device__ int PeriodicFiring (const float r, const CTYPE time, const CTYPE freq, const CTYPE start){
	CTYPE t = time - start;
	CTYPE T = sec/freq;
	return ( ((int)((t-DT)/T)) != ((int)(t/T)) );
}





__device__ int Background_noise(const float r, const CTYPE time){
	int flag = 0;
		flag = PoissonProcess( r, time,1, -1 );
	return flag;
}

__device__ int Stim1(const float r, const CTYPE time){
	int flag = 0;
	if(300 <= time && time < 350)		flag = PeriodicFiring( r, time,150, 300 );
	return flag;
}

__host__ void InitInputStimulation( InputFunctionsStruct *List, Neuron *host_neurons){
	FILE *fp;
	char str[256] = {'\0'};
	unsigned int *host_Id_list;
	int i;


	if ( (fp = fopen( "Background_noise.dat" , "r")) == NULL){
		fprintf(stderr, "cannot open file: Background_noise.dat\n");
		exit(1);
	}

	List[0].type = Glomerulus;
	List[0].base_id = host_neurons[Glomerulus].base_id;
	List[0].num = 1804;
	List[0].func_id = 0;
	i=0;
	host_Id_list = (unsigned int*)malloc(sizeof(unsigned int)*List[0].num);
	cudaMalloc( &(List[0].IdList), sizeof(unsigned int)*List[0].num );
	while( fgets(str, 256, fp) != NULL ){
		sscanf(str, "%u", &host_Id_list[i]);
		i++;
	}
	cudaMemcpy( List[0].IdList, host_Id_list, sizeof(unsigned int)*List[0].num, cudaMemcpyHostToDevice);
	fclose(fp);
	free(host_Id_list);
	if ( (fp = fopen( "Stim1.dat" , "r")) == NULL){
		fprintf(stderr, "cannot open file: Stim1.dat\n");
		exit(1);
	}

	List[1].type = Glomerulus;
	List[1].base_id = host_neurons[Glomerulus].base_id;
	List[1].num = 813;
	List[1].func_id = 1;
	i=0;
	host_Id_list = (unsigned int*)malloc(sizeof(unsigned int)*List[1].num);
	cudaMalloc( &(List[1].IdList), sizeof(unsigned int)*List[1].num );
	while( fgets(str, 256, fp) != NULL ){
		sscanf(str, "%u", &host_Id_list[i]);
		i++;
	}
	cudaMemcpy( List[1].IdList, host_Id_list, sizeof(unsigned int)*List[1].num, cudaMemcpyHostToDevice);
	fclose(fp);
	free(host_Id_list);

return;
}
__device__ pointFunction_t d_pInputFunctions[] = {Background_noise,Stim1};

__global__ void InputStimulation( const int n, int *spike,
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

		int spike_ = 0;
		for(int step = 0; step < STEP_MAX;step++){
			spike_ += d_pInputFunctions[func_id](r[step], (CTYPE)n +((CTYPE)step)*DT);
		}
		spike[target_row*total_nn + global_id] = spike_;
	}
	return;
}
