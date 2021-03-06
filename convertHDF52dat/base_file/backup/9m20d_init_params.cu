#include "init_params.h"
#include "option.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>

FILE *wout = fopen("weight.dat","w");

int LoadConnectivityFile(const char *file_name,unsigned int **host_rptr, unsigned int **d_rptr, unsigned int **d_cindices, CTYPE **d_val, CTYPE weight ,int PreSN_num,int PostSN_num){
	PreSN_num = (PreSN_num < 1)? 1: PreSN_num;
	PostSN_num = (PostSN_num < 1)? 1000: PostSN_num;


	FILE *fp;
	if((fp = fopen( file_name ,"r")) == NULL ){
		fprintf(stderr, "can't open file :  %s\n",file_name);
		exit(1);
	}

	weight = (weight < 0)?(-1)*weight: weight;
	int num_of_data = PostSN_num*10;
	unsigned int *rptr = NULL;
	unsigned int *cindices = NULL;
	CTYPE *val =NULL;

	int max_conv = 0;


	rptr = (unsigned int *)malloc( (PostSN_num+1)*sizeof(unsigned int) );
	cindices = (unsigned int *)malloc( num_of_data*sizeof(unsigned int) );
	val = (CTYPE *)malloc( num_of_data*sizeof(CTYPE) );

	if(rptr == NULL || cindices == NULL || val == NULL){
		fprintf(stderr,"malloc error\n");
		exit(1);
	}

	char str[256] = {'\0'};
	int i = 0;
	int prev_post_id = 0;
	int post_id;
	rptr[0] = 0;
	while( fgets(str, 256, fp) != NULL ){

		//sscanf(str, "%d %d %f", &cindices[i], &post_id, &val[i] );
		sscanf(str, "%d %d", &cindices[i], &post_id );

        if( cindices[i] >= PreSN_num || post_id >= PostSN_num ){
            fprintf(stderr, "connectivity file corupsed. loaded id exceed the limit number.\n" );
            fprintf(stderr, "load(%d,%d), limit(%d,%d)\n", cindices[i], post_id, PreSN_num, PostSN_num );
        }
		
		val[i] = 1;

		// 本来はpost_id > prev_post_id (ソート済み前提)
		if(post_id != prev_post_id) {
			for(int j=prev_post_id+1;j<post_id+1;j++) rptr[j] = i;
			prev_post_id = post_id;
		}
		i++;

		// 拡張
		if(i > num_of_data-1){
			float avg = (post_id != 0)?(float)i/(float)(post_id):i;
			num_of_data = (int)(avg*PostSN_num);

			//fprintf(stderr, "realloc phase %d to %d\n", i, num_of_data);

			unsigned int *i_tmp=NULL;
			CTYPE *c_tmp=NULL;
			if(( i_tmp = (unsigned int *)realloc(cindices, num_of_data*sizeof(unsigned int))) == NULL){
				free(cindices);
				exit(1);
			}else{
				if(cindices != i_tmp){
					cindices = i_tmp;
				}
			}

			if(( c_tmp = (CTYPE *)realloc(val, num_of_data*sizeof(CTYPE) )) == NULL){
				free(val);
				exit(1);
			}else{
				if(val != c_tmp){
					val = c_tmp;
				}
			}

		}
	}

	if(num_of_data != i){
		num_of_data = i;
		for(int j = post_id+1; j < PostSN_num+1;j++){
			rptr[j] = num_of_data;
		}

		// 縮小
		//fprintf(stderr, "realloc phase :to %d\n", num_of_data);
		unsigned int *i_tmp = NULL;
		CTYPE *c_tmp = NULL;
		if(( i_tmp = (unsigned int *)realloc(cindices, num_of_data*sizeof(unsigned int))) == NULL){
			fprintf(stderr, "can't realloc memory in roading phase: %s\n", file_name);
			free(cindices);
			exit(1);
		}else{
			if(cindices != i_tmp)cindices = i_tmp;
		}

		if(( c_tmp = (CTYPE *)realloc(val, num_of_data*sizeof(CTYPE))) == NULL){
			fprintf(stderr, "can't realloc memory in roading phase: %s\n", file_name);
			free(val);
			exit(1);
		}else{
			if(val != c_tmp) val = c_tmp;
		}
	}

	for(int i = 0; i < PostSN_num; i++) max_conv = (max_conv < rptr[i+1]-rptr[i])?rptr[i+1]-rptr[i]:max_conv;

	CUDA_SAFE_CALL( cudaMalloc( d_rptr, sizeof(unsigned int)*(PostSN_num+1)) );
	CUDA_SAFE_CALL( cudaMalloc( d_cindices, sizeof(unsigned int)*num_of_data) );
	CUDA_SAFE_CALL( cudaMalloc( d_val, sizeof(CTYPE)*num_of_data));

	CUDA_SAFE_CALL( cudaMemcpy( *d_rptr, rptr, sizeof(unsigned int)*(PostSN_num+1), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy( *d_cindices, cindices, sizeof(unsigned int)*num_of_data, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy( *d_val, val, sizeof(CTYPE)*num_of_data, cudaMemcpyHostToDevice));

	*host_rptr = rptr;

	fclose(fp);
	//free(rptr);
	free(cindices);
	free(val);

	return max_conv;
}

int set_neuron_params(Neuron *n,enum NeuronType type,const char* filename, char duplicate, int num, CTYPE Cm, CTYPE tau_m, CTYPE El, CTYPE dt_ref, CTYPE Ie, CTYPE Vr, CTYPE Vth, CTYPE tau_exc, CTYPE tau_inh, CTYPE gL, enum DeviceType device_type  ){
	static int CPU = 0;
	static int GPU = 0;
	int target = 0;
	if( device_type == NORMAL ){
		target = GPU;
		GPU++;
	} else {
	       	target = (int)TotalNumOfCellTypes - 1 - CPU;
		CPU++;
	}
	n[target].type = target;
	strcpy(n[target].filename, filename);
	n[target].num = num;
	n[target].Cm = Cm;
	n[target].tau_m = tau_m;
	n[target].El = El;
	n[target].dt_ref = dt_ref;
	n[target].Ie = Ie;
	n[target].Vr = Vr;
	n[target].Vth = Vth;
	n[target].tau_exc = tau_exc;
	n[target].tau_inh = tau_inh;
	n[target].gL = gL;
	n[target].duplicate = duplicate;
	n[target].dev_type = device_type;
	return target;
}
int set_connectivity_params(Connectivity *c, Neuron *neurons, enum ConnectionType type,const char*filename, int preNum, int postNum, int preType, int postType, CTYPE initial_weight, CTYPE delay, int UseParallelReduction  ){
	static int GPU = 0;
	static int CPU = 0;
	int target = 0;

	if ( neurons[postType].dev_type == OUTPUT ){
		target = (int)TotalNumOfConnectivityTypes - 1 - CPU;
		CPU++;
	}else{
		target = GPU;
		GPU++;
	}

	c[target].type = type;
	c[target].preNum = preNum;
	c[target].postNum = postNum;
	c[target].preType = preType;
	c[target].postType = postType;
	c[target].initial_weight = initial_weight;
	c[target].delay = delay;
	c[target].max_conv = LoadConnectivityFile(filename,&c[target].host_rptr, &c[target].rptr, &c[target].cindices, &c[target].val,initial_weight, preNum, postNum );
	c[target].pr = (UseParallelReduction);


	return target;
}

int set_base_id(Neuron *Neurons){
	int base = 0;
	for(int i = 0;i < TotalNumOfCellTypes;i++){
		Neurons[i].base_id = base;
		base += Neurons[i].num;
	}
	return base;
}

__global__ void InitParams( CTYPE *u, CTYPE *g_exc, CTYPE *dg_exc, CTYPE *g_inh, CTYPE *dg_inh, int *refractory_time_left, char *spike , Neuron *Neurons ,char *type, curandStatePhilox4_32_10_t *state, const int total_nn){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if( i < total_nn){
		u[i] = Neurons[type[i]].Vr + (Neurons[type[i]].Vth - Neurons[type[i]].Vr)*curand_uniform(&state[i]);
		g_exc[i] = 0.f;
		dg_exc[i] = 0.f;
		g_inh[i] = 0.f;
		dg_inh[i] = 0.f;
		refractory_time_left[i] = 0;
		spike[i] = 0;
	}
};

__host__ void Host_InitParams( CTYPE *u, CTYPE *g_exc, CTYPE *dg_exc, CTYPE *g_inh, CTYPE *dg_inh, int *refractory_time_left, char *spike , Neuron *Neurons ,char *type, const int total_nn){
	srand( (unsigned)time(NULL) );
	for(int i = 0 ; i < total_nn; i++){
		u[i] = Neurons[type[i]].Vr + (Neurons[type[i]].Vth - Neurons[type[i]].Vr)*(  (rand() + 1.0)/(2.0 + RAND_MAX)  );
		g_exc[i] = 0.f;
		dg_exc[i] = 0.f;
		g_inh[i] = 0.f;
		dg_inh[i] = 0.f;
		refractory_time_left[i] = 0;
		spike[i] = 0;
	}
};


void init_neurons_params( Neuron *Neurons, int *NeuronTypeID){
    {set_neuron_params}
}

void init_connectivity_params(Connectivity *connectivities, Neuron *neurons, int *NeuronTypeID, int *ConnectivityTypeID){
    {set_connectivity_params}
}

