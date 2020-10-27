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

int set_neuron_params(Neuron *n,enum NeuronType type,const char* filename, char duplicate, int num, int base_id, CTYPE Cm, CTYPE tau_m, CTYPE El, CTYPE dt_ref, CTYPE Ie, CTYPE Vr, CTYPE Vth, CTYPE tau_exc, CTYPE tau_inh, CTYPE gL, enum DeviceType device_type  ){
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
	n[target].base_id = base_id;
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

	int	granule_cell_num = 91473,
		glomerulus_num = 7193,
		purkinje_cell_num = 72,
		golgi_cell_num = 216,
		stellate_cell_num = 598,
		basket_cell_num = 592,
		dcn_cell_num = 14,
		dcn_interneuron_num = 14,
		io_cell_num = 11;

	NeuronTypeID[granule_cell] = set_neuron_params(
		Neurons,
		granule_cell,
		"granule_cell.dat",
		0,
		granule_cell_num,
		9069,
		7.0,
		24.13793103448276,
		-62.0,
		1.5,
		0.0,
		-70.0,
		-41.0,
		5.8,
		13.61,
		0.29,
		NORMAL
	);

	NeuronTypeID[glomerulus] = set_neuron_params(
		Neurons,
		glomerulus,
		"glomerulus.dat",
		0,
		glomerulus_num,
		1876,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		INPUT
	);

	NeuronTypeID[purkinje_cell] = set_neuron_params(
		Neurons,
		purkinje_cell,
		"purkinje_cell.dat",
		0,
		purkinje_cell_num,
		39,
		334.0,
		47.04225352112676,
		-59.0,
		0.5,
		800.0,
		-69.0,
		-43.0,
		1.1,
		2.8,
		7.1,
		NORMAL
	);

	NeuronTypeID[golgi_cell] = set_neuron_params(
		Neurons,
		golgi_cell,
		"golgi_cell.dat",
		0,
		golgi_cell_num,
		111,
		145.0,
		43.939393939393945,
		-62.0,
		2.0,
		36.75,
		-75.0,
		-55.0,
		0.23,
		10.0,
		3.3,
		NORMAL
	);

	NeuronTypeID[stellate_cell] = set_neuron_params(
		Neurons,
		stellate_cell,
		"stellate_cell.dat",
		0,
		stellate_cell_num,
		686,
		14.6,
		9.125,
		-68.0,
		1.59,
		24.05,
		-78.0,
		-53.0,
		0.64,
		2.0,
		1.6,
		NORMAL
	);

	NeuronTypeID[basket_cell] = set_neuron_params(
		Neurons,
		basket_cell,
		"basket_cell.dat",
		0,
		basket_cell_num,
		1284,
		14.6,
		9.125,
		-68.0,
		1.59,
		24.05,
		-78.0,
		-53.0,
		0.64,
		2.0,
		1.6,
		NORMAL
	);

	NeuronTypeID[dcn_cell] = set_neuron_params(
		Neurons,
		dcn_cell,
		"dcn_cell.dat",
		0,
		dcn_cell_num,
		11,
		142.0,
		33.02325581395349,
		-45.0,
		0.8,
		180.0,
		-55.0,
		-36.0,
		1.0,
		0.7,
		4.3,
		OUTPUT
	);

	NeuronTypeID[dcn_interneuron] = set_neuron_params(
		Neurons,
		dcn_interneuron,
		"dcn_interneuron.dat",
		0,
		dcn_interneuron_num,
		25,
		56.0,
		56.0,
		-40.0,
		0.8,
		7.0,
		-55.0,
		-39.0,
		3.64,
		1.14,
		1.0,
		OUTPUT
	);

	NeuronTypeID[io_cell] = set_neuron_params(
		Neurons,
		io_cell,
		"io_cell.dat",
		0,
		io_cell_num,
		0,
		189.0,
		11.0011641443539,
		-45.0,
		1.0,
		0.0,
		-45.0,
		-35.0,
		1.0,
		60.0,
		17.18,
		INPUT
	);

}

void init_connectivity_params(Connectivity *connectivities, Neuron *neurons, int *NeuronTypeID, int *ConnectivityTypeID){

	int	granule_cell_num = 91473,
		glomerulus_num = 7193,
		purkinje_cell_num = 72,
		golgi_cell_num = 216,
		stellate_cell_num = 598,
		basket_cell_num = 592,
		dcn_cell_num = 14,
		dcn_interneuron_num = 14,
		io_cell_num = 11;

	ConnectivityTypeID[parallel_fiber_to_purkinje] = set_connectivity_params(
		connectivities,
		neurons,
		parallel_fiber_to_purkinje,
		"parallel_fiber_to_purkinje.dat",
		granule_cell_num,
		purkinje_cell_num,
		NeuronTypeID[granule_cell],
		NeuronTypeID[purkinje_cell],
		0.007,
		5.0,
		1
	);

	ConnectivityTypeID[parallel_fiber_to_golgi] = set_connectivity_params(
		connectivities,
		neurons,
		parallel_fiber_to_golgi,
		"parallel_fiber_to_golgi.dat",
		granule_cell_num,
		golgi_cell_num,
		NeuronTypeID[granule_cell],
		NeuronTypeID[golgi_cell],
		0.05,
		5.0,
		0
	);

	ConnectivityTypeID[parallel_fiber_to_stellate] = set_connectivity_params(
		connectivities,
		neurons,
		parallel_fiber_to_stellate,
		"parallel_fiber_to_stellate.dat",
		granule_cell_num,
		stellate_cell_num,
		NeuronTypeID[granule_cell],
		NeuronTypeID[stellate_cell],
		0.015,
		5.0,
		0
	);

	ConnectivityTypeID[parallel_fiber_to_basket] = set_connectivity_params(
		connectivities,
		neurons,
		parallel_fiber_to_basket,
		"parallel_fiber_to_basket.dat",
		granule_cell_num,
		basket_cell_num,
		NeuronTypeID[granule_cell],
		NeuronTypeID[basket_cell],
		0.015,
		5.0,
		0
	);

	ConnectivityTypeID[glomerulus_to_dcn] = set_connectivity_params(
		connectivities,
		neurons,
		glomerulus_to_dcn,
		"glomerulus_to_dcn.dat",
		glomerulus_num,
		dcn_cell_num,
		NeuronTypeID[glomerulus],
		NeuronTypeID[dcn_cell],
		0.05,
		3.0,
		0
	);

	ConnectivityTypeID[ascending_axon_to_golgi] = set_connectivity_params(
		connectivities,
		neurons,
		ascending_axon_to_golgi,
		"ascending_axon_to_golgi.dat",
		granule_cell_num,
		golgi_cell_num,
		NeuronTypeID[granule_cell],
		NeuronTypeID[golgi_cell],
		0.05,
		5.0,
		0
	);

	ConnectivityTypeID[ascending_axon_to_purkinje] = set_connectivity_params(
		connectivities,
		neurons,
		ascending_axon_to_purkinje,
		"ascending_axon_to_purkinje.dat",
		granule_cell_num,
		purkinje_cell_num,
		NeuronTypeID[granule_cell],
		NeuronTypeID[purkinje_cell],
		0.7,
		2.0,
		0
	);

	ConnectivityTypeID[glomerulus_to_golgi] = set_connectivity_params(
		connectivities,
		neurons,
		glomerulus_to_golgi,
		"glomerulus_to_golgi.dat",
		glomerulus_num,
		golgi_cell_num,
		NeuronTypeID[glomerulus],
		NeuronTypeID[golgi_cell],
		1.5,
		4.0,
		0
	);

	ConnectivityTypeID[golgi_to_golgi] = set_connectivity_params(
		connectivities,
		neurons,
		golgi_to_golgi,
		"golgi_to_golgi.dat",
		golgi_cell_num,
		golgi_cell_num,
		NeuronTypeID[golgi_cell],
		NeuronTypeID[golgi_cell],
		-0.3,
		1.0,
		0
	);

	ConnectivityTypeID[purkinje_to_dcn] = set_connectivity_params(
		connectivities,
		neurons,
		purkinje_to_dcn,
		"purkinje_to_dcn.dat",
		purkinje_cell_num,
		dcn_cell_num,
		NeuronTypeID[purkinje_cell],
		NeuronTypeID[dcn_cell],
		-0.4,
		4.0,
		0
	);

	ConnectivityTypeID[basket_to_purkinje] = set_connectivity_params(
		connectivities,
		neurons,
		basket_to_purkinje,
		"basket_to_purkinje.dat",
		basket_cell_num,
		purkinje_cell_num,
		NeuronTypeID[basket_cell],
		NeuronTypeID[purkinje_cell],
		-0.3,
		4.0,
		0
	);

	ConnectivityTypeID[stellate_to_purkinje] = set_connectivity_params(
		connectivities,
		neurons,
		stellate_to_purkinje,
		"stellate_to_purkinje.dat",
		stellate_cell_num,
		purkinje_cell_num,
		NeuronTypeID[stellate_cell],
		NeuronTypeID[purkinje_cell],
		-0.3,
		5.0,
		0
	);

	ConnectivityTypeID[io_to_dcn] = set_connectivity_params(
		connectivities,
		neurons,
		io_to_dcn,
		"io_to_dcn.dat",
		io_cell_num,
		dcn_cell_num,
		NeuronTypeID[io_cell],
		NeuronTypeID[dcn_cell],
		0.1,
		4.0,
		0
	);

	ConnectivityTypeID[io_to_dcn_interneuron] = set_connectivity_params(
		connectivities,
		neurons,
		io_to_dcn_interneuron,
		"io_to_dcn_interneuron.dat",
		io_cell_num,
		dcn_interneuron_num,
		NeuronTypeID[io_cell],
		NeuronTypeID[dcn_interneuron],
		0.2,
		5.0,
		0
	);

	ConnectivityTypeID[stellate_to_stellate] = set_connectivity_params(
		connectivities,
		neurons,
		stellate_to_stellate,
		"stellate_to_stellate.dat",
		stellate_cell_num,
		stellate_cell_num,
		NeuronTypeID[stellate_cell],
		NeuronTypeID[stellate_cell],
		-0.2,
		1.0,
		0
	);

	ConnectivityTypeID[basket_to_basket] = set_connectivity_params(
		connectivities,
		neurons,
		basket_to_basket,
		"basket_to_basket.dat",
		basket_cell_num,
		basket_cell_num,
		NeuronTypeID[basket_cell],
		NeuronTypeID[basket_cell],
		-0.2,
		1.0,
		0
	);

	ConnectivityTypeID[glomerulus_to_granule] = set_connectivity_params(
		connectivities,
		neurons,
		glomerulus_to_granule,
		"glomerulus_to_granule.dat",
		glomerulus_num,
		granule_cell_num,
		NeuronTypeID[glomerulus],
		NeuronTypeID[granule_cell],
		0.15,
		4.0,
		0
	);

	ConnectivityTypeID[golgi_to_granule] = set_connectivity_params(
		connectivities,
		neurons,
		golgi_to_granule,
		"golgi_to_granule.dat",
		golgi_cell_num,
		granule_cell_num,
		NeuronTypeID[golgi_cell],
		NeuronTypeID[granule_cell],
		-0.6,
		2.0,
		0
	);

	ConnectivityTypeID[io_to_basket] = set_connectivity_params(
		connectivities,
		neurons,
		io_to_basket,
		"io_to_basket.dat",
		io_cell_num,
		basket_cell_num,
		NeuronTypeID[io_cell],
		NeuronTypeID[basket_cell],
		1.0,
		70.0,
		0
	);

	ConnectivityTypeID[io_to_stellate] = set_connectivity_params(
		connectivities,
		neurons,
		io_to_stellate,
		"io_to_stellate.dat",
		io_cell_num,
		stellate_cell_num,
		NeuronTypeID[io_cell],
		NeuronTypeID[stellate_cell],
		1.0,
		70.0,
		0
	);

	ConnectivityTypeID[io_to_purkinje] = set_connectivity_params(
		connectivities,
		neurons,
		io_to_purkinje,
		"io_to_purkinje.dat",
		io_cell_num,
		purkinje_cell_num,
		NeuronTypeID[io_cell],
		NeuronTypeID[purkinje_cell],
		350.0,
		4.0,
		0
	);

}

