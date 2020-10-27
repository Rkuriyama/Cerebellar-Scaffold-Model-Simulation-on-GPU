#include "../../data/parameters.h"
#include "../../data/size.h"
#include "init_params.h"
#include "option.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

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
		
		// val[i] に現状距離が入っているので、weightに置き換える.
		val[i] = weight;

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
void set_neuron_params(Neuron *n,enum NeuronType type,const char* filename, int num, int base_id, CTYPE Cm, CTYPE tau_m, CTYPE El, CTYPE dt_ref, CTYPE Ie, CTYPE Vr, CTYPE Vth, CTYPE tau_exc, CTYPE tau_inh, CTYPE gL ){
	n[type].type = type;
	strcpy(n[type].filename, filename);
	n[type].num = num;
	n[type].base_id = base_id;
	n[type].Cm = Cm;
	n[type].tau_m = tau_m;
	n[type].El = El;
	n[type].dt_ref = dt_ref;
	n[type].Ie = Ie;
	n[type].Vr = Vr;
	n[type].Vth = Vth;
	n[type].tau_exc = tau_exc;
	n[type].tau_inh = tau_inh;
	n[type].gL = gL;
	return;
}
void set_connectivity_params(Connectivity *c, enum ConnectionType type,const char*filename, int preNum, int postNum, enum NeuronType preType, enum NeuronType postType, CTYPE initial_weight, CTYPE delay, int UseParallelReduction  ){
	c[type].type = type;
	c[type].preNum = preNum;
	c[type].postNum = postNum;
	c[type].preType = preType;
	c[type].postType = postType;
	c[type].initial_weight = initial_weight;
	c[type].delay = delay;
	c[type].max_conv = LoadConnectivityFile(filename,&c[type].host_rptr, &c[type].rptr, &c[type].cindices, &c[type].val,initial_weight, preNum, postNum );
	c[type].pr = (UseParallelReduction);


	return;
}

int set_base_id(Neuron *Neurons){
	int base = 0;
	for(int i = 0;i < TotalNumOfCellTypes;i++){
		Neurons[i].base_id = base;
		base += Neurons[i].num;
	}
	return base;
}

__global__ void InitParams( CTYPE *u, CTYPE *g_exc, CTYPE *dg_exc, CTYPE *g_inh, CTYPE *dg_inh, int *refractory_time_left, char *spike , Neuron *Neurons ,char *type, const int total_nn){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if( i < total_nn){
		u[i] = Neurons[type[i]].Vr + (Neurons[type[i]].Vth - Neurons[type[i]].Vr)*u[i];
		g_exc[i] = 0.f;
		dg_exc[i] = 0.f;
		g_inh[i] = 0.f;
		dg_inh[i] = 0.f;
		refractory_time_left[i] = 0;
		spike[i] = 0;
	}
};


void init_neurons_params( Neuron *Neurons){

	set_neuron_params(
		Neurons,
		granule_cell,
		"granule_cell.dat",	//filename
		88158, 			//num
		0,			//base_id
		3.0,			//C_m
		2,			//tau_m
		-74.0,			//E_L
		1.5,			//dt_ref
		0.0,			//I_e
		-84.0,			//Vr
		-42.0,			//Vth
		0.5,			//tau_exc
		10.0,			//tau_inh
		1.5			//g_L
	);

	set_neuron_params(
		Neurons,
		glomerulus,
		"glomerulus.dat",
		7070,
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
		0
	);

	set_neuron_params(
		Neurons,
		purkinje_cell,
		"purkinje_cell.dat",
		69,
		0,
		620.0,
		88.,
		-62.0,
		0.8,
		600.0,
		-72.0,
		-47.0,
		0.5,
		1.6,
		7.0
	);

	set_neuron_params(
		Neurons,
		golgi_cell,
		"golgi_cell.dat",
		219,
		0,
		(76.),
		(21.),
		(-65.),
		(2.),
		(36.8),
		(-75.),
		(-55.),
		(0.5),
		(15.),
		(3.6)
	);

	set_neuron_params(
		Neurons,
		stellate_cell,
		"stellate_cell.dat",
		603,
		1280,
		(14.6),
		(14.6),
		(-68.),
		(1.6),
		(15.6),
		(-78.),
		(-53.),
		(0.64),
		(2.),
		(1.0)
	);

	set_neuron_params(
		Neurons,
		basket_cell,
		"basket_cell.dat",
		603,
		891,
		(14.6),
		(14.6),
		(-68.),
		(1.6),
		(15.6),
		(-78.),
		(-53.),
		(0.64),
		(2.),
		(1.0)
	);

	set_neuron_params(
		Neurons,
		dcn_cell,
		"dcn_cell.dat",
		12,
		11,
		(89.),
		(57.),
		(-59.),
		(3.7),
		(43.0),
		(-69.),
		(-48.),
		(0.5) ,
		(25.6),
		(1.56)
	);
}

void init_connectivity_params(Connectivity *connectivities){

	set_connectivity_params(
		connectivities,
		glomerulus_to_granule,
		"glomerulus_to_granule.dat",
		7070,
		88158,
		glomerulus,
		granule_cell,
		9.0,
		4.0,
		0
	);

	set_connectivity_params(
		connectivities,
		glomerulus_to_golgi,
		"glomerulus_to_golgi.dat",
		7070,
		219,
		glomerulus,
		golgi_cell,
		2.0,
		4.0,
		0
	);

	set_connectivity_params(
		connectivities,
		ascending_axon_to_golgi,
		"ascending_axon_to_golgi.dat",
		88158,
		219,
		granule_cell,
		golgi_cell,
		20.0,
		2.0,
		0
	);

	set_connectivity_params(
		connectivities,
		parallel_fiber_to_golgi,
		"parallel_fiber_to_golgi.dat",
		88158,
		219,
		granule_cell,
		golgi_cell,
		0.4,
		5.0,
		0
	);

	set_connectivity_params(
		connectivities,
		golgi_to_granule,
		"golgi_to_granule.dat",
		219,
		88158,
		golgi_cell,
		granule_cell,
		-5.0,
		2.0,
		0
	);

	set_connectivity_params(
		connectivities,
		ascending_axon_to_purkinje,
		"ascending_axon_to_purkinje.dat",
		88158,
		69,
		granule_cell,
		purkinje_cell,
		75.0,
		2.0,
		0
	);

	set_connectivity_params(
		connectivities,
		parallel_fiber_to_purkinje,
		"parallel_fiber_to_purkinje.dat",
		88158,
		69,
		granule_cell,
		purkinje_cell,
		0.02,
		5.0,
		1
	);

	set_connectivity_params(
		connectivities,
		parallel_fiber_to_basket,
		"parallel_fiber_to_basket.dat",
		88158,
		603,
		granule_cell,
		basket_cell,
		0.2,
		5.0,
		0
	);

	set_connectivity_params(
		connectivities,
		parallel_fiber_to_stellate,
		"parallel_fiber_to_stellate.dat",
		88158,
		603,
		granule_cell,
		stellate_cell,
		0.2,
		5.0,
		0
	);

	set_connectivity_params(
		connectivities,
		basket_to_purkinje,
		"basket_to_purkinje.dat",
		603,
		69,
		basket_cell,
		purkinje_cell,
		-9.0,
		4.0,
		0
	);

	set_connectivity_params(
		connectivities,
		stellate_to_purkinje,
		"stellate_to_purkinje.dat",
		603,
		69,
		stellate_cell,
		purkinje_cell,
		-8.5,
		5.0,
		0
	);

	set_connectivity_params(
		connectivities,
		stellate_to_stellate,
		"stellate_to_stellate.dat",
		603,
		603,
		stellate_cell,
		stellate_cell,
		-2.0,
		1.0,
		0
	);

	set_connectivity_params(
		connectivities,
		basket_to_basket,
		"basket_to_basket.dat",
		603,
		603,
		basket_cell,
		basket_cell,
		-2.5,
		1.0,
		0
	);

	set_connectivity_params(
		connectivities,
		golgi_to_golgi,
		"golgi_to_golgi.dat",
		219,
		219,
		golgi_cell,
		golgi_cell,
		-8.0,
		1.0,
		0
	);

	set_connectivity_params(
		connectivities,
		purkinje_to_dcn,
		"purkinje_to_dcn.dat",
		69,
		12,
		purkinje_cell,
		dcn_cell,
		-0.0075,
		4.0,
		0
	);

	set_connectivity_params(
		connectivities,
		glomerulus_to_dcn,
		"glomerulus_to_dcn.dat",
		7070,
		12,
		glomerulus,
		dcn_cell,
		0.006,
		4.0,
		0
	);

}

