#include "init_params.h"
#include "option.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <assert.h>
#include <curand_kernel.h>

FILE *wout = fopen("weight.dat","w");

void LoadConnectivityFile_ELL(const char *file_name, unsigned int max_conv, int **d_cindices, CTYPE **d_val, CTYPE weight ,int PreSN_num,int PostSN_num){
    //max_convergence = #entities_per_row

    int *h_cindices;
    CTYPE *h_val;
    int *h_col;
    
    h_cindices = (int *)malloc(sizeof(int)*max_conv*PostSN_num);
    h_val = (CTYPE *)malloc(sizeof(CTYPE)*max_conv*PostSN_num);
    h_col = (int *)malloc(sizeof(int)*PostSN_num);

    for(int i = 0; i < max_conv*PostSN_num; i++){
        h_cindices[i] = -1;
        h_val[i] = 0;
        if(i < PostSN_num){
            h_col[i] = 0;
        }
    }

	FILE *fp;
	if((fp = fopen( file_name ,"r")) == NULL ){
		fprintf(stderr, "can't open file :  %s\n",file_name);
		exit(1);
	}

	char str[256] = {'\0'};
    while( fgets(str, 256, fp) != NULL){
        int pre_id, post_id;
		//sscanf(str, "%d %d %f", &cindices[i], &post_id, &val[i] );
		sscanf(str, "%d %d", &pre_id, &post_id );
        if( h_col[post_id] < max_conv ){

            h_cindices[ post_id*max_conv + h_col[post_id] ] = pre_id;
            h_val[ post_id*max_conv + h_col[post_id] ] = 1;
            
            h_col[post_id]++;
        }else{
            fprintf(stderr, "exceed max_conv: post_id-%d, max_conv-%d\n", post_id, max_conv);
        }
    
    }

	CUDA_SAFE_CALL( cudaMalloc( d_cindices, sizeof(int)*max_conv*PostSN_num) );
	CUDA_SAFE_CALL( cudaMalloc( d_val, sizeof(CTYPE)*max_conv*PostSN_num));

	CUDA_SAFE_CALL( cudaMemcpy( *d_cindices, h_cindices, sizeof(int)*max_conv*PostSN_num, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy( *d_val, h_val, sizeof(CTYPE)*max_conv*PostSN_num, cudaMemcpyHostToDevice));

    fclose(fp);
    free(h_cindices);
    free(h_val);
    free(h_col);

    return;
}

__global__
void check_consistency( unsigned int *csr_rptr, unsigned int *csr_cindices, CTYPE *csr_val, int *ell_cindices, CTYPE *ell_val, unsigned int max_conv, int post_num ){
    int post_id = threadIdx.x + blockIdx.x*blockDim.x;
    if(post_id < post_num){
        int csr_start = csr_rptr[post_id];
        int csr_end = csr_rptr[post_id+1];
        int width = csr_end - csr_start;
        for(int idx = 0; idx < width; idx++){
            if( csr_cindices[csr_start + idx] != ell_cindices[max_conv*post_id + idx]  ){
                printf("consistency corrupted(cindices). post_id-%d, csr-%d, ell-%d\n", post_id, csr_cindices[csr_start + idx], ell_cindices[max_conv*post_id + idx] );
                assert(0);
            }else if( csr_val[csr_start + idx] != ell_val[max_conv*post_id + idx]  ){
                printf("consistency corrupted(val). post_id-%d, pre_id-%d-%d csr-%d, ell-%d\n", post_id, csr_cindices[csr_start + idx], ell_cindices[max_conv*post_id + idx], csr_val[csr_start + idx], ell_val[max_conv*post_id + idx] );
                assert(0);
            }
            
        }
        for(int idx = width; idx < max_conv; idx++){
            if(ell_cindices[idx + post_id*max_conv] != -1){
                printf("overloaded cindices at ELL. post_id-%d\n", post_id);
                assert(0);
            }
        }
    }
    return;
}

int GetMaxConv( const char *file_name, int PreSN_num, int PostSN_num ){
    //max_convergence = #entities_per_row

    int *h_col;
    int max_conv = 0;
    
    h_col = (int *)malloc(sizeof(int)*PostSN_num);

    for(int i = 0; i < PostSN_num; i++){
            h_col[i] = 0;
    }

	FILE *fp;
	if((fp = fopen( file_name ,"r")) == NULL ){
		fprintf(stderr, "can't open file :  %s\n",file_name);
		exit(1);
	}

	char str[256] = {'\0'};
    while( fgets(str, 256, fp) != NULL){
        int pre_id, post_id;
		//sscanf(str, "%d %d %f", &cindices[i], &post_id, &val[i] );
		sscanf(str, "%d %d", &pre_id, &post_id );
        if( pre_id < PreSN_num && post_id < PostSN_num ){ 
            h_col[post_id]++;
            max_conv = (max_conv > h_col[post_id])? max_conv: h_col[post_id];
        }
    
    }

    free(h_col);
    return max_conv;
}

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

int set_neuron_params(Neuron *n,enum NeuronType type,const char* filename, char duplicate, int num, CTYPE Cm, CTYPE El, CTYPE dt_ref, CTYPE Ie, CTYPE Vr, CTYPE Vth, CTYPE gL, enum DeviceType device_type, int c_num, ...){
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

    fprintf(stderr,"\t%s: %d %d\n", filename, target, type);

	n[target].type = target;
	strcpy(n[target].filename, filename);
	n[target].num = num;
	n[target].Cm = Cm;
	n[target].El = El;
	n[target].dt_ref = dt_ref;
	n[target].Ie = Ie;
	n[target].Vr = Vr;
	n[target].Vth = Vth;
	n[target].gL = gL;
	n[target].duplicate = duplicate;
	n[target].dev_type = device_type;
    n[target].c_num = c_num;

    if( c_num > 0 ){
        n[target].w = (CTYPE *)malloc(sizeof(CTYPE)*c_num);
        n[target].E = (CTYPE *)malloc(sizeof(CTYPE)*c_num);
        n[target].g_bar = (CTYPE *)malloc(sizeof(CTYPE)*c_num);
        n[target].tau = (CTYPE *)malloc(sizeof(CTYPE)*c_num);
    }

    va_list args;
    va_start(args, c_num);
    for(int c = 0; c < c_num; c++){
        n[target].w[c] = (CTYPE) va_arg(args, double);
        n[target].E[c] = (CTYPE) va_arg(args, double);
        n[target].g_bar[c] = (CTYPE) va_arg(args, double);
        n[target].tau[c] = (CTYPE) va_arg(args, double);

        fprintf(stderr, "ch-%d: %lf, %lf, %lf, %lf\n", c, n[target].w[c], n[target].E[c], n[target].g_bar[c], n[target].tau[c]);
    }
    va_end(args);

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

    fprintf(stderr,"\t%s: %d %d\n", filename, target, type);

	c[target].type = type;
	c[target].preNum = preNum;
	c[target].postNum = postNum;
	c[target].preType = preType;
	c[target].postType = postType;
	c[target].initial_weight = initial_weight;
	c[target].delay = delay;
    c[target].max_conv = GetMaxConv( filename, preNum, postNum );
	c[target].pr = (UseParallelReduction);

    LoadConnectivityFile_ELL( filename, c[target].max_conv, &c[target].ELL_cindices, &c[target].ELL_val, initial_weight, preNum, postNum );

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

__global__ void InitParams( CTYPE *u, CTYPE *dg_exc, CTYPE *dg_inh, int *refractory_time_left, char *spike , Neuron *Neurons ,char *type, curandStatePhilox4_32_10_t *state, const int total_nn){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if( i < total_nn){
		u[i] = Neurons[type[i]].Vr + (Neurons[type[i]].Vth - Neurons[type[i]].Vr)*curand_uniform(&state[i]);
		dg_exc[i] = 0.f;
		dg_inh[i] = 0.f;
		refractory_time_left[i] = 0;
		spike[i] = 0;
	}
};

__host__ void Host_InitParams( CTYPE *u, CTYPE *dg_exc, CTYPE *dg_inh, int *refractory_time_left, char *spike , Neuron *Neurons ,char *type, const int total_nn){
	srand( (unsigned)time(NULL) );
	for(int i = 0 ; i < total_nn; i++){
		u[i] = Neurons[type[i]].Vr + (Neurons[type[i]].Vth - Neurons[type[i]].Vr)*(  (rand() + 1.0)/(2.0 + RAND_MAX)  );
		dg_exc[i] = 0.f;
		dg_inh[i] = 0.f;
		refractory_time_left[i] = 0;
		spike[i] = 0;
	}
};


void init_neurons_params( Neuron *Neurons, int *NeuronTypeID){

	int	granule_cell_num = 88158,
		glomerulus_num = 7073,
		purkinje_cell_num = 69,
		golgi_cell_num = 219,
		stellate_cell_num = 603,
		basket_cell_num = 603,
		dcn_cell_num = 12,
        io_cell_num = 12;

	NeuronTypeID[granule_cell] = set_neuron_params(
		Neurons,
		granule_cell,
		"granule_cell.dat",
		0,
		granule_cell_num,
		3.1,        //Cm
		-58.0,      //E_L
		1.5,        //dt_ref
		0.0,        //Ie
		-82.0,      //Vr
		-35.0,      //Vth
		0.43,       //g_L
		NORMAL,
        2,
        1., 0., 0.18, 1.2, //exc
        1., -85., 0.028, 7.0 //inh
	);

	NeuronTypeID[glomerulus] = set_neuron_params(
		Neurons,
		glomerulus,
		"glomerulus.dat",
		0,
		glomerulus_num,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		INPUT,
        0

	);

	NeuronTypeID[purkinje_cell] = set_neuron_params(
		Neurons,
		purkinje_cell,
		"purkinje_cell.dat",
		0,
		purkinje_cell_num,
		107.0,        //Cm
		-68.0,        //E_L
		0.8,          //dt_ref
		160.0, //250.0,        //Ie //850.0
		-70.0,        //Vr
		-55.0,        //Vth
		2.32,
		NORMAL,
        2,
        1., 0., 0.7, 8.3, //exc
        1., -85., 1.0, 10.0 //inh

	);

	NeuronTypeID[golgi_cell] = set_neuron_params(
		Neurons,
		golgi_cell,
		"golgi_cell.dat",
		0,
		golgi_cell_num,
		28.0,         //Cm
		-55.0,        //E_L
		2.0,          //dt_ref
		0.0,         //Ie
		-72.7,        //Vr
		-52.0,        //Vth
		2.3,
		NORMAL,
        1,
        1., 0.0, 45.5, 1.5 //exc
	);

	NeuronTypeID[stellate_cell] = set_neuron_params(
		Neurons,
		stellate_cell,
		"stellate_cell.dat",
		0,
		stellate_cell_num,
		107.0,         //Cm
		-68.0,        //E_L
		1.6,          //dt_ref
		30.0, //0.0         //Ie
		-70.0,        //Vr
		-55.0,        //Vth
		2.32, //1.6
		NORMAL,
        1,
        1.0, 0., 0.7 ,8.3 //exc

	);

	NeuronTypeID[basket_cell] = set_neuron_params(
		Neurons,
		basket_cell,
		"basket_cell.dat",
		0,
		basket_cell_num,
		107.0,
		-68.0,
		1.6,
		30.0,// 0.0
		-70.0,
		-55.0,
		2.32, //1.6
		NORMAL,
        1,
        1.0, 0., 0.7, 8.3 //exc

	);

	NeuronTypeID[dcn_cell] = set_neuron_params(
		Neurons,
		dcn_cell,
		"dcn_cell.dat",
		0,
		dcn_cell_num,
		122.3,
		-56.0,
		3.7,
		500.0,
		-70.0,
		-38.8,
		1.63, //1.0
		OUTPUT,
        2,
        1.0, 0.0, 75.8, 10.0, //exc
        1.0, -85.0, 30.0, 26.6 //inh

	);

	NeuronTypeID[io_cell] = set_neuron_params(
		Neurons,
		io_cell,
		"io_cell.dat",
		1,
		io_cell_num,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		INPUT,
        0

	);

}

void init_connectivity_params(Connectivity *connectivities, Neuron *neurons, int *NeuronTypeID, int *ConnectivityTypeID){

	int	granule_cell_num = 88158,
		glomerulus_num = 7073,
		purkinje_cell_num = 69,
		golgi_cell_num = 219,
		stellate_cell_num = 603,
		basket_cell_num = 603,
		dcn_cell_num = 12,
        io_cell_num = 12;

	ConnectivityTypeID[parallel_fiber_to_purkinje] = set_connectivity_params(
		connectivities,
		neurons,
		parallel_fiber_to_purkinje,
		"parallel_fiber_to_purkinje.dat",
		granule_cell_num,
		purkinje_cell_num,
		NeuronTypeID[granule_cell],
		NeuronTypeID[purkinje_cell],
		0.003*2.0,
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
		0.00004*15.0,
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
		0.003*29.0,
		3.0, //5.0,
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
		0.003*29.0,
		3.0, //5.0,
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
		0.002, //0.006
		4.0,
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
		-0.008*0.7,
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
		-5.3*0.07,
		2.0, //4.0,
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
		-5.3*0.07,
		2.0, //5.0,
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
		4.0,
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
		-10.0*3.5,
		2.0,
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
		1.0,
		4.0,
		0
	);
}

