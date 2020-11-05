#ifndef STRUCT_ENUM_DEF_H
#define STRUCT_ENUM_DEF_H

#include <stdio.h>
#include <curand_kernel.h>
#include <pthread.h>

typedef float CTYPE;

enum	NeuronType{
	NONE=-1,
	granule_cell,
	glomerulus,
	purkinje_cell,
	golgi_cell,
	stellate_cell,
	basket_cell,
	dcn_cell,
    io_cell,
	TotalNumOfCellTypes
};

enum	ConnectionType{
	C_NONE=-1,
	parallel_fiber_to_purkinje,
	parallel_fiber_to_golgi,
	parallel_fiber_to_stellate,
	parallel_fiber_to_basket,
	glomerulus_to_dcn,
	purkinje_to_dcn,
	basket_to_purkinje,
	stellate_to_purkinje,
	glomerulus_to_granule,
	golgi_to_granule,
    io_to_purkinje,
	TotalNumOfConnectivityTypes
};


enum DeviceType {
	NORMAL,
	INPUT,
	OUTPUT
};

enum ChannelType {
    AMPA,
    NMDA,
    inh,
    NumOfChannelTypes  
};

typedef struct {
	int type;
	char filename[20];
	char duplicate;
	int num;
	int base_id;
	CTYPE Cm, tau_m, El, dt_ref, Ie, Vr, Vth, tau_exc, tau_inh, gL;
    CTYPE *w;
    CTYPE *E;
    CTYPE *g_bar;
    CTYPE *tau;
    CTYPE *g;
    int c_num;
	enum DeviceType dev_type;
} Neuron;

typedef struct {
	enum ConnectionType type;
	char filename[20];
	int preNum, postNum;
	int preType, postType;
	CTYPE initial_weight;
	CTYPE delay;
	unsigned int max_conv;
    int *ELL_cindices;
    CTYPE *ELL_val;
	int pr;
	CTYPE *pr_out,*tmp;
} Connectivity;

typedef struct {
	Connectivity *Connectivities;
	int CPU_side,GPU_side;
	int num;
} Connectome;

typedef char (*pointFunction_t)(const float r, const CTYPE time);

typedef struct {
	int type;
	int base_id;
	unsigned int *IdList;
	int num;
	int func_id;
	curandStatePhilox4_32_10_t *state;
}InputFunctionsStruct;

typedef struct {
    char *spike,*from_spike, *host_spike, *type;
    int n,delay_max_row,total_nn,own_neuron_num;
    Neuron *neurons,*host_neurons;
	int *NeuronTypeID;
    unsigned int *start;
    cudaStream_t stream;
    FILE *fp;
	pthread_t th;
	int id,dev_id;
} pthread_val;

typedef struct {
    CTYPE *u,
          *g_exc,
          *g_inh,
          *Inoise,
          *dg_exc,
          *dg_inh;
    char *spike, *type;
    int *refractory_time_left;
    unsigned int start[TotalNumOfCellTypes], end[TotalNumOfCellTypes];
    Neuron *dev_neurons, neurons[TotalNumOfCellTypes];// host側かdevice側か
    Connectivity connectivities[TotalNumOfConnectivityTypes];
	int gpu_neurons_num;
	int gpu_connections_num;
    int neuron_num = 0,
        pre_neuron_num = 0,
        next_neuron_num = 0,
        total_neuron_num = 0,
        device_base = 0;
    InputFunctionsStruct InputStimList[3];
    curandStatePhilox4_32_10_t *state;
    cudaStream_t *streams;
    char *host_spike[2];
    pthread_t *print_thread, *cpu_sim_thread;
    pthread_val *print_arg;
    FILE *fp;
} Sim_cond_lif_exp;


typedef struct {
	pthread_t th;
	int Dev_num;
	int id;
	int n,delay_max_row;
	int *NeuronTypeID;
	Sim_cond_lif_exp Host;
	Sim_cond_lif_exp *Dev;
    FILE *fp;
} cpu_sim_thread_val;


// synaptic plasticity
typedef CTYPE (*stdp_Coefficient_t)( CTYPE w, CTYPE s );
enum STDP_COEFFICIENTS { A0, A1_I, A1_J, A2_JI, A2_IJ };
enum STDP_RULE { Hebb, Teacher, Perceptron };// 教師ありのやつなんていうの?

typedef struct {
    enum STDP_RULE rule;
    int target;
    int teacher = C_NONE;
    stdp_Coefficient_t *coefficients;
    int time_window;
}STDP_PLASTICITY;


#endif
