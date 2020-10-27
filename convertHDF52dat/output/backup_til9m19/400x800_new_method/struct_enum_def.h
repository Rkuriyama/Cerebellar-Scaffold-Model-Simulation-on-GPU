#ifndef STRUCT_ENUM_DEF_H
#define STRUCT_ENUM_DEF_H

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
	dcn_interneuron,
	io_cell,
	TotalNumOfCellTypes
};

enum	ConnectionType{
	Input=0,
	glomerulus_to_granule,
	glomerulus_to_golgi,
	ascending_axon_to_golgi,
	parallel_fiber_to_golgi,
	golgi_to_granule,
	ascending_axon_to_purkinje,
	parallel_fiber_to_purkinje,
	parallel_fiber_to_basket,
	parallel_fiber_to_stellate,
	basket_to_purkinje,
	stellate_to_purkinje,
	stellate_to_stellate,
	basket_to_basket,
	golgi_to_golgi,
	purkinje_to_dcn,
	io_to_purkinje,
	io_to_basket,
	io_to_stellate,
	io_to_dcn,
	io_to_dcn_interneuron,
	dcn_interneuron_to_io,
	glomerulus_to_dcn,
	TotalNumOfConnectivityTypes
};

typedef struct {
	enum NeuronType type;
	char filename[20];
	int num;
	int base_id;
	CTYPE Cm, tau_m, El, dt_ref, Ie, Vr, Vth, tau_exc, tau_inh, gL;
} Neuron;

typedef struct {
	enum ConnectionType type;
	char filename[20];
	int preNum;
	int postNum;
	enum NeuronType preType;
	enum NeuronType postType;
	CTYPE initial_weight;
	CTYPE delay;
	unsigned int *rptr, *host_rptr;
	unsigned int *cindices;
	CTYPE *val;
	unsigned int max_conv;
	int pr;
	CTYPE *pr_out,*tmp;
} Connectivity;

typedef char (*pointFunction_t)(const float r, const CTYPE time);

typedef struct {
	enum NeuronType type;
	int base_id;
	unsigned int *IdList;
	int num;
	int func_id;
}InputFunctionsStruct;



#endif
