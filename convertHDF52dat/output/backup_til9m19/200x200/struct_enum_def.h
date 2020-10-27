#ifndef STRUCT_ENUM_DEF_H
#define STRUCT_ENUM_DEF_H

typedef float CTYPE;

enum	NeuronType{
	NONE=-1,
	GranuleCell,
	PurkinjeCell,
	GolgiCell,
	StellateCell,
	BasketCell,
	DCNCell,
	Glomerulus,
	TotalNumOfCellTypes
};

enum	ConnectionType{
	Input=0,
	GlomerulusGolgi,
	GlomerulusGranule,
	GranuleGolgi,
	GolgiGranule,
	AscAxonPurkinje,
	PFPurkinje,
	PFBasket,
	PFStellate,
	GapJunctionsStellate,
	GapJunctionsBasket,
	GapJunctionsGolgi,
	PurkinjeDCN,
	GlomerulusDCN,
	BasketPurkinje,
	StellatePurkinje,
	AscendingAxonGolgi,
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
} Connectivity;

typedef int (*pointFunction_t)(const float r, const CTYPE time);

typedef struct {
	enum NeuronType type;
	int base_id;
	unsigned int *IdList;
	int num;
	int func_id;
}InputFunctionsStruct;



#endif
