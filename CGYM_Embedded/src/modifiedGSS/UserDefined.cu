// core
#include <stdio.h>
#include "struct_enum_def.h"
#include "init_params.h"
#include "InputStimuli.h"
#include "SynapticPlasticity.h"
/////////////////////////////////////////////////////////////
/// Initialize Neurons' params
void init_neurons_params( Neuron *Neurons, int *NeuronTypeID){

	int	granule_cell_num = 88158,
		glomerulus_num = 7073,
		purkinje_cell_num = 69,
		golgi_cell_num = 219,
		stellate_cell_num = 603,
		basket_cell_num = 603,
		dcn_cell_num = 12,
        io_cell_num = 12;

	set_neuron_params(
		Neurons,
        NeuronTypeID,
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

	set_neuron_params(
		Neurons,
        NeuronTypeID,
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

	set_neuron_params(
		Neurons,
        NeuronTypeID,
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
        1., -85., 1.0, 10.0 // 10.0 //inh

	);

	set_neuron_params(
		Neurons,
        NeuronTypeID,
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

	set_neuron_params(
		Neurons,
        NeuronTypeID,
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

	set_neuron_params(
		Neurons,
        NeuronTypeID,
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

	set_neuron_params(
		Neurons,
        NeuronTypeID,
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

	set_neuron_params(
		Neurons,
        NeuronTypeID,
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
//////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////
/// Initialize Connectivities' params
void init_connectivity_params(Connectivity *connectivities, Neuron *neurons, int *NeuronTypeID, int *ConnectivityTypeID){

	int	granule_cell_num = 88158,
		glomerulus_num = 7073,
		purkinje_cell_num = 69,
		golgi_cell_num = 219,
		stellate_cell_num = 603,
		basket_cell_num = 603,
		dcn_cell_num = 12,
        io_cell_num = 12;

	set_connectivity_params(
		connectivities,
        ConnectivityTypeID,
		neurons,
		parallel_fiber_to_purkinje,
		"parallel_fiber_to_purkinje.dat",
		granule_cell_num,
		purkinje_cell_num,
		NeuronTypeID[granule_cell],
		NeuronTypeID[purkinje_cell],
		//0.00355*2.0,//0.003*2.0,
		0.0036*2.0,//0.003*2.0,
		2.0, //5.0,
		1
	);

	set_connectivity_params(
		connectivities,
        ConnectivityTypeID,
		neurons,
		parallel_fiber_to_golgi,
		"parallel_fiber_to_golgi.dat",
		granule_cell_num,
		golgi_cell_num,
		NeuronTypeID[granule_cell],
		NeuronTypeID[golgi_cell],
		0.00004*15.0,
		1.0, //5.0,
		0
	);

	set_connectivity_params(
		connectivities,
        ConnectivityTypeID,
		neurons,
		parallel_fiber_to_stellate,
		"parallel_fiber_to_stellate.dat",
		granule_cell_num,
		stellate_cell_num,
		NeuronTypeID[granule_cell],
		NeuronTypeID[stellate_cell],
		0.003*29.0,
		1.0, //5.0,
		0
	);

	set_connectivity_params(
		connectivities,
        ConnectivityTypeID,
		neurons,
		parallel_fiber_to_basket,
		"parallel_fiber_to_basket.dat",
		granule_cell_num,
		basket_cell_num,
		NeuronTypeID[granule_cell],
		NeuronTypeID[basket_cell],
		0.003*29.0,
		1.0, //5.0,
		0
	);

	set_connectivity_params(
		connectivities,
        ConnectivityTypeID,
		neurons,
		glomerulus_to_dcn,
		"glomerulus_to_dcn.dat",
		glomerulus_num,
		dcn_cell_num,
		NeuronTypeID[glomerulus],
		NeuronTypeID[dcn_cell],
	    0.006,
		4.0,
		0
	);

	set_connectivity_params(
		connectivities,
        ConnectivityTypeID,
		neurons,
		purkinje_to_dcn,
		"purkinje_to_dcn.dat",
		purkinje_cell_num,
		dcn_cell_num,
		NeuronTypeID[purkinje_cell],
		NeuronTypeID[dcn_cell],
		-0.008*0.7,
		1.0,
		0
	);

	set_connectivity_params(
		connectivities,
        ConnectivityTypeID,
		neurons,
		basket_to_purkinje,
		"basket_to_purkinje.dat",
		basket_cell_num,
		purkinje_cell_num,
		NeuronTypeID[basket_cell],
		NeuronTypeID[purkinje_cell],
		-5.3*0.07,
		1.0, //4.0,
		0
	);

	set_connectivity_params(
		connectivities,
        ConnectivityTypeID,
		neurons,
		stellate_to_purkinje,
		"stellate_to_purkinje.dat",
		stellate_cell_num,
		purkinje_cell_num,
		NeuronTypeID[stellate_cell],
		NeuronTypeID[purkinje_cell],
		-5.3*0.07,
		1.0, //5.0,
		0
	);

	set_connectivity_params(
		connectivities,
        ConnectivityTypeID,
		neurons,
		glomerulus_to_granule,
		"glomerulus_to_granule.dat",
		glomerulus_num,
		granule_cell_num,
		NeuronTypeID[glomerulus],
		NeuronTypeID[granule_cell],
		4.0,
		1.0, //4.0,
		0
	);

	set_connectivity_params(
		connectivities,
        ConnectivityTypeID,
		neurons,
		golgi_to_granule,
		"golgi_to_granule.dat",
		golgi_cell_num,
		granule_cell_num,
		NeuronTypeID[golgi_cell],
		NeuronTypeID[granule_cell],
		0.0,//-10.0*3.5,
		1.0, //2.0,
		0
	);

	set_connectivity_params(
		connectivities,
        ConnectivityTypeID,
		neurons,
		io_to_purkinje,
		"io_to_purkinje.dat",
		io_cell_num,
		purkinje_cell_num,
		NeuronTypeID[io_cell],
		NeuronTypeID[purkinje_cell],
		1.0, //1.0,
		1.0, //4.0,
		0
	);
}
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
/// Initialize Synaptic plasticity
__device__ CTYPE PF_PC_A1_J( CTYPE w, CTYPE s ){
    //return 0.000001*( 1.0 - w );
    return 0.0;
}
__device__ CTYPE PF_PC_A2_TI( CTYPE w, CTYPE s ){
    //return -0.0027*w;
    return -0.1*w;
}

__device__ __managed__ stdp_Coefficient_t PF_PC_Coefficients[] = {NULL, NULL, PF_PC_A1_J, PF_PC_A2_TI, NULL};


void Init_Plasticity( STDP_PLASTICITY **p, int *ConnectivityTypeID ){
    *p = (STDP_PLASTICITY *)malloc(sizeof(STDP_PLASTICITY)*NumOfPlasticity);

    (*p)[p_PF_PC].rule = Teacher;
    (*p)[p_PF_PC].target = ConnectivityTypeID[ parallel_fiber_to_purkinje ];
    (*p)[p_PF_PC].teacher = ConnectivityTypeID[ io_to_purkinje ];
    (*p)[p_PF_PC].coefficients = PF_PC_Coefficients;
    (*p)[p_PF_PC].time_window = 50;

    fprintf(stderr, "-  %p: %p, %p \n", PF_PC_Coefficients, PF_PC_Coefficients[2], (*p)[p_PF_PC].coefficients[2]);

    return;
}
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
/// Initialize Input Stims
__host__ void InitInputStimulation( InputFunctionsStruct *List, Neuron *host_neurons, int *NeuronTypeID){
    int stim_id = 0;

    setInputStim( &(List[stim_id]), host_neurons, NeuronTypeID, glomerulus, "background_noise.dat");
    stim_id++;

    /*
    setInputStim( &(List[stim_id]), host_neurons, NeuronTypeID, glomerulus, "cartVelocityL.dat");
    stim_id++;

    setInputStim( &(List[stim_id]), host_neurons, NeuronTypeID, glomerulus, "cartVelocityR.dat");
    stim_id++;

    setInputStim( &(List[stim_id]), host_neurons, NeuronTypeID, glomerulus, "PoleAngleL.dat");
    stim_id++;

    setInputStim( &(List[stim_id]), host_neurons, NeuronTypeID, glomerulus, "PoleAngleR.dat");
    stim_id++;

    setInputStim( &(List[stim_id]), host_neurons, NeuronTypeID, glomerulus, "PoleAngVelocityL.dat");
    stim_id++;

    setInputStim( &(List[stim_id]), host_neurons, NeuronTypeID, glomerulus, "PoleAngVelocityR.dat");
    stim_id++;
    */

    setInputStim( &(List[stim_id]), host_neurons, NeuronTypeID, io_cell, "errPoleAngL.dat");
    stim_id++;

    setInputStim( &(List[stim_id]), host_neurons, NeuronTypeID, io_cell, "errPoleAngR.dat");
    stim_id++;

    return;
}
//////////////////////////////////////////////////////////////////////
