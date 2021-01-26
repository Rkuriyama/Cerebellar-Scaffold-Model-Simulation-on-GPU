#ifndef __GSS_N_C_LIST_
#define __GSS_N_C_LIST_

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



#endif
