#ifndef USER_DEFINED_CU
#define USER_DEFINED_CU

#include "user_defined.h"

void init_neurons_params( Neuron *Neurons){

	set_neurons_params(
		Neurons,
		GranuleCell,
		"GranuleCell.dat",
		22625,
		2294,
		3,
		-58,
		1,
		0,
		-70,
		-35,
		0.5,
		10,
		1.5
	);

	set_neurons_params(
		Neurons,
		PurkinjeCell,
		"PurkinjeCell.dat",
		126,
		0,
		620,
		-70,
		1,
		22,
		-70,
		-50,
		0.5,
		1.6,
		7.0
	);

	set_neurons_params(
		Neurons,
		GolgiCell,
		"GolgiCell.dat",
		54,
		134,
		76,
		-70,
		1,
		0,
		-70,
		-50,
		0.5,
		10,
		3.6
	);

	set_neurons_params(
		Neurons,
		StellateCell,
		"StellateCell.dat",
		102,
		188,
		14.6,
		-70,
		1,
		0,
		-70,
		-55,
		0.64,
		2,
		1.0
	);

	set_neurons_params(
		Neurons,
		BasketCell,
		"BasketCell.dat",
		200,
		290,
		14.6,
		-70,
		1,
		0,
		-70,
		-55,
		0.64,
		2,
		1.0
	);

	set_neurons_params(
		Neurons,
		DCNCell,
		"DCNCell.dat",
		8,
		126,
		89,
		-70,
		1,
		0,
		-70,
		-40,
		7.1,
		13.6,
		1.56
	);

	set_neurons_params(
		Neurons,
		Glomerulus,
		"Glomerulus.dat",
		1804,
		490,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		0
	);

}

void init_connectivity_params(Connectivity *connectivities){

	set_connectivity_params(
		connectivities,
		GlomerulusGolgi,
		"GlomerulusGolgi.dat",
		1804,
		54,
		Glomerulus,
		GolgiCell,
		2.0,
		4.0,
	);

	set_connectivity_params(
		connectivities,
		GolgiGlomerulus,
		"GolgiGlomerulus.dat",
		54,
		1804,
		GolgiCell,
		Glomerulus,
		0,
		0,
	);

	set_connectivity_params(
		connectivities,
		GlomerulusGranule,
		"GlomerulusGranule.dat",
		1804,
		22625,
		Glomerulus,
		GranuleCell,
		9.0,
		4.0,
	);

	set_connectivity_params(
		connectivities,
		GranuleGolgi,
		"GranuleGolgi.dat",
		22625,
		54,
		GranuleCell,
		GolgiCell,
		0.4,
		5.0,
	);

	set_connectivity_params(
		connectivities,
		GolgiGranule,
		"GolgiGranule.dat",
		54,
		22625,
		GolgiCell,
		GranuleCell,
		-5.0,
		2.0,
	);

	set_connectivity_params(
		connectivities,
		AscAxonPurkinje,
		"AscAxonPurkinje.dat",
		22625,
		126,
		GranuleCell,
		PurkinjeCell,
		75.0,
		2.0,
	);

	set_connectivity_params(
		connectivities,
		PFPurkinje,
		"PFPurkinje.dat",
		22625,
		126,
		GranuleCell,
		PurkinjeCell,
		0.02,
		5.0,
	);

	set_connectivity_params(
		connectivities,
		PFBasket,
		"PFBasket.dat",
		22625,
		200,
		GranuleCell,
		BasketCell,
		0.2,
		5.0,
	);

	set_connectivity_params(
		connectivities,
		PFStellate,
		"PFStellate.dat",
		22625,
		102,
		GranuleCell,
		StellateCell,
		0.2,
		5.0,
	);

	set_connectivity_params(
		connectivities,
		GapJunctionsStellate,
		"GapJunctionsStellate.dat",
		102,
		102,
		StellateCell,
		StellateCell,
		-2.0,
		1.0,
	);

	set_connectivity_params(
		connectivities,
		GapJunctionsBasket,
		"GapJunctionsBasket.dat",
		200,
		200,
		BasketCell,
		BasketCell,
		-2.5,
		1.0,
	);

	set_connectivity_params(
		connectivities,
		GapJunctionsGolgi,
		"GapJunctionsGolgi.dat",
		54,
		54,
		GolgiCell,
		GolgiCell,
		-8.0,
		1.0,
	);

	set_connectivity_params(
		connectivities,
		PurkinjeDCN,
		"PurkinjeDCN.dat",
		126,
		8,
		PurkinjeCell,
		DCNCell,
		-0.0075,
		4.0,
	);

	set_connectivity_params(
		connectivities,
		GlomerulusDCN,
		"GlomerulusDCN.dat",
		1804,
		8,
		Glomerulus,
		DCNCell,
		0.006,
		4.0,
	);

	set_connectivity_params(
		connectivities,
		BasketPurkinje,
		"BasketPurkinje.dat",
		200,
		126,
		BasketCell,
		PurkinjeCell,
		-9.0,
		4.0,
	);

	set_connectivity_params(
		connectivities,
		StellatePurkinje,
		"StellatePurkinje.dat",
		102,
		126,
		StellateCell,
		PurkinjeCell,
		-8.5,
		5.0,
	);

	set_connectivity_params(
		connectivities,
		AscendingAxonGolgi,
		"AscendingAxonGolgi.dat",
		22625,
		54,
		GranuleCell,
		GolgiCell,
		20.0,
		2.0,
	);

}

