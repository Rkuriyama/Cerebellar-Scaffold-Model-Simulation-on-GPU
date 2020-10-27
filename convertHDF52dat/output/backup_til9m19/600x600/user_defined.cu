#ifndef USER_DEFINED_CU
#define USER_DEFINED_CU

#include "user_defined.h"

void init_neurons_params( Neuron *Neurons){

	set_neurons_params(
		Neurons,
		GranuleCell,
		"GranuleCell.dat",
		206222,
		20616,
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
		1121,
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
		486,
		1225,
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
		918,
		1711,
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
		1784,
		2629,
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
		104,
		1121,
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
		16203,
		4413,
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
		16203,
		486,
		Glomerulus,
		GolgiCell,
		2.0,
		4.0,
	);

	set_connectivity_params(
		connectivities,
		GolgiGlomerulus,
		"GolgiGlomerulus.dat",
		486,
		16203,
		GolgiCell,
		Glomerulus,
		0,
		0,
	);

	set_connectivity_params(
		connectivities,
		GlomerulusGranule,
		"GlomerulusGranule.dat",
		16203,
		206222,
		Glomerulus,
		GranuleCell,
		9.0,
		4.0,
	);

	set_connectivity_params(
		connectivities,
		GranuleGolgi,
		"GranuleGolgi.dat",
		206222,
		486,
		GranuleCell,
		GolgiCell,
		0.4,
		5.0,
	);

	set_connectivity_params(
		connectivities,
		GolgiGranule,
		"GolgiGranule.dat",
		486,
		206222,
		GolgiCell,
		GranuleCell,
		-5.0,
		2.0,
	);

	set_connectivity_params(
		connectivities,
		AscAxonPurkinje,
		"AscAxonPurkinje.dat",
		206222,
		1121,
		GranuleCell,
		PurkinjeCell,
		75.0,
		2.0,
	);

	set_connectivity_params(
		connectivities,
		PFPurkinje,
		"PFPurkinje.dat",
		206222,
		1121,
		GranuleCell,
		PurkinjeCell,
		0.02,
		5.0,
	);

	set_connectivity_params(
		connectivities,
		PFBasket,
		"PFBasket.dat",
		206222,
		1784,
		GranuleCell,
		BasketCell,
		0.2,
		5.0,
	);

	set_connectivity_params(
		connectivities,
		PFStellate,
		"PFStellate.dat",
		206222,
		918,
		GranuleCell,
		StellateCell,
		0.2,
		5.0,
	);

	set_connectivity_params(
		connectivities,
		GapJunctionsStellate,
		"GapJunctionsStellate.dat",
		918,
		918,
		StellateCell,
		StellateCell,
		-2.0,
		1.0,
	);

	set_connectivity_params(
		connectivities,
		GapJunctionsBasket,
		"GapJunctionsBasket.dat",
		1784,
		1784,
		BasketCell,
		BasketCell,
		-2.5,
		1.0,
	);

	set_connectivity_params(
		connectivities,
		GapJunctionsGolgi,
		"GapJunctionsGolgi.dat",
		486,
		486,
		GolgiCell,
		GolgiCell,
		-8.0,
		1.0,
	);

	set_connectivity_params(
		connectivities,
		PurkinjeDCN,
		"PurkinjeDCN.dat",
		1121,
		104,
		PurkinjeCell,
		DCNCell,
		-0.0075,
		4.0,
	);

	set_connectivity_params(
		connectivities,
		GlomerulusDCN,
		"GlomerulusDCN.dat",
		16203,
		104,
		Glomerulus,
		DCNCell,
		0.006,
		4.0,
	);

	set_connectivity_params(
		connectivities,
		BasketPurkinje,
		"BasketPurkinje.dat",
		1784,
		1121,
		BasketCell,
		PurkinjeCell,
		-9.0,
		4.0,
	);

	set_connectivity_params(
		connectivities,
		StellatePurkinje,
		"StellatePurkinje.dat",
		918,
		1121,
		StellateCell,
		PurkinjeCell,
		-8.5,
		5.0,
	);

	set_connectivity_params(
		connectivities,
		AscendingAxonGolgi,
		"AscendingAxonGolgi.dat",
		206222,
		486,
		GranuleCell,
		GolgiCell,
		20.0,
		2.0,
	);

}

