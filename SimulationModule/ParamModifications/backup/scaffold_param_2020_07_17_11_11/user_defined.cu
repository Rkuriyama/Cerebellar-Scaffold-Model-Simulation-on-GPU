#ifndef USER_DEFINED_CU
#define USER_DEFINED_CU

#include "user_defined.h"

void init_neurons_params( Neuron *Neurons){

	set_neurons_params(
		Neurons,
		GranuleCell,
		"GranuleCell.dat",
		367637,
		36648,
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
		2002,
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
		864,
		2186,
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
		1632,
		3050,
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
		3168,
		4682,
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
		184,
		2002,
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
		28798,
		7850,
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
		28798,
		864,
		Glomerulus,
		GolgiCell,
		2.0,
		4.0,
	);

	set_connectivity_params(
		connectivities,
		GolgiGlomerulus,
		"GolgiGlomerulus.dat",
		864,
		28798,
		GolgiCell,
		Glomerulus,
		0,
		0,
	);

	set_connectivity_params(
		connectivities,
		GlomerulusGranule,
		"GlomerulusGranule.dat",
		28798,
		367637,
		Glomerulus,
		GranuleCell,
		9.0,
		4.0,
	);

	set_connectivity_params(
		connectivities,
		GranuleGolgi,
		"GranuleGolgi.dat",
		367637,
		864,
		GranuleCell,
		GolgiCell,
		0.4,
		5.0,
	);

	set_connectivity_params(
		connectivities,
		GolgiGranule,
		"GolgiGranule.dat",
		864,
		367637,
		GolgiCell,
		GranuleCell,
		-5.0,
		2.0,
	);

	set_connectivity_params(
		connectivities,
		AscAxonPurkinje,
		"AscAxonPurkinje.dat",
		367637,
		2002,
		GranuleCell,
		PurkinjeCell,
		75.0,
		2.0,
	);

	set_connectivity_params(
		connectivities,
		PFPurkinje,
		"PFPurkinje.dat",
		367637,
		2002,
		GranuleCell,
		PurkinjeCell,
		0.02,
		5.0,
	);

	set_connectivity_params(
		connectivities,
		PFBasket,
		"PFBasket.dat",
		367637,
		3168,
		GranuleCell,
		BasketCell,
		0.2,
		5.0,
	);

	set_connectivity_params(
		connectivities,
		PFStellate,
		"PFStellate.dat",
		367637,
		1632,
		GranuleCell,
		StellateCell,
		0.2,
		5.0,
	);

	set_connectivity_params(
		connectivities,
		GapJunctionsStellate,
		"GapJunctionsStellate.dat",
		1632,
		1632,
		StellateCell,
		StellateCell,
		-2.0,
		1.0,
	);

	set_connectivity_params(
		connectivities,
		GapJunctionsBasket,
		"GapJunctionsBasket.dat",
		3168,
		3168,
		BasketCell,
		BasketCell,
		-2.5,
		1.0,
	);

	set_connectivity_params(
		connectivities,
		GapJunctionsGolgi,
		"GapJunctionsGolgi.dat",
		864,
		864,
		GolgiCell,
		GolgiCell,
		-8.0,
		1.0,
	);

	set_connectivity_params(
		connectivities,
		PurkinjeDCN,
		"PurkinjeDCN.dat",
		2002,
		184,
		PurkinjeCell,
		DCNCell,
		-0.0075,
		4.0,
	);

	set_connectivity_params(
		connectivities,
		GlomerulusDCN,
		"GlomerulusDCN.dat",
		28798,
		184,
		Glomerulus,
		DCNCell,
		0.006,
		4.0,
	);

	set_connectivity_params(
		connectivities,
		BasketPurkinje,
		"BasketPurkinje.dat",
		3168,
		2002,
		BasketCell,
		PurkinjeCell,
		-9.0,
		4.0,
	);

	set_connectivity_params(
		connectivities,
		StellatePurkinje,
		"StellatePurkinje.dat",
		1632,
		2002,
		StellateCell,
		PurkinjeCell,
		-8.5,
		5.0,
	);

	set_connectivity_params(
		connectivities,
		AscendingAxonGolgi,
		"AscendingAxonGolgi.dat",
		367637,
		864,
		GranuleCell,
		GolgiCell,
		20.0,
		2.0,
	);

}

