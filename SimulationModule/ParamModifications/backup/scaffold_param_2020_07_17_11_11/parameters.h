#ifndef  _PARAMETERS_H_
#define  _PARAMETERS_H_



// neuron geometric params
#define GoC_soma_radius 8.
#define GoC_dendrite_radius 50.
#define GoC_axon_xlength 150.
#define GoC_axon_ylength 150.
#define GoC_axon_zlength 8.

#define Glom_soma_radius 1.5

#define GrC_soma_radius 2.5
#define GrC_dendrite_max_length 40.

#define PC_dendrite_width 150. // width of dendrite plane base
#define PC_soma_radius 7.5

#define BC_dendrite_radius 15.
#define BC_soma_radius 6.

#define SC_dendrite_radius 15.
#define SC_soma_radius 4.



// connection params
#define Glom_GrC_con 4
#define GoC_Glom_div 40
#define aa_GoC_con 400
#define pf_GoC_con 1200
#define PC_DCNC_div 5
#define Glom_DCNC_con 147



//synaptic params
#define Glom_GrC_WEIGHT 9.
#define Glom_GoC_WEIGHT 2.
#define GoC_Glom_WEIGHT -5.
#define GoC_GoC_WEIGHT  -8.
#define aa_GoC_WEIGHT   20.
#define pf_GoC_WEIGHT   0.4
#define SC_SC_WEIGHT    -2.
#define BC_BC_WEIGHT    -2.5
#define pf_SC_WEIGHT    0.2
#define pf_BC_WEIGHT    0.2
#define SC_PC_WEIGHT    -8.5
#define BC_PC_WEIGHT    -9.
#define aa_PC_WEIGHT    75.
#define pf_PC_WEIGHT    0.02
#define PC_DCNC_WEIGHT   -0.0075
#define Glom_DCNC_WEIGHT 0.006


#define Glom_GrC_DELAY 4.
#define Glom_GoC_DELAY 4.
#define GoC_Glom_DELAY 2.
#define GoC_GoC_DELAY  1.
#define aa_GoC_DELAY   2.
#define pf_GoC_DELAY   5.
#define SC_SC_DELAY    1.
#define BC_BC_DELAY    1.
#define pf_SC_DELAY    5.
#define pf_BC_DELAY    5.
#define SC_PC_DELAY    5.
#define BC_PC_DELAY    4.
#define aa_PC_DELAY    2.
#define pf_PC_DELAY    5.
#define PC_DCNC_DELAY   4.
#define Glom_DCNC_DELAY 4.

// neuron params
#define GoC_Cm		76.
#define GoC_tau_m	21.
#define GoC_El		-65.
#define GoC_dt_ref	2.
#define GoC_Ie		36.8
#define GoC_Vr		-75.
#define GoC_Vth		-55.
#define GoC_tau_exc	0.5
#define GoC_tau_inh 10.


#define GrC_Cm		3.
#define GrC_tau_m	2.
#define GrC_El		-74.
#define GrC_dt_ref	1.5
#define GrC_Ie		0.
#define GrC_Vr		-84.
#define GrC_Vth		-42.
#define GrC_tau_exc	0.5
#define GrC_tau_inh 10.


#define PC_Cm		620.
#define PC_tau_m	88.
#define PC_El		-62.
#define PC_dt_ref	0.8
#define PC_Ie		600.
#define PC_Vr		-72.
#define PC_Vth		-47.
#define PC_tau_exc	0.5
#define PC_tau_inh  1.6


#define BC_Cm		14.6
#define BC_tau_m	14.6
#define BC_El		-68.
#define BC_dt_ref	1.6
#define BC_Ie		15.6
#define BC_Vr		-78.
#define BC_Vth		-53.
#define BC_tau_exc	0.64
#define BC_tau_inh  2.


#define SC_Cm		14.6
#define SC_tau_m	14.6
#define SC_El		-68.
#define SC_dt_ref	1.6
#define SC_Ie		15.6
#define SC_Vr		-78.
#define SC_Vth		-53.
#define SC_tau_exc	0.64
#define SC_tau_inh  2.


#define DCNC_Cm			89.
#define DCNC_tau_m		57.
#define DCNC_El			-59.
#define DCNC_dt_ref		3.7
#define DCNC_Ie			55.8
#define DCNC_Vr			-69.
#define DCNC_Vth		-48.
#define DCNC_tau_exc	7.1
#define DCNC_tau_inh 	13.6


#endif
