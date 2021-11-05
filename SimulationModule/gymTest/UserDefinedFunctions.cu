#include "struct_enum_def.h"


/****************************** Input Stimuli ********************************/
__device__ char background_noise(const float r, const CTYPE time){
	char flag = 0;
	flag = ( (0.0 <= time ) && PoissonProcess( r, time, 1.0, 0.0 ) );
	return (flag)?1:0;
}

__device__ char tone_stim(const float r, const CTYPE time){
	char flag = 0;
	flag = ((300.0 <= time) && (time < 351) && PeriodicFiring( r, time, 140.0, 292.8 ) ) || PoissonProcess(r, time, 1.0, 0.0);
	return (flag)?1:0;
}



__device__ char puff_stim(const float r, const CTYPE time){
	char flag = 0;
    CTYPE fleq;
    fleq = SinusoidalOscillation( 3.f, 1.5f, 2, 0, time );
	flag = ((0.0 <= time ) && PoissonProcess( r, time, fleq, 0 ) );
	return (flag)?1:0;
}



/****************************** Synapse Plasticity ********************************/

// Pythonでコードを吐き出すのであれば、どうするか
// 数式打ち込みできるようにしたい。

// __device__ CTYPE <pre>_<post>_<termID>( CTYPE w, CTYPE s ){}
// Coefficientsではない... term?
// w: current weight
// s: diff

// 現状、pre->postの発火順には対応しているが、post->preの発火順には対応していない.まぁ後でいいか

// __device__ __managed__ stdp_Term_t <pre>_<post>_<rule>[] = {, , ,};
stdp_Term_t* Coefficients_Function_Groups[] = {};


////////////////////////////////////////////////////////

void Init_Plasticity( STDP_PLASTICITY **p, int *ConnectivityTypeID ){
    
    *p = (STDP_PLASTICITY *)malloc(sizeof(STDP_PLASTICITY)*NumOfPlasticity);

    //Set_Plasticity( *p, ConnectivityTypeID, <rule>, <target>, <teacher>, <temrs>, time_window );

    for(int i = 0; i < NumOfPlasticity; i++){
        (*p)[i].rule = Teacher;
        (*p)[i].target = ConnectivityTypeID[ parallel_fiber_to_purkinje ];
        (*p)[i].teacher = ConnectivityTypeID[ io_to_purkinje ];
        (*p)[i].coefficients = PF_PC_Coefficients;
        (*p)[i].time_window = 50;

        //fprintf(stderr, "-  %p: %p %p \n", PF_PC_Coefficients, PF_PC_Coefficients[2], (*p)[p_PF_PC].coefficients[2]);
    }

    return;
}






