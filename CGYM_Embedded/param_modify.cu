#include <stdio.h>
#include <stdlib.h>
#include "./src/GSS/GPUScaffoldSimulation.h"

#include <math.h>

#define BACKGROUND_NOISE (5.f)
#define __T_MAX (1000)

float host_SinusoidalOscillation (const float max, const float mean, const float osci, const float shift, const CTYPE time){
    return max/2 * (1 - cosf( 2*M_PI*(time/sec)/osci + shift ));
}


void CalcFreq( float *freq_list, int t){
    //float fleq = BACKGROUND_NOISE + host_SinusoidalOscillation( 30.f, 15.f, 1., 0, t);
    float fleq = 30.0;
        freq_list[0] = fleq;
        //freq_list[1] = fleq;
        //freq_list[2] = fleq;
        //freq_list[3] = fleq;
        //freq_list[4] = fleq;
        //freq_list[5] = fleq;
        freq_list[1] = 10.0;
        freq_list[2] = 0;
    return;
}


int main( int argc, char **argv ){
    int trials = 1;
    int T_cpu_sim = 10;
    int T_print = 50;


    switch(argc){
        case 4:
            T_print = atoi(argv[3]);
        case 3:
            T_cpu_sim = atoi(argv[2]);
        case 2:
            trials = atoi(argv[1]);
    }


    /////////////////////////////////////////////
    /// GSS
    NetworkEnvironment Env;
    CreateNetworkEnv( &Env, DEV_NUM, TotalNumOfCellTypes, TotalNumOfConnectivityTypes, INPUT_STIM_NUM, NumOfPlasticity );

    InitializeNetworkEnv( &Env, T_print, T_cpu_sim ); //Env, T_print, T_cpu_sim
    float *freq_list = (float *)malloc(sizeof(float)*Env.num_of_inputs);
    char *spike_output = NULL;
    int output_neuron_num = Env.Host_sim_val[0].Host.neuron_num;
    fprintf(stderr, "output-neuron-num = %d\n", output_neuron_num);
    float activity[2] = {0};
    //////////////////////////////////////////////

    for(int trial = 0; trial < trials; trial++){
        fprintf(stderr, "%d", trial); // log

        int t = 0;
        int step = 0;

        // reset membrane potential
        activity[0] = 0; activity[1] = 0;
        freq_list[0] = 10.0;
        //freq_list[1] = fleq;
        //freq_list[2] = fleq;
        //freq_list[3] = fleq;
        //freq_list[4] = fleq;
        //freq_list[5] = fleq;
        freq_list[1] = 10.;
        freq_list[2] = 0;

        //freq_list[0] = BACKGROUND_NOISE;
        //freq_list[1] = BACKGROUND_NOISE;
        //freq_list[2] = BACKGROUND_NOISE;
        //freq_list[3] = BACKGROUND_NOISE;
        //freq_list[4] = BACKGROUND_NOISE;
        //freq_list[5] = BACKGROUND_NOISE;
        //freq_list[6] = 0;
        //freq_list[7] = 0;
        SetInputFreq(&Env, freq_list);

        loop_n_steps( &Env, t, T_cpu_sim, trial, &spike_output );
        t += T_cpu_sim;

        while( t < __T_MAX)
        {
            CalcFreq( freq_list, t );
            SetInputFreq(&Env, freq_list);
            loop_n_steps( &Env, t, T_cpu_sim, trial, &spike_output ); // t+=T_cpu_sim はloop側でやるべき？
            t += T_cpu_sim;

        }
        ResetNetworkEnv( &Env, trial);
    }


    FinalizeNetworkEnv(&Env);

    return 0;
}
