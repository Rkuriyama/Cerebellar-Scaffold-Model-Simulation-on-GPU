#include "GPUScaffoldSimulation.h"

//User's function
void CalcFreq(float *freq_list, int t){
    // background_noise
    freq_list[0] = 4.f;
    // tone_stim
    freq_list[1] = (30.-4)/2 * (1 - cosf( 2*M_PI*( (float) t / sec)/2.f )) ;
    // puff_stim
    freq_list[2] = (3.f)/2 * (1 - cosf( 2*M_PI*( (float) t / sec)/2.f ));
}



int main( int argc, char **argv ){
    int trials = 1;
    int T_cpu_sim = 25;
    int T_print = 50;

    char* spike_train = NULL;


    switch(argc){
        case 4:
            T_print = atoi(argv[3]);
        case 3:
            T_cpu_sim = atoi(argv[2]);
        case 2:
            trials = atoi(argv[1]);
    }


    NetworkEnvironment Env;
    
    //Create Network Environment
    CreateNetworkEnv( &Env, DEV_NUM, TotalNumOfCellTypes, TotalNumOfConnectivityTypes, INPUT_STIM_NUM, NumOfPlasticity );

    //Initialize Network Environment. Now, it just constructs a network which are hard-coded. We plan it could be done by importing json configuration file in the future.
    InitializeNetworkEnv( &Env, T_print, T_cpu_sim ); //Env, T_print, T_cpu_sim

    int neuron_num =  Env.Host_sim_val[0].Host.neuron_num;
    fprintf(stderr, "neuron_num = %d\n", neuron_num );

    float *freq_list = (float *)malloc(sizeof(float)*Env.num_of_inputs);

    for(int trial = 0; trial < trials; trial++){
        printf("%d\n", trial);

        int t = 0;
        while( t < T_MAX ){ // while(1) でcgymの判定でぬける.
            CalcFreq(freq_list, t);
            SetInputFreq(&Env, freq_list);

            loop_n_steps( &Env, t, T_cpu_sim, trial, &spike_train ); // t+=T_cpu_sim はloop側でやるべき？

            /*
            for(int c_t = 0; c_t < T_cpu_sim; c_t++){
                for(int i = 0; i < neuron_num; i++){
                    if(spike_train[ c_t*neuron_num + i] ){
                         fprintf(stdout, "%lf\t%d\n", (float)(t+c_t)/1000, i );
                         break;
                    }
                }
            }
            */

            t += T_cpu_sim;
        }
        ResetNetworkEnv( &Env, trial);
    }

    FinalizeNetworkEnv(&Env);

    return 0;
}
