#include <stdio.h>
#include <stdlib.h>
#include "./src/cgym/cgym.h"
#include "./src/GSS/GPUScaffoldSimulation.h"

#include <math.h>

#define BACKGROUND_NOISE (0.f)
#define RESET_NOISE (3.f)
//#define TAU (100.f)

float W = 2.f;

float TAU = 100.f;

float GaussianDist( float x, float mu, float sig ){
    return exp( - (x - mu)*(x - mu)/(2*sig*sig) );
}

void CalcFreq(float *freq_list, float64 *state, int t){
    const float x = state[0], x_dot = state[1], theta = state[2], theta_dot = state[3];
    float max[2] = {0., 0.};
    float tmp[2];


    /////////////////////////////////
    //  ReLU
    //freq_list[0] = ( state[1] < 0 )? state[1]*( -20) + 10.0+ BACKGROUND_NOISE: BACKGROUND_NOISE;
    //freq_list[1] = ( state[1] > 0 )? state[1]*( 20)  + 10.0+ BACKGROUND_NOISE: BACKGROUND_NOISE;

    //freq_list[2] = ( state[2] < 0 )? state[2]*(-180) + 10.0+ BACKGROUND_NOISE: BACKGROUND_NOISE;
    //freq_list[3] = ( state[2] > 0 )? state[2]*( 180) + 10.0+ BACKGROUND_NOISE: BACKGROUND_NOISE;

    //freq_list[4] = ( state[3] < 0 )? state[3]*(-30) + 10.0+ BACKGROUND_NOISE: BACKGROUND_NOISE;
    //freq_list[5] = ( state[3] > 0 )? state[3]*( 30) + 10.0+ BACKGROUND_NOISE: BACKGROUND_NOISE;


    //tmp[0] = (x_dot > 0)? x_dot * 7.f: 0;
    //tmp[1] = (x_dot < -0)? x_dot * -7.f: 0;
    //max[0] = ( max[0] < tmp[0] )? tmp[0] : max[0];
    //max[1] = ( max[1] < tmp[1] )? tmp[1] : max[1];

    //tmp[0] = (theta < -0)? theta * (-66.f): 0;
    //tmp[1] = (theta > 0)? theta * (66.f): 0;
    //max[0] = ( max[0] < tmp[0] )? tmp[0] : max[0];
    //max[1] = ( max[1] < tmp[1] )? tmp[1] : max[1];

    //tmp[0] = (theta_dot < -0)? theta_dot * (-10.f): 0;
    //tmp[1] = (theta_dot > 0)? theta_dot * (10.f): 0;
    //max[0] = ( max[0] < tmp[0] )? tmp[0] : max[0];
    //max[1] = ( max[1] < tmp[1] )? tmp[1] : max[1];
    //////////////////////////////////



    //////////////////////////////////
    /// Constant 
    freq_list[0] = ( theta_dot < 0 )? 30.0+ BACKGROUND_NOISE: BACKGROUND_NOISE;
    freq_list[1] = ( theta_dot > 0 )? 30.0+ BACKGROUND_NOISE: BACKGROUND_NOISE;

    freq_list[2] = ( theta_dot < 0 )? 30.0+ BACKGROUND_NOISE: BACKGROUND_NOISE;
    freq_list[3] = ( theta_dot > 0 )? 30.0+ BACKGROUND_NOISE: BACKGROUND_NOISE;

    freq_list[4] = ( theta_dot < 0 )? 30.0+ BACKGROUND_NOISE: BACKGROUND_NOISE;
    freq_list[5] = ( theta_dot > 0 )? 30.0+ BACKGROUND_NOISE: BACKGROUND_NOISE;


    max[0] = ( theta_dot < 0)? 20 : 0;
    //max[1] = ( theta_dot > 0)? 20 : 0;
    max[1] = ( theta_dot > 0)? 20 : 0;



    //////////////////////////////////

    /*
    /////// left
    // x
    if( !( theta > 0 && x > 0) ){
        tmp[0] = fabsf(x);
        max[0] = ( max[0] < tmp[0] )? tmp[0] : max[0];
    }
    // x dot
    if( !( x_dot < 0) ){
        tmp[0] = fabsf(x_dot)*(7.f);
    }
    max[0] = ( max[0] < tmp[0] )? tmp[0] : max[0];

    // theta
    if( !(theta > 0 && theta_dot > 0) ){
        tmp[0] = fabsf(theta)*(66.f);
    }
    max[0] = ( max[0] < tmp[0] )? tmp[0] : max[0];

    // theta dot
    //tmp[0] = ( state[3] < 0 )? state[3]*(-10) : 0.01;
    //max[0] = ( max[0] < tmp[0] )? tmp[0] : max[0];

    //////// right
    // x
    if( !( theta < 0 && x < 0) ){
        tmp[1] = fabsf(x);
        max[1] = ( max[1] < tmp[1] )? tmp[1] : max[1];
    }
    // x dot
    if( !( x_dot > 0 ) ){
        tmp[1] = fabsf(x_dot)*7.f;
        max[1] = ( max[1] < tmp[1] )? tmp[1] : max[1];
    }
    // theta
    if( !( theta < 0 && theta_dot < 0) ){
        tmp[1] = fabsf(theta)*66.f;
        max[1] = ( max[1] < tmp[1] )? tmp[1] : max[1];
    }
    */


    freq_list[6] = max[0];
    freq_list[7] = max[1];


    // cartVelocity - L - R
    //freq_list[0] = 50.f*GaussianDist( state[1],   2., 1.0) + BACKGROUND_NOISE;
    //freq_list[1] = 50.f*GaussianDist( state[1],  -2., 1.0) + BACKGROUND_NOISE;
    //freq_list[0] = 50.f*GaussianDist( state[2],  -0.21, 0.11);// + BACKGROUND_NOISE;
    //freq_list[1] = 50.f*GaussianDist( state[2],   0.21, 0.11);// + BACKGROUND_NOISE;
    //freq_list[0] = 50.f*GaussianDist( state[3], -3., 1.5) + BACKGROUND_NOISE;
    //freq_list[1] = 50.f*GaussianDist( state[3],  3., 1.5) + BACKGROUND_NOISE;

    // poleAngle - L - R
    //freq_list[2] = 50.f*GaussianDist( state[2], -0.21, 0.11);// + BACKGROUND_NOISE;
    //freq_list[3] = 50.f*GaussianDist( state[2],  0.21, 0.11);// + BACKGROUND_NOISE;
    //freq_list[2] = 50.f*GaussianDist( state[3], -3., 1.5) + BACKGROUND_NOISE;
    //freq_list[3] = 50.f*GaussianDist( state[3],  3., 1.5) + BACKGROUND_NOISE;

    // poleVelocity -L - R
    //freq_list[4] = 50.f*GaussianDist( state[3], -3., 1.5) + BACKGROUND_NOISE;
    //freq_list[5] = 50.f*GaussianDist( state[3],  3., 1.5) + BACKGROUND_NOISE;


    // errPoleAng - L - R
    //freq_list[6] = ( state[2] < 0 )? state[2]*(-30) : 0.f;
    //freq_list[7] = ( state[2] > 0 )? state[2]*( 30) : 0.f;

    //float max[2] = {0., 0.};
    //float tmp[2];

    //tmp[0] = GaussianDist( state[1],  2., 1. );
    //tmp[1] = GaussianDist( state[1], -2., 1. );
    //max[0] = ( max[0] < tmp[0] )? tmp[0] : max[0];
    //max[1] = ( max[1] < tmp[1] )? tmp[1] : max[1];
    //
    //tmp[0] = GaussianDist( state[2], -0.21, 0.11);
    //tmp[1] = GaussianDist( state[2],  0.21, 0.11);
    //max[0] = ( max[0] < tmp[0] )? tmp[0] : max[0];
    //max[1] = ( max[1] < tmp[1] )? tmp[1] : max[1];

    //tmp[0] = GaussianDist( state[3], -3., 1.5);
    //tmp[1] = GaussianDist( state[3],  3., 1.5);
    //max[0] = ( max[0] < tmp[0] )? tmp[0] : max[0];
    //max[1] = ( max[1] < tmp[1] )? tmp[1] : max[1];

    //freq_list[6] = 10.f*max[0];
    //freq_list[7] = 10.f*max[1];

}

FILE *fp_dg = fopen("dg.res", "w+");

int calcAct( float *activity, char *spike, int neuron_num, int T_cpu_sim, int old_action ){
    float dg[2] = {0};
    int a = 0;
    static int time = 0;
    for(int t = 0; t < T_cpu_sim; t++){
        dg[0] = 0; dg[1] = 0;
        for(int id = 0; id < neuron_num/2; id++){
            dg[0] += (spike[neuron_num*t + id])? 1: 0;
        }
        for(int id = neuron_num/2; id < neuron_num; id++){
            dg[1] += (spike[neuron_num*t + id])? 1: 0;
        }
        dg[0] = ( -activity[0] + W* dg[0]/neuron_num*2 )/TAU;
        dg[1] = ( -activity[1] + W* dg[1]/neuron_num*2 )/TAU;
        activity[0] += dg[0];
        activity[1] += dg[1];
        a += (activity[0] > activity[1])? 1: -1;
    }

    fprintf(fp_dg, "dg_0\t%f\tdg_1\t%f\n", dg[0], dg[1]);

    time += T_cpu_sim;

    //return ( (a > 0)? 0: 1 );
    //fprintf(stdout, "%d %f %f\n", time, activity[0],activity[1] );
    //return (activity[0] > activity[1] )? 0: (activity[0] == activity[1])? !(old_action) : 1;
    return (activity[0] > activity[1] )? 0: 1;
}

int calcSpikes( char *spike, int neuron_num, int T_cpu_sim, int old_action ){
    int a_spike[2] = {0};
    static int time = 0;
    static int count_stop = 0;
    for(int t = 0; t < T_cpu_sim; t++){
        for(int id = 0; id < neuron_num/2; id++){
            a_spike[0] += (spike[neuron_num*t + id])? 1: 0;
        }
        for(int id = neuron_num/2; id < neuron_num; id++){
            a_spike[1] += (spike[neuron_num*t + id])? 1: 0;
        }
    }
    //if( a_spike[0] == a_spike[1] ){
    //    count_stop++;
    //    fprintf(stderr, "equal: %d\n", count_stop);
    //}
    //fprintf(stderr, "\n%d-%d\n", a_spike[0], a_spike[1] );
    return (a_spike[0] > a_spike[1])? 0: (a_spike[0] == a_spike[1])? !(old_action) : 1;
}

int main( int argc, char **argv ){
    int trials = 1;
    int T_cpu_sim = 10;
    int T_print = 50;

    int bin = 20;

    switch(argc){
        case 5:
            TAU = atof(argv[4]);
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

    //////////////////////////////////////////////
    /// CGYM
    cgym_t *gym = gymCreate("CartPole-v1");
    gymEnvInfo(gym);

    float64* state = (float64*)gymState(gym);
    int64* action = (int64*)gymAction(gym);
    float reward = 0.f;
    int done = 0;
    /////////////////////////////////////////////

    float64 max[4]={0};
    float64 min[4]={0};

    char file_name[256];
    sprintf(file_name, "CartPoleResults/%d_%d.dat", (int)TAU, T_cpu_sim);

    FILE *trial_time = fopen(file_name, "w+");
    fprintf(trial_time, "#trial\tt(step*T_cpu_sim)\tcartPos\tCartVel\tPoleAng\tPoleAngVel\n");

    FILE *freq_dat = fopen("freq_data.res", "w+");
    FILE *activity_dat = fopen("state_activity_data.res", "w+");

    int sum_t = 0;
    for(int trial = 0; trial < trials; trial++){
        fprintf(stderr, "%d", trial); // log
        if( trial % bin == 0){
            //fprintf(stdout, "%d\t%d\n", trial, sum_t/bin);
            sum_t = 0;
        }

        int t = 0;
        int step = 0;

        // reset membrane potential
        freq_list[0] = RESET_NOISE;
        freq_list[1] = RESET_NOISE;
        freq_list[2] = RESET_NOISE;
        freq_list[3] = RESET_NOISE;
        freq_list[4] = RESET_NOISE;
        freq_list[5] = RESET_NOISE;
        freq_list[6] = BACKGROUND_NOISE;
        freq_list[7] = BACKGROUND_NOISE;
        SetInputFreq(&Env, freq_list);

        loop_n_steps( &Env, t, T_cpu_sim, trial, &spike_output );
        activity[0] = 0; activity[1] = 0;

        t += T_cpu_sim;

        while(1)
        {
            //printf("%f %f %f %f %f %d\n",state[0], state[1], state[2], state[3], reward, done);
            //printf("%f %f => %lld\n", activity[0], activity[1], action[0]);

            CalcFreq(freq_list, state, t);
            fprintf(freq_dat, "%d %f %f %f %f %f %f %f %f\n", step, freq_list[0], freq_list[1], freq_list[2], freq_list[3], freq_list[4], freq_list[5], freq_list[6], freq_list[7]);

            SetInputFreq(&Env, freq_list);

            loop_n_steps( &Env, t, T_cpu_sim, trial, &spike_output ); // t+=T_cpu_sim はloop側でやるべき？
            // select_action();
            action[0] = calcAct( activity, spike_output, output_neuron_num, T_cpu_sim, action[0] );
            //action[0] = calcSpikes( spike_output, output_neuron_num, T_cpu_sim, action[0] );
            fprintf(activity_dat, "%d %f %f %f %f %f %f\n", step, state[0], state[1], state[2], state[3], activity[0], activity[1]);
            //action[0] = calcSpikes( spike_output, output_neuron_num, T_cpu_sim, action[0] );

            if(step == 0) fprintf(stderr, "\t%d\t", (int)action[0]);

            gymStep(gym);
            reward = gymReward(gym);
            done   = gymDone(gym);
            step++;

            for(int i = 0; i < 4; i++){
                max[i] = (state[i] > max[i])? state[i] : max[i];
                min[i] = (state[i] < min[i])? state[i] : min[i];
            }

            //gymRender(gym);

            t += T_cpu_sim;

            if(done)
            {
                sum_t += t;
                fprintf(trial_time, "%d\t%d\t%d\t%lf\t%lf\t%lf\t%lf\n", trial, step,  t, state[0], state[1], state[2], state[3] );
                fflush(trial_time);
                fprintf(stderr, "\t%d\n", step); // log

                break;
            }
        }

        gymReset(gym);
        ResetNetworkEnv( &Env, trial);

        fprintf(freq_dat, "\n"); // block separator for gnuplot
        fprintf(activity_dat, "\n"); // block separator for gnuplot
        fprintf(fp_dg, "\n");
    }


    for(int i = 0; i < 4; i++) fprintf(stderr, "state[%d] = [%f, %f]\n", i,  min[i], max[i]);

    gymDestroy(gym);
    FinalizeNetworkEnv(&Env);
    fclose(trial_time);
    fclose(freq_dat);
    fclose(activity_dat);
    fclose(fp_dg);

    return 0;
}
