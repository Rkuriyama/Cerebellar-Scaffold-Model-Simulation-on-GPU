#include <stdio.h>
#include "struct_enum_def.h"
#include "option.h"

__global__ void Hebb_plasticity( const char *spike, const int max_conv, const int *cindices, CTYPE *val, const int target_row, const int tail,
                            const int post_num, const int post_base, const int pre_base, const int delay, const stdp_Coefficient_t *coefficients, const int time_window, const int total_nn){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int pre_id;
    char S_pre = 0;
    char S_post = 0;
    int t_row = 0;

    int post_id = tid/max_conv;
    int pre_t = target_row - delay;

    if( tid < max_conv*post_num ){
        pre_id = cindices[ tid ];
        if( pre_id >= 0){
            CTYPE dw = 0;
            CTYPE w = val[ tid ];

            dw += (coefficients[A0] != NULL)? coefficients[ A0 ](w, 0) : 0;

            pre_t = ((pre_t < 0)? target_row + tail - delay :pre_t );
            S_pre = spike[ pre_t*total_nn + pre_base + pre_id ];
            S_post = spike[ target_row*total_nn + post_base + post_id ];

            dw += (coefficients[A1_J] != NULL && S_pre )? coefficients[ A1_J ](w, 0) : 0;
            dw += (coefficients[A1_I] != NULL && S_post)? S_post*coefficients[ A1_I ](w, 0) : 0;

            if( coefficients[A2_JI] != NULL && S_post ){
                for( int s = 0; s < time_window; s++ ){
                    t_row = (pre_t >= s )? pre_t - s : pre_t + tail - s;
                    dw += ( spike[ t_row*total_nn + pre_base  + pre_id  ] != 0 )? coefficients[A2_JI](w, s) : 0;
                }
            }
            w = w + dw;
            val[ tid ] = (w < 0)? 0 : (w > 1)? 1 : w;
        }
    }
}


__global__ void Teacher_plasticity( const char *spike, const int max_conv, const int *cindices, CTYPE *val, const int t_max_conv, const int *t_cindices,
                            const int target_row, const int tail,
                            const int post_num, const int teacher_base, const int pre_base, const int delay, const int t_delay, stdp_Coefficient_t *coefficients, const int time_window, const int total_nn){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int pre_id;
    char S_pre = 0;
    char S_teacher = 0;
    int t_row = 0;

    int post_id = tid/max_conv;
    int pre_t = target_row - delay;
    int teacher_t = target_row - t_delay;

    int teacher_id = (post_id < post_num)? t_cindices[ post_id*t_max_conv + 0 ] : -1; // CF用

    if( tid < max_conv*post_num ){
        pre_id = cindices[ tid ];
        if( pre_id >= 0){
            CTYPE dw = 0;
            CTYPE w = val[ tid ];

            dw += (coefficients[A0] != NULL)? coefficients[ A0 ](w, 0) : 0;

            pre_t = (pre_t < 0)? target_row + tail - delay : pre_t;
            S_pre = spike[ pre_t*total_nn + pre_base + pre_id ];

            teacher_t = ((teacher_t < 0)? target_row + tail - t_delay : teacher_t );
            S_teacher = (teacher_id < 0)? 0 : spike[ teacher_t*total_nn + teacher_base + teacher_id ];

            dw += (coefficients[A1_J] != NULL && S_pre    )? coefficients[ A1_J ](w, 0) : 0;
            dw += (coefficients[A1_I] != NULL && S_teacher)? coefficients[ A1_I ](w, 0) : 0;

            if( coefficients[A2_JI] != NULL && S_teacher ){
                for( int s = 0; s < 50; s++ ){
                    t_row = (pre_t >= s )? pre_t - s : pre_t + tail - s;
                    dw += ( spike[ t_row*total_nn + pre_base  + pre_id  ] != 0 )? coefficients[A2_JI](w, s) : 0;
                }
                dw = -w;
            }
            val[ tid ] = (w + dw < 0)? 0 : w + dw;
        }
    }
}


__host__ void invoke_stdp_plasticity( char *spike, Neuron *d_neurons,  Connectivity *d_connections, STDP_PLASTICITY *p, int target_row, int tail, int total_nn, cudaStream_t *streams ){
    for(int i = 0; i < NumOfPlasticity; i++){
        Connectivity *target = &d_connections[ p[i].target ];
        Connectivity *teacher = (p[i].teacher == C_NONE)? NULL : &d_connections[ p[i].teacher ];
        switch( p[i].rule ){
            case Hebb:
                Hebb_plasticity<<< (target->max_conv*target->postNum + 127)/128, 128, 0, streams[target->postType]>>>(spike, target->max_conv, target->ELL_cindices, target->ELL_val,
                            target_row, tail, target->postNum, d_neurons[target->postType].base_id, d_neurons[target->preType].base_id,
                            target->delay, p[i].coefficients, p[i].time_window, total_nn);
                break;
            case Teacher:
                Teacher_plasticity<<< (target->max_conv*target->postNum + 127)/128, 128, 0, streams[target->postType]>>>(spike, target->max_conv, target->ELL_cindices, target->ELL_val, teacher->max_conv, teacher->ELL_cindices,
                            target_row, tail, target->postNum, d_neurons[teacher->preType].base_id, d_neurons[target->preType].base_id,
                            target->delay, teacher->delay, p[i].coefficients, p[i].time_window, total_nn);
                break;
        }
    }
}

