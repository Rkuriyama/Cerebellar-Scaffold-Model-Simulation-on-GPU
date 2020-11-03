#include <stdio.h>
#include "struct_enum_def.h"
#include "option.h"

__global__ void PF_PC_LTD_LTP_ELL( char *spike, int max_conv, int *cindices, CTYPE *val, int teacher_max_conv, int *teacher_cindices,
                                int target_row, int tail,
                                int post_num, int pre_base, int teacher_base, int delay_pre, int delay_teacher, int total_nn ){
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int start = 0;
    unsigned int pre_id;
    unsigned int teacher_id;
    CTYPE w_max = 1.0;
    char S_pre = 0;
    char S_teacher = 0;
    char has_a2 = 1;
    int s_upper_bound = 50;
    int t_row = 0;

    int post_id = tid/max_conv;
    int teacher_t = target_row - delay_teacher, pre_t = target_row - delay_pre;

    teacher_id = teacher_cindices[ post_id*teacher_max_conv + 0 ]; // CF用

    
    if( tid < max_conv*post_num ){
        pre_id = cindices[ tid ];
        if( pre_id >= 0){
            CTYPE dw = 0;
            CTYPE w = val[ tid ];

            pre_t = ((pre_t > 0)?pre_t:target_row + tail - delay_pre );
            S_pre = spike[ pre_t*total_nn + pre_base + pre_id ];
            dw += (S_pre != 0)? 0.001*( w_max - w ) : 0;

            teacher_t = ((teacher_t > 0)?teacher_t:target_row + tail - delay_teacher );
            S_teacher = (teacher_id < 0)? 0 : spike[ teacher_t*total_nn + teacher_base + teacher_id ];

            if( has_a2 && S_teacher ){
                for( int s = 0; s < 50; s++ ){
                    t_row = (pre_t >= s )? pre_t - s : pre_t + tail - s;
                    dw += ( spike[ t_row*total_nn + pre_base  + pre_id  ] != 0 )? (-0.01)*w : 0;
                }
            }
            val[ tid ] = (w + dw < 0)? 0 : w + dw;
        }
    }
    
};

/*

__global__ void HebbianRule( STDP *plasticity, char *spike, unsigned int *rptr, unsigned int *cindices, CTYPE *val,
                             int target_row, int tail,
                             int post_num, int post_base, int pre_base, int total_nn){
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int start = 0;
    unsigned int target_index = 0;
    unsigned int pre_id = 0;
    CTYPE w = 0;
    CTYPE w_max = plasticity->w_max;
    CTYPE dw = 0;
    char has_a2 = plasticity->has_a2;
    char S_pre;
    char S_post;
    int s_upper_bound = plasticity->s_upper_bound;

    for( unsigned int post_id = 0; post_id < post_num; post_id++ ){
        start = rptr[post_id];
        target_index = start + tid;

        if( tid < rptr[post_id + 1] - rptr[post_id] ){
            pre_id = cindices[ target_index ];
            w = val[ target_index ];

            S_pre = spike[ target_row*total_nn + pre_base + pre_id ];
            S_post = spike[ target_row*total_nn + post_base + post_id ];

            dw = plasticity->a0( w, w_max, 0 );
            dw += plasticity->a1_j( w, w_max, 0 )*S_pre;
            dw += plasticity->a1_i( w, w_max, 0 )*S_post;

            if( has_a2 && ( S_pre || S_post ) ){
                for( int s = 0; s < s_upper_bound; s++ ){
                    // target_row の修正
                    target_row = (target_row - s < 0)?:target_row = tail + s - 1 ;
    
                    dw += (S_pre  &&  spike[ (target_row - s)*total_nn + post_base + post_id ] ) ? plasticity->a2_ji( w, w_max, s ) : 0;
                    dw += (S_post &&  spike[ (target_row - s)*total_nn + pre_base  + pre_id  ] ) ? plasticity->a2_ij( w, w_max, s ) : 0;
                }
            }

            val[ target_index ] = w + dw;
        }
    }
};


__global__ void PerceptronRule( STDP plasticity, char *spike, unsigned int *rptr, unsigned int *cindices, CTYPE *val, unsigned int *teacher_rptr, unsigned int *teacher_cindices,
                                int target_row, int tail,
                                int post_num, int pre_base, int teacher_base, int total_nn ){
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int start = 0;
    unsigned int target_index = 0;
    unsigned int pre_id;
    unsigned int teacher_id;
    CTYPE w = 0;
    CTYPE w_max = plasticity.w_max;
    CTYPE dw = 0;
    char S_pre;
    char S_teacher;
    char has_a2 = plasticity.has_a2;
    int s_upper_bound = plasticity.s_upper_bound;

    for( unsigned int post_id = 0; post_id < post_num; post_id++ ){
        start = rptr[post_id];
        target_index = start + tid;
        teacher_id = teacher_cindices[ teacher_rptr[ post_id ] ];

        if( tid < rptr[post_id + 1] - rptr[post_id] ){
            pre_id = cindices[ target_index ];
            w = val[ target_index ];

            S_pre = spike[ target_row + pre_base + pre_id ];
            S_teacher = spike[ target_row + teacher_base + teacher_id ];

            dw = plasticity.a0( w, w_max , 0);
            dw += plasticity.a1_j( w, w_max, 0 )*S_pre;
            dw += plasticity.a1_i( w, w_max, 0 )*S_teacher;

            if( has_a2 && (S_pre || S_teacher) ){
                for( int s = 0; s < s_upper_bound; s++ ){
                    // target_row の修正
                    target_row = (target_row - s < 0)?:target_row = tail + s - 1 ;
    
                    dw += (S_pre  &&  spike[ (target_row - s)*total_nn + teacher_base + teacher_id ] ) ? plasticity.a2_ji( w, w_max, s ) : 0;
                    dw += (S_teacher &&  spike[ (target_row - s)*total_nn + pre_base  + pre_id  ] ) ? plasticity.a2_ij( w, w_max, s ) : 0;
                }
            }
            val[ target_index ] = w + dw;
        }
    }
};
*/

