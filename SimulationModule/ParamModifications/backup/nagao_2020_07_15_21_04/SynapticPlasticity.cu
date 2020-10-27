#include <stdio.h>
#include "struct_enum_def.h"
#include "option.h"

__global__ void PF_PC_LTP(   char *spike, unsigned int *rptr, unsigned int *cindices, CTYPE *val,
                             int target_row, int tail,
                             int post_num, int post_base, int pre_base, int total_nn){
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int start = 0;
    unsigned int target_index = 0;
    unsigned int pre_id = 0;
    CTYPE w = 0;
    CTYPE w_max = 1.0;
    char S_pre;
    char S_post;

    for( unsigned int post_id = 0; post_id < post_num; post_id++ ){
        CTYPE dw = 0;
        start = rptr[post_id];
        target_index = start + tid;

        if( tid < rptr[post_id + 1] - rptr[post_id] ){
            pre_id = cindices[ target_index ];
            w = val[ target_index ];

            S_pre = spike[ target_row*total_nn + pre_base + pre_id ];
            S_post = spike[ target_row*total_nn + post_base + post_id ];

            dw += (S_pre != 0)? 0.01*( w_max - w ) : 0;

            val[ target_index ] = (w + dw > 1)? 1 : w+dw;
        }
    }
};


__global__ void PF_PC_LTD_LTP( char *spike, unsigned int *rptr, unsigned int *cindices, CTYPE *val, unsigned int *teacher_rptr, unsigned int *teacher_cindices,
                                int target_row, int tail,
                                int post_num, int pre_base, int teacher_base, int total_nn ){
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int start = 0;
    unsigned int target_index = 0;
    unsigned int pre_id;
    unsigned int teacher_id;
    CTYPE w = 0;
    CTYPE w_max = 1.0;
    char S_pre;
    char S_teacher;
    char has_a2 = 1;
    int s_upper_bound = 50;
    int t_row;

    for( unsigned int post_id = 0; post_id < post_num; post_id++ ){
        start = rptr[post_id];
        target_index = start + tid;
        teacher_id = teacher_cindices[ teacher_rptr[ post_id ] ];

        if( tid < rptr[post_id + 1] - rptr[post_id] ){
            CTYPE dw = 0;
            pre_id = cindices[ target_index ];
            w = val[ target_index ];



            S_pre = spike[ target_row*total_nn + pre_base + pre_id ];
            S_teacher = spike[ target_row*total_nn + teacher_base + teacher_id ];

            dw += (S_pre != 0)? 0.01*( w_max - w ) : 0;

            if( has_a2 && S_teacher ){
                for( int s = 0; s < 50; s++ ){
                    t_row = (target_row >= s )? target_row - s : target_row + tail - s;
                    dw += ( spike[ t_row*total_nn + pre_base  + pre_id  ] != 0 )? (-0.1)*w : 0;
                }
            }


            val[ target_index ] = w + dw;
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

