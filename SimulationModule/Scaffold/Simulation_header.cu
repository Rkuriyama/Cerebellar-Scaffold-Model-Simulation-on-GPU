#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>
#include "struct_enum_def.h"
//#include "Simulation_header.h"
#include "option.h"
//#include <curand_kernel.h>
/*
変更履歴
論理演算をそのまま利用するようにしてみた。あまり実行時間に変化なし。
for(; i < rptr[];) => for(; i < end;) : ちょい速くなった？わからん。0.2secとかその程度

nextPow2関数を利用.
reduceを複数ブロック用と1ブロック用で分けた.
NT=NBをやめた
pf_PCで2stream使うことにし、1repeatで2つ呼ぶようにした。
最終段のreduceを呼ぶ前に、cudaStreamSynchronize()を呼んでみた。


1ms分まとめると影響が大きくなり、周波数によっては発振してしまう現象がみられた。
まず、refractory time と被っていた分に関して、(1ms-残ってた秒数)/1msを掛けた。
それでも発振することがあったので、勝手にスパイクの影響を0.8倍することで調整した。(weightを0.8倍すれば良いのでは？)
なお、0.8は適当に決めた tau_と1msで計算すれば良い気はする。

*/



__global__ void Philox_setup_kernel(unsigned long long seed, curandStatePhilox4_32_10_t *state, unsigned int N){
	unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i < N)
		curand_init(seed, i, 0, &state[i]);
	return;
}

__global__ void Philox_generate_normal(CTYPE *a, CTYPE mean, CTYPE std_div, curandStatePhilox4_32_10_t *state, unsigned int N){
	unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i < N)
		a[i] = std_div*curand_normal(&state[i])+mean;
	return;
}


__global__ void Philox_generate_uniform4(CTYPE *a, curandStatePhilox4_32_10_t *state, unsigned int N){
	unsigned int i = threadIdx.x*4 + (blockDim.x*4)*blockIdx.x;
	unsigned int state_id = threadIdx.x + blockDim.x*blockIdx.x;
	float4 r;
	if(i < N){
		r = curand_uniform4(&state[state_id]);
		if(i   < N) a[i]   = r.x;
		if(i+1 < N) a[i+1] = r.y;
		if(i+2 < N) a[i+2] = r.z;
		if(i+3 < N) a[i+3] = r.w;
	}
	return;
}


__global__ void Philox_generate_uniform(CTYPE *a, curandStatePhilox4_32_10_t *state, unsigned int N){
	unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
	float4 r;
	if(i < N){
		r = curand_uniform4(&state[i]);
		 a[i] = r.x;
	}
	return;
}

__global__ void Philox_generate_uniform4_v2(CTYPE *a, curandStatePhilox4_32_10_t *state, unsigned int N){
	unsigned int i = threadIdx.x + (blockDim.x*4)*blockIdx.x;
	unsigned int state_id = threadIdx.x + blockDim.x*blockIdx.x;
	float4 r;
	if(i < N){
		r = curand_uniform4(&state[state_id]);
		a[i] = (i   < N)? r.x : a[i];
		a[i+blockDim.x]   = (i+blockDim.x < N)?   r.y : a[i+blockDim.x];
		a[i+2*blockDim.x] = (i+2*blockDim.x < N)? r.z : a[i+2*blockDim.x];
		a[i+3*blockDim.x] = (i+3*blockDim.x < N)? r.w : a[i+3*blockDim.x];
	}
	return;
}

__global__ void spike_propagation(const int post_base_id, const int postNum, CTYPE *dg, const int max_conv, const int *cindices,  const CTYPE *weight, const CTYPE w_bar, const char *spike, const int base){
	unsigned int post_id = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int global_post_id = post_id + post_base_id;
    int pre_id;
    //int idx = post_id;
	CTYPE _dg = 0;

	if(post_id < postNum){
        for(int i = 0; i < max_conv; i++){
            //pre_id = cindices[idx];
            //_dg += (pre_id < 0)? 0 :spike[base+pre_id]*weight[ idx ];
            //idx += postNum;
            pre_id = cindices[max_conv*post_id + i];
            _dg += (pre_id < 0 || !spike[base+pre_id] )? 0 : weight[ max_conv*post_id + i ];
        }
		dg[global_post_id] += _dg*w_bar;
	}
	return;
}


__global__ void spike_propagation_mThreads(const int post_base_id, const int postNum, CTYPE *dg, const int max_conv, const int *cindices,  const CTYPE *weight, const CTYPE w_bar, const char *spike, const int base, const unsigned int threadsPerNeuron){
	unsigned int post_id = (threadIdx.x / threadsPerNeuron) +  (blockDim.x / threadsPerNeuron) *blockIdx.x;
	unsigned int global_post_id = post_base_id + post_id;

	unsigned int area = threadIdx.x & (threadsPerNeuron-1);
	int start = (max_conv + threadsPerNeuron - 1) / threadsPerNeuron * area;
	int end = (max_conv + threadsPerNeuron - 1 ) / threadsPerNeuron * (area+1);
    end = ( area != threadsPerNeuron - 1 )? end : max_conv;

    int pre_id;
	CTYPE _dg = 0;


	if(post_id < postNum){
		for(int i = start; i < end; i++)
		{
             // idx  + if
            //int idx =  max_conv*post_id + i;
            //pre_id = cindices[ idx ];
            //_dg += (pre_id < 0 || !spike[base+pre_id] ) ? 0: weight[ idx ];

            // if
            pre_id = cindices[ max_conv*post_id + i ];
            _dg += (pre_id < 0 || !spike[base+pre_id] ) ? 0: weight[ max_conv * post_id + i ];

            // mul
            //pre_id = cindices[ max_conv*post_id + i ];
            //_dg += (pre_id >= 0 )? spike[base+pre_id] * weight[ max_conv*post_id + i ] : 0;
		}
		__syncthreads(); // unnecessary ?
		for(int offset = 1; offset < threadsPerNeuron; offset <<= 1)
	        	_dg += __shfl_down_sync(0xffffffff, _dg, offset, warpSize);
		if( !area ) dg[global_post_id] += _dg*w_bar;
	}
	return;
}

// K40ではreduce5でなければいけないっぽい。
unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

__global__ void init_and_reduce_ELL( const int *cindices, const CTYPE *weight, const int max_conv, const char *spike, const int base, CTYPE *out, const unsigned int postNum){
	extern __shared__ volatile CTYPE sdata[];
	unsigned int blockSize = blockDim.x;
	unsigned int i = threadIdx.x + (blockSize*4) * blockIdx.x;
	unsigned int tid = threadIdx.x;
	CTYPE mySum;
	unsigned int idx;


	for(int post_id = 0; post_id < postNum; post_id++){

        idx = max_conv*post_id + i;
		mySum =  (i< max_conv              && cindices[ idx ] >= 0 &&  spike[base + cindices[ idx ] ])?weight[ idx ]:0;

        idx += blockSize;
		mySum += (i+blockSize < max_conv   && cindices[ idx ] >= 0 &&  spike[base + cindices[ idx ] ])?weight[ idx ]:0;

        idx += blockSize;
		mySum += (i+blockSize*2 < max_conv && cindices[ idx ] >= 0 &&  spike[base + cindices[ idx ] ])?weight[ idx ]:0;
        
        idx += blockSize;
		mySum += (i+blockSize*3 < max_conv && cindices[ idx ] >= 0 &&  spike[base + cindices[ idx ] ])?weight[ idx ]:0;


		sdata[tid] = mySum;
		__syncthreads();

		for(unsigned int s=blockDim.x/2; s>32; s>>=1){
			sdata[tid] =( mySum = (tid < s) ? mySum + sdata[tid+s]: mySum );
			__syncthreads();
		}

		if ( tid < 32 ){ // ここのif文: warp間での動作は共通だからそこまでロスなし？
        		// Fetch final intermediate sum from 2nd warp
			mySum += (blockSize >=  64)? sdata[tid + 32] : 0;
        		// Reduce final warp using shuffle
        		mySum +=__shfl_down_sync(0xffffffff, mySum, 16);
        		mySum +=__shfl_down_sync(0xffffffff, mySum, 8) ;
        		mySum +=__shfl_down_sync(0xffffffff, mySum, 4) ;
        		mySum +=__shfl_down_sync(0xffffffff, mySum, 2) ;
        		mySum +=__shfl_down_sync(0xffffffff, mySum, 1) ;	
        		//for (int offset = warpSize/2; offset > 0; offset /= 2) {
        		//    mySum += __shfl_down_sync(0xffffffff,mySum, offset);
        		//}
    		}
		if(tid==0)out[ post_id*gridDim.x + blockIdx.x] = mySum; // ここのif文
	}
}

template < unsigned int previousBlockSize > __global__ void reduce_last_phase( const CTYPE *in,  CTYPE *out, const int postNum ){
	unsigned int i = threadIdx.x +  blockDim.x*blockIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int postId =  (blockIdx.x << 2) + (tid >> 5); // 1ブロック128スレッド内で4 post synaptic cells 分の計算をしている。
	CTYPE mySum;

	if( postId < postNum){
		switch(previousBlockSize){
			case 2:
				mySum = ( (tid&31) < 2 )? in[i] : 0;
				break;
			case 4:
				mySum = ( (tid&31) < 4 )? in[i] : 0;
				break;
			case 8:
				mySum = ( (tid&31) < 8 )? in[i] : 0;
				break;
			case 16:
				mySum = ((tid&31) < 16)? in[i] : 0;
				break;
			case 32:
				mySum = in[i];
				break;
			case 64:
				mySum = in[2*i] + in[2*i+1];
				break;
			case 128:
				mySum = in[4*i] + in[4*i+1] + in[4*i+2] + in[4*i+3];
				break;
		}

		__syncthreads();

	        mySum += __shfl_down_sync(0xffffffff, mySum, 16, warpSize);
	        mySum += __shfl_down_sync(0xffffffff, mySum, 8, warpSize);
	        mySum += __shfl_down_sync(0xffffffff, mySum, 4, warpSize);
	        mySum += __shfl_down_sync(0xffffffff, mySum, 2, warpSize);
	        mySum += __shfl_down_sync(0xffffffff, mySum, 1, warpSize);

		//for (int offset = warpSize >> 1; offset; offset >>= 1) {
	        //     mySum += __shfl_down_sync(0xffffffff, mySum, offset, warpSize);
	    	//}

		//  if( !(tid%warpSize) ) out[ ( blockIdx.x*(blockDim.x/warpSize) ) + (tid/warpSize)  ] = mySum; // ここ遅い。どうにかビット演算系で済ませられないか。というか、warpのうち1つしか起動しないのはとても効率悪いがそれは後で考える。
		//  blockDim.x を128固定, warpSizeはGV100でも32だった。これも固定されてるとして。
		//  tid%warpSize ->  if( !(tid & 0x0000001f) ) out[ (blockIdx.x << 2) + (tid >> 5 ) ]
		if( !(tid & 31) ) out[postId] = mySum;
	}
}
__global__ void add_tmp_to_dg( CTYPE *tmp, CTYPE *dg, const CTYPE w_bar, const int post_base, const int postNum ){
	unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < postNum) dg[post_base + i] +=  tmp[i]*w_bar ;
	return;
}

// 次にやること	: reduce_last_phase と add_tmp_to_dg の統合

__host__ void spike_propagation_PR(CTYPE *out, CTYPE *tmp, const int max_conv, const int pre_base_id,const int postNum,const int post_base_id, CTYPE *dg, const int *cindices, const CTYPE *weight, const CTYPE w_bar, const char *spike,const int row,  const int total_nn, cudaStream_t stream){
	//最初に入力行列を作成。
	unsigned int NPOW;
	int N;
	int NT;
	int NB;

	N = max_conv;
	NPOW = nextPow2( N );
	NT = 128;
	NB = (NPOW+NT*4-1 )/(NT*4);

    init_and_reduce_ELL<<< NB, NT, NT*sizeof(CTYPE), stream >>>( cindices, weight, max_conv, spike, row + pre_base_id, out, postNum);


	switch(NB){
		case 2:
			reduce_last_phase<2><<< postNum, 128, 0, stream  >>>( out, tmp, postNum);
			break;
		case 4:
			reduce_last_phase<4><<< postNum, 128, 0, stream  >>>( out, tmp, postNum);
			break;
		case 8:
			reduce_last_phase<8><<< postNum, 128, 0, stream  >>>( out, tmp, postNum);
			break;
		case 16:
			reduce_last_phase<16><<< postNum, 128, 0, stream  >>>( out, tmp, postNum);
			break;
		case 32:
			reduce_last_phase<32><<< postNum, 128, 0, stream  >>>( out, tmp, postNum);
			break;
		case 64:
			reduce_last_phase<64><<< postNum, 128, 0, stream  >>>( out, tmp, postNum);
			break;
		case 128:
			reduce_last_phase<128><<< postNum, 128, 0, stream  >>>( out, tmp, postNum);
			break;
	}
	if(NB != 1){
		add_tmp_to_dg<<< (postNum+127)/128, 128, 0, stream >>>(tmp, dg, w_bar, post_base_id, postNum);
	}else{
		add_tmp_to_dg<<< (postNum+127)/128, 128, 0, stream >>>(out, dg, w_bar, post_base_id, postNum);
	}
	return;
}

__global__ void update_lif_ch(CTYPE *u,  CTYPE *dg_exc, CTYPE *dg_inh,CTYPE *Inoise, char *spike,int *refractory_time_left ,const Neuron target, const int target_row){
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    CTYPE *g = target.g;
    const CTYPE *w = target.w;
    const CTYPE *E = target.E;
    const CTYPE *g_bar = target.g_bar;
    const CTYPE *tau = target.tau;
    const int base_id = target.base_id;
    const int num = target.num;

	if(tid < num){
	    unsigned int global_id = base_id + threadIdx.x + blockIdx.x*blockDim.x;
		//const char type = type_array[global_id];

		const CTYPE u_rest = target.El;
		const CTYPE u_reset = target.Vr;
		const CTYPE u_th = target.Vth;
		const int ref_step = (int)(target.dt_ref/DT);

        int c_num = target.c_num;
        const int ch_base = tid * c_num;

		const CTYPE Cm = target.Cm;
		const CTYPE Ie = target.Ie;
		const CTYPE gL = target.gL;
		CTYPE u_ = u[global_id];
		CTYPE du_;
		int IsRef = refractory_time_left[global_id];

		int spike_;

        CTYPE Isyn = 0;

		{
            Isyn = 0;
            for(int c = 0; c < c_num; c++){
                Isyn += g[ch_base + c]*(u_ - E[c]);
            }

			du_ = (DT/Cm)*( Ie -gL*(u_ - u_rest) - Isyn );

            for(int c = 0; c < c_num; c++){
                g[ch_base + c] += -(DT/tau[c])*g[ch_base + c] + g_bar[c]*w[c]*( ( E[c] > u_rest )? dg_exc[global_id] : dg_inh[global_id] );
            }

			spike_ = (u_ > u_th);
			IsRef = (u_ > u_th) ? ref_step: (IsRef > 0)?IsRef-1:0;
			u_ = (IsRef)?u_reset: u_ + du_;
		}

		for(int step = 1; step < STEP_MAX; step++){
            Isyn = 0;
            for(int c = 0; c < c_num; c++){
                Isyn += g[ch_base + c]*(u_ - E[c]);
            }

			du_ = (DT/Cm)*(  Ie -gL*(u_ - u_rest) - Isyn);

            for(int c = 0; c < c_num; c++){
                g[ch_base + c] += - (DT/tau[c]) * g[ch_base + c];
            }

			spike_ += (u_ > u_th);
			IsRef = (u_ > u_th)?ref_step: (IsRef > 0)?IsRef-1:0;
			u_ = (IsRef)?u_reset: u_ + du_;
		}

		u[global_id] = u_;
		spike[target_row + global_id] = (spike_)?1:0;
		refractory_time_left[global_id] = IsRef;

		dg_exc[global_id] = 0;
		dg_inh[global_id] = 0;

	}
	return;
}


/******************************************    Host    ***********************************************************/
__host__ void host_update_lif_ch(CTYPE *u, CTYPE *dg_exc, CTYPE *dg_inh,CTYPE *Inoise, char *spike,int *refractory_time_left , Neuron target, const int target_row, const float t, FILE *fp, int* NeuronTypeID){


    CTYPE *g = target.g;
    const CTYPE *w = target.w;
    const CTYPE *E = target.E;
    const CTYPE *g_bar = target.g_bar;
    const CTYPE *tau = target.tau;
    const int base_id = target.base_id;
    const int num = target.num;
    const int c_num = target.c_num;

	const CTYPE u_rest = target.El;
	const CTYPE u_reset = target.Vr;
	const CTYPE u_th = target.Vth;
	const int ref_step = (int)(target.dt_ref/DT);
	const CTYPE Cm = target.Cm;
	const CTYPE Ie = target.Ie;
	const CTYPE gL = target.gL;

    for(int global_id = base_id; global_id < base_id + num; global_id++){
        const int ch_base = global_id * c_num;
		CTYPE u_ = u[global_id];
		CTYPE du_ = 0;
		int IsRef = refractory_time_left[global_id];

		int spike_ = 0;

        CTYPE Isyn = 0;

        CTYPE du_sum = 0; //debug

		{
            Isyn = 0;
            for(int c = 0; c < c_num; c++){
                Isyn += g[ch_base + c]*(u_ - E[c]);
            }

			du_ = (DT/Cm)*( Ie -gL*(u_ - u_rest) - Isyn );

            for(int c = 0; c < c_num; c++){
                g[ch_base + c] += -(DT/tau[c])*g[ch_base + c] + g_bar[c]*w[c]*( ( E[c] > u_rest )? dg_exc[global_id] : dg_inh[global_id] );
            }

			spike_ = (u_ > u_th);
			IsRef = (u_ > u_th) ? ref_step: (IsRef > 0)?IsRef-1:0;
			u_ = (IsRef > 0)?u_reset: u_ + du_;
            du_sum += (IsRef > 0)? 0:  fabs(du_);
		}

		for(int step = 1; step < STEP_MAX; step++){
            Isyn = 0;
            for(int c = 0; c < c_num; c++){
                Isyn += g[ch_base + c]*(u_ - E[c]);
            }

			du_ = (DT/Cm)*(  Ie -gL*(u_ - u_rest) - Isyn);

            for(int c = 0; c < c_num; c++){
                g[ch_base + c] += - (DT/tau[c]) * g[ch_base + c];
            }

			spike_ += (u_ > u_th);
			IsRef = (u_ > u_th)?ref_step: (IsRef > 0)?IsRef-1:0;
			u_ = (IsRef > 0)?u_reset: u_ + du_;
            du_sum += (IsRef > 0)? 0: fabs(du_);
		}

		u[global_id] = u_;

		spike[global_id] = (spike_ > 0)?1:0;
		if(spike_)fprintf(fp, "%f\t%d\t%d\n", t, global_id, NeuronTypeID[ target.type ]);

		refractory_time_left[global_id] = IsRef;

		dg_exc[global_id] = 0;
		dg_inh[global_id] = 0;

	}
	return;
}


__host__ void host_spike_propagation(const int pre_type, const int postNum, const int post_base_id, CTYPE *dg, const int max_conv, const int *cindices,  const CTYPE *weight, const CTYPE w_bar, const int target_row, const int target_block,  const Sim_cond_lif_exp *Dev){
    for(int post_id = 0; post_id < postNum; post_id++){
	    unsigned int global_post_id = post_base_id + post_id;
	    CTYPE _dg = 0;

	    unsigned int base = 0;
	    char *spike;
	    unsigned int dev_id = 0;
	    spike = Dev[dev_id].print_arg[ target_block ].host_spike;
	    base = Dev[dev_id].total_neuron_num*target_row + Dev[dev_id].neurons[pre_type].base_id;

	    for(unsigned int i = 0; i < max_conv; i++){
            if( cindices[max_conv*post_id + i] < 0) break;
	    	if( cindices[max_conv*post_id + i] >= Dev[dev_id].end[pre_type] ){
	    		dev_id++;
	    	    spike = Dev[dev_id].print_arg[ target_block ].spike;
	    		base = Dev[dev_id].total_neuron_num*target_row + Dev[dev_id].neurons[pre_type].base_id;
	    	}

	    	//unsigned int pre_offset = cindices[i];
	    	_dg += (spike[ base + cindices[ max_conv*post_id + i ] - Dev[dev_id].start[pre_type] ])*weight[max_conv*post_id + i];
	    }
	    dg[global_post_id] += _dg*w_bar;
    }
    return;
}

