#include <stdio.h>
#include <curand_kernel.h>
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


#define noise_amp (15.0)

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
            _dg += (pre_id < 0)? 0 :spike[base+pre_id]*weight[ max_conv*post_id + i ];
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
            pre_id = cindices[ max_conv*post_id + i];
            _dg += (pre_id < 0 )? 0 : spike[base+pre_id]*weight[ max_conv*post_id + i ];
			//_dg += (spike[ base + cindices[i] ] != 0)? weight[ i ] : 0;
		}
		//__syncthreads(); // unnecessary ?
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
		mySum =  (i< max_conv              && cindices[ idx ] >= 0 )? spike[base + cindices[ idx ] ]*weight[ idx ]:0;

        idx += blockSize;
		mySum += (i+blockSize < max_conv   && cindices[ idx ] >= 0 )? spike[base + cindices[ idx ] ]*weight[ idx ]:0;

        idx += blockSize;
		mySum += (i+blockSize*2 < max_conv && cindices[ idx ] >= 0 )? spike[base + cindices[ idx ] ]*weight[ idx ]:0;

        idx += blockSize;
		mySum += (i+blockSize*3 < max_conv && cindices[ idx ] >= 0 )? spike[base + cindices[ idx ] ]*weight[ idx ]:0;


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

__global__ void update_lif_ch(CTYPE *u, CTYPE *g, CTYPE *w, CTYPE *E, CTYPE *g_bar, CTYPE *tau, CTYPE *dg_exc, CTYPE *dg_inh, curandStatePhilox4_32_10_t *state, char *spike,int *refractory_time_left ,const Neuron *Neurons, const char *type_array, const int target_row,const int base_id, const int num){
    extern __shared__ CTYPE d[];// c_num * num;
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;

    
	if(tid < num){
        CTYPE I_noise = 0;//noise_amp * curand_normal(&state[tid]);

	    unsigned int global_id = base_id + threadIdx.x + blockIdx.x*blockDim.x;
		const char type = type_array[global_id];

		const CTYPE u_rest = Neurons[type].El;
		const CTYPE u_reset = Neurons[type].Vr;
		const CTYPE u_th = Neurons[type].Vth;
		const int ref_step = (int)(Neurons[type].dt_ref/DT);

        int c_num = Neurons[type].c_num;
        //int c_num = 1;
        const int ch_base = tid * c_num;

		const CTYPE Cm = Neurons[type].Cm;
		const CTYPE Ie = Neurons[type].Ie;
		const CTYPE gL = Neurons[type].gL;
		const CTYPE gahp = Neurons[type].tau_exc;
		CTYPE u_ = u[global_id];
		CTYPE du_;
		float time_ahp = refractory_time_left[global_id];

		int spike_;

        CTYPE Isyn = 0;

        //if(tid == num - 1) printf("0:c_num = %d, g[0] = %lf, w[0] = %lf, E[0] = %lf, g_bar[0] = %lf, tau[0] = %lf\n", c_num, g[tid*c_num + 1], w[1], E[1], g_bar[1], tau[1]);
		// Amplitude of Noise Current should be modified.
		// step = 0
		{
            Isyn = 0;
            for(int c = 0; c < c_num; c++){
                //g[ ch_base + c] = g[ch_base + c];
                Isyn += g[ch_base + c]*(u_ - E[c]);
            }

			//du_ = (DT/Cm)*( Ie -gL*(u_ - u_rest) - Isyn + I_noise );
			du_ = (DT/Cm)*( Ie -gL*(u_ - u_rest) - Isyn + I_noise + gahp*__expf( -time_ahp/5.0 )*(u_reset - u_) );

            for(int c = 0; c < c_num; c++){
                g[ch_base + c] += - (DT/tau[c])*g[ch_base + c] + g_bar[c]*w[c]*( ( E[c] > u_rest )? dg_exc[global_id] : dg_inh[global_id] );
            }

			spike_ = (u_ > u_th);
            time_ahp = ( u_ > u_th )? 0 : time_ahp + DT;
			//IsRef = (spike_) ? ref_step: (IsRef > 0)?IsRef-1:0;
			u_ = (u_ > u_th)? u_reset: u_ + du_;
            //u_ = (spike)? u_+du_ + gahp*__expf( - (float)(ref_step - IsRef)/5.0 )*(u_ - u_reset): u_ + du_;
		}

		for(int step = 1; step < STEP_MAX; step++){
            Isyn = 0;
            for(int c = 0; c < c_num; c++){
                Isyn += g[ch_base + c]*(u_ - E[c]);
            }

			//du_ = (DT/Cm)*( Ie -gL*(u_ - u_rest) - Isyn + I_noise );
			du_ = (DT/Cm)*( Ie -gL*(u_ - u_rest) - Isyn + I_noise + gahp*__expf( -time_ahp/5.0 )*(u_reset - u_) );

            for(int c = 0; c < c_num; c++){
                g[ch_base + c] += - (DT/tau[c]) * g[ch_base + c];
            }

			spike_ += (u_ > u_th);
            time_ahp = ( u_ > u_th )? 0 : time_ahp + DT;
			//IsRef = (spike_)?ref_step: (IsRef > 0)?IsRef-1:0;
			//u_ = (IsRef)?u_reset: u_ + du_;
            //u_ = (spike)? u_+du_ + gahp*__expf( - (float)(ref_step - IsRef)/5.0 )*(u_ - u_reset): u_ + du_;
			u_ = (u_ > u_th)? u_reset: u_ + du_;
		}

		u[global_id] = u_;
		spike[target_row + global_id] = (spike_)?1:0;
		refractory_time_left[global_id] = (int)( time_ahp + 1 );

		dg_exc[global_id] = 0;
		dg_inh[global_id] = 0;

	}
	return;
}
__global__ void update_lif(CTYPE *u, CTYPE *g_exc, CTYPE *dg_exc, CTYPE *g_inh, CTYPE *dg_inh,CTYPE *Inoise, char *spike,int *refractory_time_left ,const Neuron *Neurons,const char *type_array, const int target_row,const int base_id, const int num){
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if(tid < num){
	    unsigned int global_id = base_id + threadIdx.x + blockIdx.x*blockDim.x;
		const char type = type_array[global_id];

		const CTYPE u_rest = Neurons[type].El;
		const CTYPE u_reset = Neurons[type].Vr;
		const CTYPE tau_exc = Neurons[type].tau_exc;
		const CTYPE tau_inh = Neurons[type].tau_inh;
		const CTYPE u_th = Neurons[type].Vth;
		const int ref_step = (int)(Neurons[type].dt_ref/DT);

		const CTYPE Cm = Neurons[type].Cm;
		const CTYPE Ie = Neurons[type].Ie;
		const CTYPE gL = Neurons[type].gL;
		CTYPE u_ = u[global_id];
		CTYPE du_;
		CTYPE g_exc_ = g_exc[global_id];
		CTYPE g_inh_ = g_inh[global_id];
		int IsRef = refractory_time_left[global_id];
		float ratio =(float)(STEP_MAX - IsRef)/(float)STEP_MAX; 
		ratio = ( ratio < 0 )?0:ratio;

		int spike_;

		// Amplitude of Noise Current should be modified.
		// step = 0
		{

			du_ = (DT/Cm)*(  Ie -gL*(u_ - u_rest) - g_exc_*(u_ - 0.) -g_inh_*(u_ - (-85.)));

			//g_exc_ += -(DT/tau_exc)*g_exc_ + ratio*dg_exc[global_id];
			//g_inh_ += -(DT/tau_inh)*g_inh_ + ratio*dg_inh[global_id];
			g_exc_ += -(DT/tau_exc)*g_exc_ + dg_exc[global_id];
			g_inh_ += -(DT/tau_inh)*g_inh_ + dg_inh[global_id];

			spike_ = (u_ > u_th);
			IsRef = (u_ > u_th) ? ref_step: (IsRef > 0)?IsRef-1:0;
			u_ = (IsRef)?u_reset: u_ + du_;
		}

		for(int step = 0; step < STEP_MAX; step++){
			du_ = (DT/Cm)*(  Ie -gL*(u_ - u_rest) - g_exc_*(u_ - 0.) -g_inh_*(u_ - (-85.)));

			g_exc_ += -(DT/tau_exc)*g_exc_;
			g_inh_ += -(DT/tau_inh)*g_inh_;

			spike_ += (u_ > u_th);
			IsRef = (u_ > u_th)?ref_step: (IsRef > 0)?IsRef-1:0;
			u_ = (IsRef)?u_reset: u_ + du_;
		}

		u[global_id] = u_;
		spike[target_row + global_id] = (spike_)?1:0;
		refractory_time_left[global_id] = IsRef;
		g_exc[global_id] = g_exc_;
		g_inh[global_id] = g_inh_;
		dg_exc[global_id] = 0;
		dg_inh[global_id] = 0;

	}
	return;
}




__global__ void update(CTYPE *u, CTYPE *g_exc, CTYPE *dg_exc, CTYPE *g_inh, CTYPE *dg_inh,CTYPE *Inoise, char *spike,int *refractory_time_left ,const Neuron *Neurons,const char *type_array,const int target_row, const int total_nn){
	unsigned int global_id = threadIdx.x + blockIdx.x*blockDim.x;
	if(global_id < total_nn){
		const char type = type_array[global_id];
		//if(type == glomerulus || type == io_cell) return;
        if( Neurons[type].dev_type != NORMAL ) return;

		const CTYPE u_rest = Neurons[type].El;
		const CTYPE u_reset = Neurons[type].Vr;
		const CTYPE tau_exc = Neurons[type].tau_exc;
		const CTYPE tau_inh = Neurons[type].tau_inh;
		const CTYPE u_th = Neurons[type].Vth;
		const int ref_step = (int)(Neurons[type].dt_ref/DT);

		const CTYPE Cm = Neurons[type].Cm;
		const CTYPE Ie = Neurons[type].Ie;
		const CTYPE gL = Neurons[type].gL;
		CTYPE u_ = u[global_id];
		CTYPE du_;
		CTYPE g_exc_ = g_exc[global_id];
		CTYPE g_inh_ = g_inh[global_id];
		int IsRef = refractory_time_left[global_id];
		float ratio =(float)(STEP_MAX - IsRef)/(float)STEP_MAX; 
		ratio = ( ratio < 0 )?0:ratio;

		int spike_;

		// Amplitude of Noise Current should be modified.
		// step = 0
		{

			du_ = (DT/Cm)*(  Ie -gL*(u_ - u_rest) - g_exc_*(u_ - 0.) -g_inh_*(u_ - (-85.)));

			//g_exc_ += -(DT/tau_exc)*g_exc_ + ratio*dg_exc[global_id];
			//g_inh_ += -(DT/tau_inh)*g_inh_ + ratio*dg_inh[global_id];
			g_exc_ += -(DT/tau_exc)*g_exc_ + dg_exc[global_id];
			g_inh_ += -(DT/tau_inh)*g_inh_ + dg_inh[global_id];

			spike_ = (u_ > u_th);
			IsRef = (u_ > u_th) ? ref_step: (IsRef > 0)?IsRef-1:0;
			u_ = (IsRef)?u_reset: u_ + du_;
		}

		for(int step = 0; step < STEP_MAX; step++){
			du_ = (DT/Cm)*(  Ie -gL*(u_ - u_rest) - g_exc_*(u_ - 0.) -g_inh_*(u_ - (-85.)));

			g_exc_ += -(DT/tau_exc)*g_exc_;
			g_inh_ += -(DT/tau_inh)*g_inh_;

			spike_ += (u_ > u_th);
			IsRef = (u_ > u_th)?ref_step: (IsRef > 0)?IsRef-1:0;
			u_ = (IsRef)?u_reset: u_ + du_;
		}

		u[global_id] = u_;
		spike[target_row + global_id] = (spike_)?1:0;
		refractory_time_left[global_id] = IsRef;
		g_exc[global_id] = g_exc_;
		g_inh[global_id] = g_inh_;
		dg_exc[global_id] = 0;
		dg_inh[global_id] = 0;

	}
	return;
}


/******************************************    Host    ***********************************************************/
__host__ void host_update(CTYPE *u, CTYPE *g_exc, CTYPE *dg_exc, CTYPE *g_inh, CTYPE *dg_inh,CTYPE *Inoise, char *spike,int *refractory_time_left ,const Neuron *Neurons,const char *type_array,const int target_row, const int total_nn,const float t, FILE *fp, int* NeuronTypeID){
	for( int global_id = 0; global_id < total_nn; global_id++){
		const char type = type_array[global_id];
		if( Neurons[type].dev_type != OUTPUT ) continue;

		const CTYPE u_rest = Neurons[type].El;
		const CTYPE u_reset = Neurons[type].Vr;
		const CTYPE tau_exc = Neurons[type].tau_exc;
		const CTYPE tau_inh = Neurons[type].tau_inh;
		const CTYPE u_th = Neurons[type].Vth;
		const int ref_step = (int)(Neurons[type].dt_ref/DT);

		const CTYPE Cm = Neurons[type].Cm;
		const CTYPE Ie = Neurons[type].Ie;
		const CTYPE gL = Neurons[type].gL;
		CTYPE u_ = u[global_id];
		CTYPE du_;
		CTYPE g_exc_ = g_exc[global_id];
		CTYPE g_inh_ = g_inh[global_id];
		int IsRef = refractory_time_left[global_id];
		float ratio =(float)(STEP_MAX - IsRef)/(float)STEP_MAX; 
		ratio = ( ratio < 0 )?0:ratio;

		int spike_;

		// Amplitude of Noise Current should be modified.
		// step = 0
		{


			du_ = (DT/Cm)*(  Ie -gL*(u_ - u_rest) - g_exc_*(u_ - 0.) -g_inh_*(u_ - (-85.)));

			//g_exc_ += -(DT/tau_exc)*g_exc_ + ratio*dg_exc[global_id];
			//g_inh_ += -(DT/tau_inh)*g_inh_ + ratio*dg_inh[global_id];
			g_exc_ += -(DT/tau_exc)*g_exc_ + dg_exc[global_id];
			g_inh_ += -(DT/tau_inh)*g_inh_ + dg_inh[global_id];

			spike_ = (u_ > u_th);
			IsRef = (u_ > u_th) ? ref_step: (IsRef > 0)?IsRef-1:0;
			u_ = (IsRef)?u_reset: u_ + du_;
		}

		for(int step = 0; step < STEP_MAX; step++){
			du_ = (DT/Cm)*(  Ie -gL*(u_ - u_rest) - g_exc_*(u_ - 0.) -g_inh_*(u_ - (-85.)));

			g_exc_ += -(DT/tau_exc)*g_exc_;
			g_inh_ += -(DT/tau_inh)*g_inh_;

			spike_ += (u_ > u_th);
			IsRef = (u_ > u_th)?ref_step: (IsRef > 0)?IsRef-1:0;
			u_ = (IsRef)?u_reset: u_ + du_;
		}

		u[global_id] = u_;

		spike[global_id] = spike_;
		if(spike_)fprintf(fp, "%f\t%d\t%d\n", t, global_id, NeuronTypeID[type]);

		refractory_time_left[global_id] = IsRef;
		g_exc[global_id] = g_exc_;
		g_inh[global_id] = g_inh_;
		dg_exc[global_id] = 0;
		dg_inh[global_id] = 0;

	}
	fflush(fp);
	return;
}

__host__ void host_spike_propagation(const int pre_type, const int postNum, const int post_base_id, CTYPE *dg, const int max_conv, const int *cindices,  const CTYPE *weight, const CTYPE w_bar, const int target_row, const int target_block,  const Sim_cond_lif_exp *Dev){
    for(int post_id = 0; post_id < postNum; post_id++){
	    unsigned int global_post_id = post_base_id + post_id;
	    CTYPE _dg = 0;

	    unsigned int base = 0;
	    char *spike;
	    unsigned int dev_id = 0;
	    spike = Dev[dev_id].print_arg[ target_block ].spike;
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

