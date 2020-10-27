#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <omp.h>

#include "init_params.h"
#include "Simulation_header.h"
#include "option.h"

#include "UserInputFunctions.h"
#include "SynapticPlasticity.h"



#define DEBUG 0
#define PROGRESS 0
#define DEBUG_T 500
#define DEBUG_LOG "dlog.log"
#define MEASURE 1

#define PRINT 1
#define PRINT_T 50

#define DEV_NUM 2
#define TRIALS (1)


pthread_barrier_t ready[2];
pthread_barrier_t A_done[2];
pthread_barrier_t B_done[2];

void *SimulationOnCPU(void *p){
	cpu_sim_thread_val *pv = (cpu_sim_thread_val *)p;
	int count = 0;
	Sim_cond_lif_exp *h = &pv->Host;
	int id = pv->id;
	int n;
//    char file_name[50]; FILE *fp;
//    sprintf(file_name, "spike_output/spike_host%d.dat", id); fp = fopen(file_name, "w");
	while(true){
		pthread_barrier_wait(&ready[id]);
		pthread_barrier_wait(&A_done[id]);
		fprintf(stderr,"CPU_Simulation:%d\n", count);
        n = pv->n  - pv->delay_max_row;
		for(int target_row = 0; target_row < pv->delay_max_row; target_row++){
			for(int i = h->gpu_connections_num; i < TotalNumOfConnectivityTypes; i++){
	        	Connectivity *c = &h->connectivities[i];
                int preType = c->preType;
                int postType = c->postType;
                Neuron *preNeuron = &h->neurons[preType];
                Neuron *postNeuron = &h->neurons[postType];
                CTYPE *dg = (c->initial_weight > 0)?h->dg_exc:h->dg_inh;
                int delay = (int)(c->delay);
                int read_row = target_row - delay;
				int target_block = (read_row < 0)? (id+1)%2: id;
                read_row = (target_row - delay >= 0)? read_row: pv->delay_max_row + target_row - delay ;

				host_calculate_current_diff( preType, postNeuron->num, postNeuron->base_id,
						dg, h->refractory_time_left , c->host_rptr, c->cindices,  c->val,
						read_row, target_block, pv->Dev);
			}
			host_update(h->u, h->g_exc, h->dg_exc, h->g_inh, h->dg_inh, h->Inoise, h->spike, h->refractory_time_left, h->neurons, h->type, target_row, h->neuron_num, n+target_row, pv->fp, pv->NeuronTypeID);
		}
		count++;
		fflush(pv->fp);
		pthread_barrier_wait(&B_done[id]);
	}
}


void* print_spike_train(void *p){
    pthread_val *pv = (pthread_val *)p;
    int delay_max_row = pv->delay_max_row;
    int total_nn = pv->total_nn;
    char *spike = pv->spike;
    char *type = pv->type;
    int n;
	int id = pv->id;
	cudaSetDevice(pv->dev_id);

	while(true){
	    pthread_barrier_wait(&ready[id]);
        n = pv->n  - delay_max_row;
        CUDA_SAFE_CALL( cudaMemcpyAsync(pv->host_spike, pv->from_spike, sizeof(char)*delay_max_row*pv->total_nn, cudaMemcpyDeviceToHost, pv->stream ));
        //CUDA_SAFE_CALL( cudaMemsetAsync( pv->from_spike , 0, sizeof(char)*delay_max_row*pv->total_nn, pv->stream ) );
        cudaStreamSynchronize(pv->stream);
	    memcpy(pv->spike, pv->host_spike, sizeof(char)*delay_max_row*pv->total_nn);
	    fprintf(stderr,"print_spike_train %d  id:%d own_neuron_num: %d\n", n, pv->id, pv->own_neuron_num);
	    pthread_barrier_wait(&A_done[id]);

	    for(int step = 0 ; step < delay_max_row;step++){
        	CTYPE t = (n+step);
        	for(int i=0;i<pv->own_neuron_num;i++){
            	if(spike[step*total_nn + i] !=0){
		            //fprintf(pv->fp, "%f\t%d\t%d\n",t, i - pv->neurons[type[i]].base_id + pv->start[type[i]] + pv->host_neurons[type[i]].base_id, pv->NeuronTypeID[ type[i] ] );
		            fprintf(pv->fp, "%f\t%d\t%d\n",t, i  + pv->start[type[i]] - pv->neurons[type[i]].base_id, pv->NeuronTypeID[ type[i] ] );
		            //fprintf(pv->fp, "%f\t%d\t%d\n",t, i, pv->NeuronTypeID[ type[i] ] );
		        }
            }
            fflush(pv->fp);
      	}
	    pthread_barrier_wait(&B_done[id]);
	}
}
/*
void print_spike_train(int *spike, int *type ,int n, int delay_max_row, int total_nn, FILE *fp){
        n-=delay_max_row;
        for(int step = 0 ; step < delay_max_row;step++){
                CTYPE t = DT*(n+step);
                for(int i=0;i<total_nn;i++){
                        if(spike[step*total_nn + i] !=0)fprintf(fp, "%f\t%d\t%d\n",t,i,type[i]);
                }
        }
        return;
}
*/

void print_param(FILE *fp, CTYPE *p, int n, int total_nn){
        for(int j = 0; j < DEBUG_T; j++){
                fprintf(fp, "%d", n-DEBUG_T+j+1);
                for(int i=0; i < total_nn; i++){
                        fprintf(fp, "\t%f",p[i + j*total_nn]);
                }
                fprintf(fp, "\n");
        }
        return;
}

__global__ void Modifie( unsigned int *rptr, unsigned int *cindices, unsigned int rptr_0, unsigned int width, unsigned int post_sn_num,
                         unsigned int neuron_num, unsigned int pre_sn_start, unsigned int pre_sn_end, unsigned int pre_sn_base,
                         unsigned int pre_neuron_num, unsigned int pre_pre_sn_start, unsigned int pre_pre_sn_base,
                         unsigned int next_neuron_num, unsigned int next_pre_sn_start, unsigned int next_pre_sn_base){
        int tid = threadIdx.x + blockIdx.x*blockDim.x;

        if(tid < width){
                //cindices[tid] = (cindices[tid] < pre_sn_start)?  neuron_num + pre_pre_sn_base + (cindices[tid] - pre_pre_sn_start) - pre_sn_base :
                //              (pre_sn_end <= cindices[tid]) ?  neuron_num + pre_neuron_num + next_pre_sn_base + (cindices[tid] - next_pre_sn_start) - pre_sn_base:
                //                              (cindices[tid] - pre_sn_start);

                if( cindices[tid] < pre_sn_start) cindices[tid] = neuron_num + pre_pre_sn_base + (cindices[tid] - pre_pre_sn_start) - pre_sn_base;
                else if( cindices[tid] >= pre_sn_start && cindices[tid] < pre_sn_end ) cindices[tid] = cindices[tid] - pre_sn_start;
                else if( cindices[tid] >= pre_sn_end) cindices[tid] = neuron_num + pre_neuron_num + next_pre_sn_base + (cindices[tid] - next_pre_sn_start) - pre_sn_base;
                else printf("%u, %u\n",cindices[tid], pre_sn_start);
        }

        if(tid <= post_sn_num){
                rptr[tid] = rptr[tid] - rptr_0;
        }
        return;
}


void Initialization( Sim_cond_lif_exp *Dev, cpu_sim_thread_val **Host_sim, int *NeuronTypeID, int *ConnectivityTypeID, Neuron *host_Neurons, Connectivity *host_Connectivities, InputFunctionsStruct *host_InputStimList, int delay_max_row, int total_nn){

    time_t seed;
    time(&seed);

	int gpu_neurons_num = 0;
	int gpu_connections_num = 0;

	for(int i = 0; i < TotalNumOfCellTypes; i++){
		if (host_Neurons[i].dev_type == NORMAL){
			gpu_neurons_num++;
		}
	}
	printf("%d\n", gpu_neurons_num);
	for(int i = 0; i < TotalNumOfConnectivityTypes; i++){
		if( host_Neurons[ host_Connectivities[i].postType ].dev_type == NORMAL ){
			gpu_connections_num++;
		}
	}

	int *ReverseNeuronTypeID;
	ReverseNeuronTypeID = (int *)malloc( sizeof(int)* TotalNumOfCellTypes);
	for(int i = 0; i < TotalNumOfCellTypes; i++){
		ReverseNeuronTypeID[ NeuronTypeID[i] ] = i;
	}

	/*
	printf("%d\n", gpu_connections_num);
	printf("%p\n", host_Connectivities);
	printf("%p\n", host_InputStimList);
	*/

    /// split Neurons
    for (int dev_id = 0; dev_id < DEV_NUM; dev_id++){
        cudaSetDevice(dev_id);
        Neuron *neurons;
        neurons = Dev[dev_id].neurons;
        if(dev_id > 0){
                Dev[dev_id].pre_neuron_num = Dev[dev_id-1].neuron_num;
                Dev[dev_id].total_neuron_num += Dev[dev_id].pre_neuron_num;
                Dev[dev_id].device_base = Dev[dev_id-1].device_base + Dev[dev_id-1].neuron_num;
        }

                
        for( int i = 0; i < TotalNumOfCellTypes; i++){
            Neuron h_n = host_Neurons[i];
            neurons[i].type = h_n.type;
            neurons[i].base_id = Dev[dev_id].neuron_num;
			neurons[i].duplicate = h_n.duplicate;

			if(neurons[i].duplicate){
				fprintf(stderr, "duplicate%d\n", i);
				neurons[i].num = h_n.num;
			}else{
            	neurons[i].num = (h_n.num + DEV_NUM - 1)/DEV_NUM;
               	if (dev_id == DEV_NUM-1){
                    neurons[i].num =  h_n.num - (DEV_NUM-1)*neurons[i].num;
                }
			}

            neurons[i].Cm = h_n.Cm;
            neurons[i].tau_m = h_n.tau_m;
            neurons[i].El = h_n.El;
            neurons[i].dt_ref = h_n.dt_ref;
            neurons[i].Ie = h_n.Ie;
            neurons[i].Vr = h_n.Vr;
            neurons[i].Vth = h_n.Vth;
            neurons[i].tau_exc = h_n.tau_exc;
            neurons[i].tau_inh = h_n.tau_inh;
            neurons[i].gL = h_n.gL;

			if(neurons[i].duplicate){
				Dev[dev_id].start[i] = 0;
				Dev[dev_id].end[i] = neurons[i].num;
			}else{
            	Dev[dev_id].start[i] = (dev_id > 0)?Dev[dev_id-1].end[i]:0;
            	Dev[dev_id].end[i] = Dev[dev_id].start[i] + neurons[i].num;
			}

            Dev[dev_id].neuron_num += neurons[i].num;
            fprintf(stderr, "%s: %d-%d  %d-%d\n", host_Neurons[i].filename, neurons[i].base_id, Dev[dev_id].neuron_num, Dev[dev_id].start[i], Dev[dev_id].end[i]);
        }
        Dev[dev_id].total_neuron_num += Dev[dev_id].neuron_num;
        if(dev_id > 0){
            Dev[dev_id-1].next_neuron_num = Dev[dev_id].neuron_num;
            Dev[dev_id-1].total_neuron_num += Dev[dev_id-1].next_neuron_num;
        }

        CUDA_SAFE_CALL(cudaMalloc(&Dev[dev_id].dev_neurons,sizeof(Neuron)*TotalNumOfCellTypes));
        CUDA_SAFE_CALL(cudaMemcpy(Dev[dev_id].dev_neurons, neurons, sizeof(Neuron)*TotalNumOfCellTypes, cudaMemcpyHostToDevice));
		Dev[dev_id].gpu_neurons_num = Dev[dev_id].neurons[ gpu_neurons_num ].base_id;
		Dev[dev_id].gpu_connections_num = gpu_connections_num;
    }

    for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
        fprintf(stderr, "Dev %d - %d neurons\n", dev_id, Dev[dev_id].neuron_num );
    }


    /// split InputStim
    for(int stim_id = 0; stim_id < INPUT_STIM_NUM; stim_id++){
        int start = 0;
        int end = 0;
        unsigned long long seed;
        for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
            cudaSetDevice(dev_id);
            InputFunctionsStruct *dev_stim = &(Dev[dev_id].InputStimList[stim_id]);
            dev_stim->type = host_InputStimList[stim_id].type;
            dev_stim->base_id = Dev[dev_id].neurons[dev_stim->type].base_id;
            dev_stim->func_id = host_InputStimList[stim_id].func_id;


			if( Dev[dev_id].neurons[dev_stim->type].duplicate ){
				dev_stim->num = host_InputStimList[stim_id].num;
				seed = (dev_id == 0) ? (unsigned long long) time(NULL) : seed;
			}else{
            	start = end;
	            while( end < host_InputStimList[stim_id].num && host_InputStimList[stim_id].IdList[end] < Dev[dev_id].end[dev_stim->type] ){
                    host_InputStimList[stim_id].IdList[end] -= Dev[dev_id].start[dev_stim->type];
        	        end++;
	            }
               	dev_stim->num = end - start;
			}
			if( dev_stim->num != 0){
            	CUDA_SAFE_CALL( cudaMalloc( &(dev_stim->IdList), sizeof(unsigned int)*(dev_stim->num)) );
                CUDA_SAFE_CALL( cudaMemcpy( dev_stim->IdList, &host_InputStimList[stim_id].IdList[start], sizeof(unsigned int)*(dev_stim->num), cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL( cudaMalloc( &(dev_stim->state), sizeof(curandStatePhilox4_32_10_t)*(dev_stim->num)) );
                Philox_setup_kernel<<< (dev_stim->num + 255)/256, 256 >>>( seed , dev_stim->state, dev_stim->num);
			}else{
				dev_stim->IdList = NULL;
				dev_stim->state = NULL;
			}
        }
    }


    // Split Connectivities format array
    for(int i = 0; i < gpu_connections_num; i++){
        int start = 0;
        int end = 0;
        int width = 0;
        for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
            cudaSetDevice(dev_id);
            Connectivity *con = &(Dev[dev_id].connectivities[i]);

            con->type = host_Connectivities[i].type;
            con->preType = host_Connectivities[i].preType;
            con->postType = host_Connectivities[i].postType;
    
            con->preNum = Dev[dev_id].neurons[con->preType].num;
            con->postNum = Dev[dev_id].neurons[con->postType].num;


            con->initial_weight = host_Connectivities[i].initial_weight;
            con->delay = host_Connectivities[i].delay;
            con->max_conv = host_Connectivities[i].max_conv;
			con->pr = 0;

			if( con->max_conv > 5000  ){
			    con->pr = -1; // infinite
			}else if( con->max_conv > 1500 ){
			    for(int pow2 = 1; con->postNum*pow2 < CUDA_CORES && pow2 <= DEV_NUM; pow2*=2 )con->pr = pow2;
			}
                        
            end = start + con->postNum;
            width = host_Connectivities[i].host_rptr[end] - host_Connectivities[i].host_rptr[start];

            /// CSR
            con->host_rptr = (unsigned int*)malloc(sizeof(unsigned int)*(con->postNum+1));
            CUDA_SAFE_CALL( cudaMalloc( &(con->rptr), sizeof(unsigned int)*(con->postNum+1)) );
            CUDA_SAFE_CALL( cudaMalloc( &(con->cindices), sizeof(unsigned int)*width) );
            CUDA_SAFE_CALL( cudaMalloc( &(con->val), sizeof(CTYPE)*width));

            
            memcpy(con->host_rptr, &host_Connectivities[i].host_rptr[start], sizeof(unsigned int)*(con->postNum+1) );
            CUDA_SAFE_CALL( cudaMemcpy( con->rptr, &host_Connectivities[i].host_rptr[start], sizeof(unsigned int)*(con->postNum+1), cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL( cudaMemcpy( con->cindices, &host_Connectivities[i].cindices[ host_Connectivities[i].host_rptr[start] ], sizeof(unsigned int)*width, cudaMemcpyDeviceToDevice));
            CUDA_SAFE_CALL( cudaMemcpy( con->val,  &host_Connectivities[i].val[ host_Connectivities[i].host_rptr[start] ], sizeof(CTYPE)*width, cudaMemcpyDeviceToDevice));

            //Malloc memory for ParallelReduction
            if( con->pr == -1 ){
                CUDA_SAFE_CALL( cudaMalloc( &(con->pr_out), sizeof(CTYPE)*(Dev[dev_id].neurons[con->preType].num)) );
                CUDA_SAFE_CALL( cudaMalloc( &(con->tmp), sizeof(CTYPE)*(Dev[dev_id].neurons[con->postType].num)) );
            }

            // treat IDs
                        
            int pre_presn_neuron_start = (dev_id > 0)?Dev[dev_id - 1].start[con->preType]:0;
            int pre_presn_neuron_base_id = (dev_id > 0)?Dev[dev_id-1].neurons[con->preType].base_id:0;
            int next_presn_neuron_start = (dev_id < DEV_NUM-1)?Dev[dev_id+1].start[con->preType]:0;
            int next_presn_neuron_base_id = (dev_id < DEV_NUM-1)?Dev[dev_id+1].neurons[con->preType].base_id:0;

            int m = (width > con->preNum)?width:con->postNum;

            Modifie<<< (m+127)/128 ,128>>>( con->rptr, con->cindices, host_Connectivities[i].host_rptr[start], width, con->postNum,
                                        Dev[dev_id].neuron_num, Dev[dev_id].start[ con->preType ], Dev[dev_id].end[con->preType], Dev[dev_id].neurons[con->preType].base_id,
                                        Dev[dev_id].pre_neuron_num, pre_presn_neuron_start, pre_presn_neuron_base_id,
                                        Dev[dev_id].next_neuron_num, next_presn_neuron_start,next_presn_neuron_base_id );
            start = end;
        }
    }


    // Malloc & Init Param & Create Streams
    for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
        cudaSetDevice(dev_id);
        Sim_cond_lif_exp *d = &Dev[dev_id];
        //CUDA Malloc
        CUDA_SAFE_CALL(cudaMalloc( &(d->u), sizeof(CTYPE)*Dev[dev_id].neuron_num ));
        CUDA_SAFE_CALL(cudaMalloc( &(d->g_exc), sizeof(CTYPE)*(Dev[dev_id].neuron_num)) );
        CUDA_SAFE_CALL(cudaMalloc( &(d->g_inh), sizeof(CTYPE)*Dev[dev_id].neuron_num));
        CUDA_SAFE_CALL(cudaMalloc( &(d->dg_exc), sizeof(CTYPE)*Dev[dev_id].neuron_num));
        CUDA_SAFE_CALL(cudaMalloc( &(d->dg_inh), sizeof(CTYPE)*Dev[dev_id].neuron_num));
        CUDA_SAFE_CALL(cudaMalloc( &(d->refractory_time_left), sizeof(int)*Dev[dev_id].neuron_num));
        CUDA_SAFE_CALL(cudaMalloc( &(d->Inoise), sizeof(CTYPE)*Dev[dev_id].neuron_num));

        CUDA_SAFE_CALL(cudaMalloc( &(d->state), sizeof(curandStatePhilox4_32_10_t)*Dev[dev_id].neuron_num ));

        Philox_setup_kernel<<< (Dev[dev_id].neuron_num + 255)/256, 256 >>>((unsigned long long)time(NULL) , d->state, Dev[dev_id].neuron_num);

        CUDA_SAFE_CALL(cudaMalloc( &(d->type), sizeof(char)*d->neuron_num));
        CUDA_SAFE_CALL(cudaMalloc( &(d->spike), sizeof(char)*delay_max_row*(d->neuron_num + d->pre_neuron_num + d->next_neuron_num)));


        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
        curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long int)seed );
        curandGenerateUniform(gen, Dev[dev_id].u, Dev[dev_id].neuron_num);

        char *host_type = (char *)malloc(sizeof(char)*d->neuron_num);
        for(int type = 0,i=0; type < TotalNumOfCellTypes; type++){
            for(int j = 0; j < d->neurons[type].num; j++){
                host_type[i] = (char)type;
                i++;
            }
        }
        CUDA_SAFE_CALL( cudaMemcpy( d->type, host_type, sizeof(char)*d->neuron_num, cudaMemcpyHostToDevice));

        InitParams<<< (Dev[dev_id].neuron_num + 127)/128, 128>>>( Dev[dev_id].u, Dev[dev_id].g_exc, Dev[dev_id].dg_exc, Dev[dev_id].g_inh, Dev[dev_id].dg_inh, Dev[dev_id].refractory_time_left, Dev[dev_id].spike, Dev[dev_id].dev_neurons, Dev[dev_id].type, Dev[dev_id].state, Dev[dev_id].neuron_num);



        //Stream
        int streamSize = TotalNumOfCellTypes*2+5;
        int TransferStreamId = streamSize - 1;
        int P2PTransfer_prev = streamSize - 3;
        int P2PTransfer_next = streamSize - 2;
        int UpdateStream = streamSize - 4;

        d->streams = (cudaStream_t *)malloc(sizeof(cudaStream_t)*streamSize);
        for(int i=0;i< streamSize-3;i++){
            cudaStreamCreate( &(d->streams[i]) );
        }
        cudaStreamCreateWithFlags( &(d->streams[TransferStreamId]) ,cudaStreamNonBlocking);
        cudaStreamCreateWithFlags( &(d->streams[P2PTransfer_prev]) ,cudaStreamNonBlocking);
        cudaStreamCreateWithFlags( &(d->streams[P2PTransfer_next]) ,cudaStreamNonBlocking);


        // pthread
        if(PRINT){
            int pthread_num = (T_MAX + delay_max_row/2-1)/(delay_max_row/2);
        
        
            d->print_thread = (pthread_t *)malloc( sizeof(pthread_t)*2 );
            d->print_arg = (pthread_val *)malloc( sizeof(pthread_val)*2 );
        
            CUDA_SAFE_CALL(cudaMallocHost( &(d->host_spike[0]) ,sizeof(char)*delay_max_row*(d->pre_neuron_num + d->neuron_num + d->next_neuron_num)));
          	d->host_spike[1] = &(d->host_spike[0][ delay_max_row*(d->pre_neuron_num + d->neuron_num + d->next_neuron_num)/2 ]);

			char *own_spike = (char *)malloc(sizeof(char)*delay_max_row*(d->pre_neuron_num + d->neuron_num + d->next_neuron_num));
        
            for(int i=0;i<2;i++){
                char file_name[50];
                sprintf(file_name, "spike_output/spike_dev%d_%d.dat", dev_id,i);
                d->print_arg[i].fp = fopen(file_name, "w");

                d->print_arg[i].type = host_type;
                d->print_arg[i].delay_max_row = delay_max_row/2;
                d->print_arg[i].total_nn = d->pre_neuron_num + d->neuron_num + d->next_neuron_num;
                d->print_arg[i].own_neuron_num = d->neuron_num;
                d->print_arg[i].spike = &own_spike[ (i%2)*delay_max_row*(d->pre_neuron_num + d->neuron_num + d->next_neuron_num)/2];
		        d->print_arg[i].host_spike = d->host_spike[ i % 2 ];
		        d->print_arg[i].from_spike = &d->spike[ (i%2)*delay_max_row/2*(d->pre_neuron_num + d->neuron_num + d->next_neuron_num)  ];
                d->print_arg[i].stream = d->streams[TransferStreamId];
		        d->print_arg[i].NeuronTypeID = ReverseNeuronTypeID;

				d->print_arg[i].id = i;
				d->print_arg[i].dev_id = dev_id;

                d->print_arg[i].neurons = d->neurons;
                d->print_arg[i].host_neurons = host_Neurons;
                d->print_arg[i].start = d->start;

                //CUDA_SAFE_CALL(cudaMallocHost( &d->arg[i].spike ,sizeof(char)*delay_max_row*( d->pre_neuron_num + d->neuron_num + d->next_neuron_num )/2));
            }
        }
    }

	Dev[0].cpu_sim_thread = (pthread_t *)malloc(sizeof(pthread_t)*2);
	cpu_sim_thread_val *Host_sim_tmp;
    Host_sim_tmp = (cpu_sim_thread_val *)malloc(sizeof(cpu_sim_thread_val)*2);
	for(int th = 0; th < 2; th++){
        Sim_cond_lif_exp *h = &Host_sim_tmp[th].Host;
		
		memcpy( &h->neurons[0], host_Neurons, sizeof(Neuron)*TotalNumOfCellTypes);
		//memcpy( &h->connectivities[0], host_Connectivities, sizeof(Connectivity)*TotalNumOfConnectivityTypes);
		h->gpu_neurons_num = gpu_neurons_num;
		h->gpu_connections_num = gpu_connections_num;
		Host_sim_tmp[th].Dev = Dev;

		h->neuron_num = 0;
		for(int i = gpu_neurons_num; i < TotalNumOfCellTypes; i++){
			h->neuron_num += h->neurons[i].num;
		}

		h->u = (CTYPE *)malloc(sizeof(CTYPE)*h->neuron_num);
		h->g_exc =  (CTYPE *)malloc(sizeof(CTYPE)*h->neuron_num);
		h->g_inh =  (CTYPE *)malloc(sizeof(CTYPE)*h->neuron_num);
		h->dg_exc =  (CTYPE *)malloc(sizeof(CTYPE)*h->neuron_num);
		h->dg_inh =  (CTYPE *)malloc(sizeof(CTYPE)*h->neuron_num);
		h->refractory_time_left = (int *)malloc(sizeof(int)*h->neuron_num);
		h->Inoise =  (CTYPE *)malloc(sizeof(CTYPE)*h->neuron_num);
		h->type =  (char *)malloc(sizeof(char)*h->neuron_num);
		h->spike =  (char *)malloc(sizeof(char)*h->neuron_num);


		for(int i = gpu_neurons_num, base_id = 0; i < TotalNumOfCellTypes; i++){
			h->neurons[i].base_id = base_id;
			for( int j = base_id; j < h->neurons[i].num + base_id; j++) h->type[j] = i;
			base_id += h->neurons[i].num;
		}

		srand( (unsigned)time(NULL) );
		for(int i = 0 ; i < h->neuron_num; i++){
			h->u[i] = host_Neurons[h->type[i]].Vr + (host_Neurons[h->type[i]].Vth - host_Neurons[h->type[i]].Vr)*(  (rand() + 1.0)/(2.0 + RAND_MAX)  );
			h->g_exc[i] = 0.f;
			h->dg_exc[i] = 0.f;
			h->g_inh[i] = 0.f;
			h->dg_inh[i] = 0.f;
			h->refractory_time_left[i] = 0;
			h->spike[i] = 0;
		}
		for(int i = gpu_connections_num; i < TotalNumOfConnectivityTypes; i++){
			unsigned int *cindices;
			CTYPE *weight;
			unsigned int width = host_Connectivities[i].host_rptr[ host_Connectivities[i].postNum ] -  host_Connectivities[i].host_rptr[0];

			cindices = (unsigned int *)malloc(sizeof(unsigned int)*width );
			weight = (CTYPE *)malloc(sizeof(CTYPE)*width );
			cudaMemcpy( cindices, host_Connectivities[i].cindices, sizeof(unsigned int)*width, cudaMemcpyDeviceToHost );
			cudaMemcpy( weight, host_Connectivities[i].val, sizeof(CTYPE)*width, cudaMemcpyDeviceToHost );


			h->connectivities[i].type = host_Connectivities[i].type;
			h->connectivities[i].preNum = host_Connectivities[i].preNum;
			h->connectivities[i].postNum = host_Connectivities[i].postNum;
			h->connectivities[i].preType = host_Connectivities[i].preType;
			h->connectivities[i].postType = host_Connectivities[i].postType;
			h->connectivities[i].initial_weight = host_Connectivities[i].initial_weight;
			h->connectivities[i].delay = host_Connectivities[i].delay;
			h->connectivities[i].max_conv = host_Connectivities[i].max_conv;
			h->connectivities[i].pr = host_Connectivities[i].pr;

			h->connectivities[i].host_rptr = host_Connectivities[i].host_rptr;
			h->connectivities[i].rptr = host_Connectivities[i].host_rptr;
			h->connectivities[i].cindices = cindices;
			h->connectivities[i].val = weight;
		}



		Host_sim_tmp[th].delay_max_row = delay_max_row/2;
		Host_sim_tmp[th].id = th;
		Host_sim_tmp[th].NeuronTypeID = ReverseNeuronTypeID;

		Host_sim_tmp[th].Dev_num = DEV_NUM;

		pthread_barrier_init(&ready[th], NULL, DEV_NUM*2+1);
		pthread_barrier_init(&A_done[th], NULL, DEV_NUM+1);
		pthread_barrier_init(&B_done[th], NULL, DEV_NUM*2+1);
	}

	*Host_sim = Host_sim_tmp;

    // P2P Setting
    for(int dev_id = 0;dev_id < DEV_NUM; dev_id++){
        cudaSetDevice(dev_id);
        int prev_id = (dev_id - 1);
        int next_id = (dev_id + 1);
        int canAccessPeer;
        if (dev_id > 0){
            cudaDeviceCanAccessPeer( &canAccessPeer, dev_id, prev_id );
            if( canAccessPeer ){
                cudaDeviceEnablePeerAccess( prev_id, 0);
                //fprintf(stderr, "%d->%d: P2P enabled.\n", dev_id, prev_id);
            }
        }
        if (next_id < DEV_NUM ){
            cudaDeviceCanAccessPeer( &canAccessPeer, dev_id, next_id );
            if( canAccessPeer ){
                cudaDeviceEnablePeerAccess( next_id, 0);
                //fprintf(stderr, "%d->%d: P2P enabled.\n", dev_id, next_id);
            }
        }
    }
}


void loop( Neuron *host_Neurons, Connectivity *host_Connectivities ){
        int total_nn;
	    int NeuronTypeID[TotalNumOfCellTypes];
	    int ConnectivityTypeID[TotalNumOfConnectivityTypes];

	    cpu_sim_thread_val *Host_sim = NULL;
	

        time_t seed;
        time(&seed);


        if(PROGRESS) fprintf(stderr, "init Neurons params\n");
        init_neurons_params(host_Neurons, NeuronTypeID);
        if(PROGRESS) fprintf(stderr, "init coonnectivity params\n");
        init_connectivity_params(host_Connectivities, host_Neurons, NeuronTypeID, ConnectivityTypeID);

        if(PROGRESS) fprintf(stderr, "set base id\n");
        total_nn = set_base_id(host_Neurons);
        if(PROGRESS) {
                for(int i = 0; i < TotalNumOfCellTypes; i++){
                        fprintf(stderr, "%s\t:%d\n", host_Neurons[i].filename, host_Neurons[i].base_id);
                }
        }



        if(PROGRESS) fprintf(stderr, "allocate spike matrix\n");
        CTYPE max_delay=0;
        for(int i=0; i< TotalNumOfConnectivityTypes;i++){
                max_delay = (host_Connectivities[i].delay > max_delay)?host_Connectivities[i].delay : max_delay;
        }
        int delay_max_row = 2*( (PRINT_T <max_delay)?max_delay:PRINT_T );
        delay_max_row = 200;


        CTYPE *par;
        FILE *debug_fp;
        if(DEBUG){
                debug_fp = fopen(DEBUG_LOG, "w");
                par = (CTYPE *)malloc(sizeof(CTYPE)*total_nn*DEBUG_T);
        }

        if(PROGRESS) fprintf(stderr, "init\n");

        int n=0;
        int target_row = 0;



        if(PROGRESS) fprintf(stderr, "start simulation\n nstep = %d, delay_max_row=%d\n,total_nn=%d\n",T_MAX,delay_max_row,total_nn);



        InputFunctionsStruct *host_InputStimList;
        host_InputStimList = (InputFunctionsStruct *)malloc( sizeof(InputFunctionsStruct)*INPUT_STIM_NUM );
        InitInputStimulation( host_InputStimList, host_Neurons, NeuronTypeID);

        Sim_cond_lif_exp Dev[DEV_NUM];
        Initialization( Dev, &Host_sim, NeuronTypeID, ConnectivityTypeID, host_Neurons, host_Connectivities, host_InputStimList, delay_max_row, total_nn);

    	for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
    		for(int i = 0; i < 2; i++)
                    pthread_create( &Dev[dev_id].print_thread[i], NULL, print_spike_train, (void *)&Dev[dev_id].print_arg[i]);
    	}
    	pthread_create(&Dev[0].cpu_sim_thread[0], NULL, SimulationOnCPU, (void *)(&Host_sim[0]) );
    	pthread_create(&Dev[0].cpu_sim_thread[1], NULL, SimulationOnCPU, (void *)(&Host_sim[1]) );


        int dev_id = 0;
        int pthread_count = 0;

        const int streamSize = TotalNumOfCellTypes*2+5;
        const int TransferStreamId = streamSize - 1;
        const int P2PTransfer_prev = streamSize - 3;
        const int P2PTransfer_next = streamSize - 2;
        const int UpdateStream = streamSize - 4;
	    const int ParallelReduction = streamSize -5;

        Sim_cond_lif_exp *d;
        cudaEvent_t m_start[DEV_NUM],m_stop[DEV_NUM];

 
        for(int trial = 0; trial < TRIALS; trial++){
            printf("%d\n", trial);
    
    	    if(PRINT){
                for(int i=0; i < 2; i++){
                    char file_name[50];
                    sprintf(file_name, "spike_output/spike_host%d.dat", i);
                    Host_sim[i].fp = fopen(file_name, "w");
                }
            }
    
            #pragma omp parallel private (dev_id, d, n, target_row, pthread_count) num_threads( DEV_NUM )
            {
                dev_id = omp_get_thread_num();
                cudaSetDevice(dev_id);
                d = &Dev[dev_id];

                if(PRINT){
                    for(int i = 0; i < 2; i++){
                                char file_name[128];
                                sprintf(file_name, "spike_output/spike_dev%d_%d.dat", dev_id,i);
                                d->print_arg[i].fp = fopen(file_name, "w");
                    }
                }
                InitParams<<< (Dev[dev_id].neuron_num + 127)/128, 128>>>( Dev[dev_id].u, Dev[dev_id].g_exc, Dev[dev_id].dg_exc, Dev[dev_id].g_inh, Dev[dev_id].dg_inh, Dev[dev_id].refractory_time_left, Dev[dev_id].spike, Dev[dev_id].dev_neurons, Dev[dev_id].type, Dev[dev_id].state, Dev[dev_id].neuron_num);
                cudaDeviceSynchronize();

                #pragma omp barrier
        	    if(MEASURE){
                  cudaEventCreate(&m_start[dev_id]);
                  cudaEventCreate(&m_stop[dev_id]);
                  cudaEventRecord(m_start[dev_id]);
                }
    
            	if(MEASURE)  cudaEventRecord(m_start[dev_id]);
                n = 0;
                target_row = 0;
                pthread_count = 0;
                
                ///////////////////////////////////   start
                while(n < T_MAX){
                    if(PROGRESS && dev_id == 0 ) fprintf(stderr,"\rn:%d",n);
                    for(int i = 0; i < INPUT_STIM_NUM; i++ ){
    				    if(d->InputStimList[i].num > 0) InputStimulation<<< (d->InputStimList[i].num+127)/128, 128, 0, d->streams[0]>>>( n, d->spike, d->InputStimList[i].state, d->InputStimList[i].num, d->InputStimList[i].base_id, d->InputStimList[i].IdList, target_row*d->total_neuron_num, d->neuron_num, d->InputStimList[i].func_id);
            		}
                    //generate_noise_current(Inoise,0.0,10.,total_nn);
                    // calc_synaptic_current_dif : Input_Glomerulus は飛ばす
                    //for(int i=0;i<TotalNumOfConnectivityTypes;i++) // ループ内部で毎回初期化するのと、ループ外で確保しておく,直接引数に渡すのどちらが速いか比較。また、構造体そのものを引数に取れるのか要調査
                    //for(int i=0;i< d->gpu_connections_num ;i++){ // ループ内部で毎回初期化するのと、ループ外で確保しておく,直接引数に渡すのどちらが速いか比較。また、構造体そのものを引数に取れるのか要調査
                    for(int i=0;i < d->gpu_connections_num ;i++){ // ループ内部で毎回初期化するのと、ループ外で確保しておく,直接引数に渡すのどちらが速いか比較。また、構造体そのものを引数に取れるのか要調査
                        Connectivity *c = &d->connectivities[i];
                        int preType = c->preType;
                        int postType = c->postType;
                        Neuron *preNeuron = &d->neurons[preType];
                        Neuron *postNeuron = &d->neurons[postType];
                        CTYPE *dg = (c->initial_weight > 0)?d->dg_exc:d->dg_inh;
                        CTYPE w_bar = (c->initial_weight > 0)?c->initial_weight: c->initial_weight * (-1);
                        int delay = (int)(c->delay);
                        int read_row = (target_row - delay >= 0)?target_row - delay: delay_max_row + target_row - delay ;
                                   
                        if( c->pr == -1  ){
                            calc_current_diff_PR( c->pr_out, c->tmp, c->max_conv, preNeuron->base_id, postNeuron->num, postNeuron->base_id, dg, d->refractory_time_left, c->rptr, c->cindices, c->val, w_bar, d->spike, read_row*(d->pre_neuron_num+d->neuron_num+d->next_neuron_num), d->neuron_num, d->streams[ ParallelReduction ]);
                            //}else if(c->type == parallel_fiber_to_basket || c->type == parallel_fiber_to_stellate || c->type == parallel_fiber_to_golgi){
                        }else if( c->pr > 1 ){
                            calculate_current_diff_arrange<<< (postNeuron->num*c->pr+127)/128, 128, 0, d->streams[ postType*2 + (c->initial_weight > 0) ]>>>( preNeuron->num, preNeuron->base_id, postNeuron->num, postNeuron->base_id, dg, d->refractory_time_left , c->rptr, c->cindices, c->val, w_bar, d->spike, read_row*(d->pre_neuron_num+d->neuron_num+d->next_neuron_num), d->neuron_num, c->pr);
                        }else{
                            calculate_current_diff<<<(postNeuron->num+127)/128, 128, 0, d->streams[ postType*2 + (c->initial_weight > 0) ]>>>( preNeuron->num, preNeuron->base_id, postNeuron->num, postNeuron->base_id, dg, d->refractory_time_left , c->rptr, c->cindices, c->val, w_bar, d->spike, read_row*(d->pre_neuron_num+d->neuron_num+d->next_neuron_num), d->neuron_num);
                        }
                    }
    
                            /*
                            Inoise生成するならここ
                            */
                    #pragma omp barrier
                    for( int stream_id = 0; stream_id < streamSize - 3; stream_id++) cudaStreamSynchronize(d->streams[stream_id]);
                    update<<< (d->neuron_num + 127)/128, 128, 0, d->streams[UpdateStream]>>>(d->u, d->g_exc, d->dg_exc, d->g_inh, d->dg_inh, d->Inoise, d->spike, d->refractory_time_left, d->dev_neurons, d->type, target_row*(d->pre_neuron_num+d->neuron_num+d->next_neuron_num), d->gpu_neurons_num);
                    cudaStreamSynchronize(d->streams[UpdateStream]);
                    ///// Sync & Memory Transfer
                    /// Communication
                    if(dev_id > 0) cudaMemcpyPeerAsync( &Dev[dev_id - 1].spike[target_row*Dev[dev_id - 1].total_neuron_num + Dev[dev_id - 1].neuron_num + Dev[dev_id -1].pre_neuron_num], dev_id - 1,   &Dev[dev_id].spike[target_row*Dev[dev_id].total_neuron_num], dev_id, sizeof(char)*Dev[dev_id].neuron_num, d->streams[P2PTransfer_prev] );
                    if(dev_id < DEV_NUM - 1) cudaMemcpyPeerAsync( &Dev[dev_id + 1].spike[target_row*Dev[dev_id + 1].total_neuron_num + Dev[dev_id + 1].neuron_num], dev_id + 1,   &Dev[dev_id].spike[target_row*Dev[dev_id].total_neuron_num], dev_id, sizeof(char)*Dev[dev_id].neuron_num, d->streams[P2PTransfer_next] );


/*
                    cudaDeviceSynchronize();
                    PF_PC_LTD_LTP<<< (d->connectivities[ ConnectivityTypeID[parallel_fiber_to_purkinje] ].max_conv + 127)/128, 128 >>>(
                        d->spike, d->connectivities[ ConnectivityTypeID[ parallel_fiber_to_purkinje ] ].rptr, d->connectivities[ ConnectivityTypeID[ parallel_fiber_to_purkinje ] ].cindices, d->connectivities[ ConnectivityTypeID[ parallel_fiber_to_purkinje ] ].val,
                                              d->connectivities[ ConnectivityTypeID[ io_to_purkinje ] ].rptr, d->connectivities[ ConnectivityTypeID[ io_to_purkinje ] ].cindices,
                                              target_row , delay_max_row,
                                              d->neurons[ NeuronTypeID[ purkinje_cell ] ].num, d->neurons[ NeuronTypeID[ granule_cell ] ].base_id, d->neurons[ NeuronTypeID[ io_cell ] ].base_id, d->total_neuron_num );
*/          
                    n++;
                    target_row = (target_row < delay_max_row-1)?target_row+1:0;
            
                    //// Output function
                    if(PRINT){
                        if( (n!= 0 && (target_row == 0 || target_row == delay_max_row/2)) ) {
    					    if(pthread_count > 1) pthread_barrier_wait(&B_done[ pthread_count%2 ]);
                            d = &Dev[dev_id];
                            d->print_arg[pthread_count%2].n = n;
    					    if(dev_id == 0) Host_sim[pthread_count%2].n = n;
    					    pthread_barrier_wait(&ready[pthread_count%2]);
                            pthread_count++;
                        }
                    }
                }
                ///////////////////////////////////   end
    		    if(PRINT){
    			    pthread_barrier_wait(&B_done[0]);
    			    pthread_barrier_wait(&B_done[1]);
    
                    for(int i = 0; i < 2; i ++) fclose( d->print_arg[i].fp );
                    #pragma omp barrier
    		    }
            	if(MEASURE){
                    cudaEventRecord(m_stop[dev_id]);
                    cudaEventSynchronize(m_stop[dev_id]);
    
                    float milliseconds=0;
                    cudaEventElapsedTime(&milliseconds, m_start[dev_id], m_stop[dev_id]);
                    fprintf(stderr, "\nDev %d: main loop took %f milliseconds\n", dev_id,  milliseconds);
                    cudaEventDestroy(m_start[dev_id]);
                    cudaEventDestroy(m_stop[dev_id]);
    		    }
            }
            ////////////////////// end of omp brock
    
            for(int i = 0; i < 2; i++) fclose( Host_sim[i].fp );
    
            char code[256];
            int syserr;
            for(int i = 0; i < DEV_NUM; i++){
                sprintf(code, "cat spike_output/spike_dev%d_*.dat > spike_output/spike_dev%d.dat", i, i);
                syserr = system(code);
                if(syserr == -1) fprintf(stderr, "system call failed: %s\n", code);
            }
    
            sprintf(code, "cat spike_output/spike_dev?.dat spike_output/spike_host?.dat > spike_output/spike_result_trial%d.dat", trial);
            syserr = system(code);
        }

        ////


        double sum = 0;
        double num = 0;
        for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
            cudaSetDevice(dev_id);
            cudaDeviceSynchronize();

                Connectivity *c = &Dev[dev_id].connectivities[ ConnectivityTypeID[parallel_fiber_to_purkinje] ];
                int width = c->host_rptr[ c->postNum ] - c->host_rptr[ 0 ] ;
                CTYPE *tmp_width = (CTYPE *)malloc(sizeof(CTYPE)*width);
                cudaMemcpy( tmp_width, c->val, sizeof(CTYPE)*width, cudaMemcpyDeviceToHost );
                num += width;
                for(int j = 0; j < width; j++){
                    sum += tmp_width[j];
                }

        }
        fprintf(stderr, "pf_PC: avg %lf, sum:%lf, num:%lf\n", sum / num, sum, num);


        ////



        if(PRINT){
                if(PROGRESS) fprintf(stderr, "write...");
        
                for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
                    for(int i = 0; i < 2; i ++){
				        fclose( Dev[dev_id].print_arg[i].fp );
			      	    pthread_cancel( Dev[dev_id].print_thread[i] );
				        printf("kill thread %d-%d\n", dev_id, i);
			        }
                    char code[100];
                    sprintf(code, "cat spike_output/spike_dev%d_*.dat > spike_output/spike_dev%d.dat", dev_id, dev_id);
                    int syserr = system(code);
                    if(syserr == -1) fprintf(stderr, "system call failed: %s\n", code);

                    sprintf(code, "cat spike_output/spike_dev?.dat > spike_output/spike_result.dat");
                    syserr = system(code);
                    if(syserr == -1) fprintf(stderr, "system call failed: %s\n", code);
                }
		        pthread_cancel( Dev[0].cpu_sim_thread[0] );
		        pthread_cancel( Dev[0].cpu_sim_thread[1] );

                if(PROGRESS) fprintf(stderr, "\ndone\n");
        }
        if(PROGRESS) fprintf(stderr, "free memory\n");

        cudaDeviceSynchronize();


        /*
        for(int i =0;i<5;i++) cudaStreamSynchronize(streams[i]);
        for(int i =0;i<5;i++) cudaStreamDestroy(streams[i]);

        // free
        cudaFree(u);
        cudaFree(g_exc);
        cudaFree(dg_exc);
        cudaFree(g_inh);
        cudaFree(dg_inh);
        cudaFree(spike);
        cudaFree(type);
        */
/*
        cudaFreeHost(host_spike0);
        cudaFreeHost(host_spike1);
*/      
        if(DEBUG){
                free(par);
                fclose(debug_fp);
        }

        for(int i = 0; i < Dev[0].gpu_connections_num; i++){
                Connectivity *c = &host_Connectivities[i];
                cudaFree(c->rptr);
                cudaFree(c->cindices);
                cudaFree(c->val);
                if(PROGRESS)fprintf(stderr, "\rcuda Free %d",i);
        }
        if(PROGRESS)fprintf(stderr, "\n");

        if(PROGRESS)fprintf(stderr, "host Free\n");

        return;
}


int main(){
        Neuron *Neurons;
        Connectivity *Connectivities;

        Neurons = (Neuron *)malloc( sizeof(Neuron)*TotalNumOfCellTypes);
        Connectivities = (Connectivity*)malloc(sizeof(Connectivity)*TotalNumOfConnectivityTypes);

        loop( Neurons, Connectivities );
        if(PROGRESS)fprintf(stderr, "end loop\n");

        free(Neurons);
        free(Connectivities);
        if(PROGRESS)fprintf(stderr, "end free\n");
        return 0;
}
