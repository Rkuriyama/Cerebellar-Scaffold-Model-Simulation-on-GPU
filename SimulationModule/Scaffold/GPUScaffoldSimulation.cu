#include "GPUScaffoldSimulation.h"

pthread_barrier_t ready[2];
pthread_barrier_t A_done[2];
pthread_barrier_t B_done[2];

void *SimulationOnCPU(void *p){
	cpu_sim_thread_val *pv = (cpu_sim_thread_val *)p;
	int count = 0;
	Sim_cond_lif_exp *h = &pv->Host;
	int id = pv->id;
    int T_cpu_sim = pv->T_cpu_sim;
    int T_print = pv->T_print;
	int n;
	while(true){
		pthread_barrier_wait(&ready[id]);
		pthread_barrier_wait(&A_done[id]);

        n = pv->n  - T_cpu_sim;

        // target_row -> read_row
		for(int target_row = 0; target_row < T_cpu_sim; target_row++){
			for(int i = h->gpu_connections_num; i < TotalNumOfConnectivityTypes; i++){
	        	Connectivity *c = &h->connectivities[i];
                int preType = c->preType;
                int postType = c->postType;
                //Neuron *preNeuron = &h->neurons[preType];
                Neuron *postNeuron = &h->neurons[postType];
                CTYPE *dg = (c->initial_weight > 0)?h->dg_exc:h->dg_inh;
                CTYPE w_bar = (c->initial_weight > 0)?c->initial_weight:c->initial_weight*(-1);
                int delay = (int)(c->delay);
                int read_row = (target_row + count*T_cpu_sim) - delay;
				int target_block = (read_row < 0)? (id+1)%2: id;
                read_row = (read_row >= 0)? read_row: T_print + target_row - delay ;

				host_spike_propagation( preType, postNeuron->num, postNeuron->base_id,
						dg, c->max_conv, c->ELL_cindices,  c->ELL_val, w_bar,
						read_row, target_block, pv->Dev);

			}
            for(int i = 0; i < TotalNumOfCellTypes; i++){
                Neuron *target = &h->neurons[i];
                if( target->dev_type == OUTPUT){
			        host_update_lif_ch(h->u, h->dg_exc, h->dg_inh, h->Inoise,  &(h->spike[target_row*h->neuron_num]), h->refractory_time_left, h->neurons[i], target_row, n+target_row, pv->fp, pv->NeuronTypeID);
                }
            }
		}
        count = (T_print/T_cpu_sim - 1 > count)? count + 1 : 0;

		fflush(pv->fp);
		pthread_barrier_wait(&B_done[id]);
	}
}


void* print_spike_train(void *p){
    pthread_val *pv = (pthread_val *)p;
    int T_print = pv->T_print;
    int T_cpu_sim = pv->T_cpu_sim;
    int total_nn = pv->total_nn;
    char *spike = pv->host_spike;
    char *type = pv->type;
    int n;
	int id = pv->id;
    int count = 0;
	cudaSetDevice(pv->dev_id);

	while(true){
	    pthread_barrier_wait(&ready[id]);
        CUDA_SAFE_CALL( cudaMemcpyAsync( &(pv->host_spike[count*T_cpu_sim*pv->total_nn]), &(pv->from_spike[count*T_cpu_sim*pv->total_nn]), sizeof(char)*T_cpu_sim*pv->total_nn, cudaMemcpyDeviceToHost, pv->stream ));
        cudaStreamSynchronize(pv->stream);
	    pthread_barrier_wait(&A_done[id]);
        count++;
        if( T_print/T_cpu_sim == count){
            n = pv->n  - T_print;
            count = 0;
	        for(int step = 0 ; step < T_print; step++){
            	CTYPE t = (n+step);
            	for(int i=0;i<pv->own_neuron_num;i++){
                	if(spike[step*total_nn + i] !=0){
		                fprintf(pv->fp, "%f\t%d\t%d\n",t, i  + pv->start[type[i]] - pv->neurons[type[i]].base_id, pv->NeuronTypeID[ type[i] ] );
		            }
                }
                fflush(pv->fp);
      	    }
        }
	    pthread_barrier_wait(&B_done[id]);
	}
}

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

__global__ void ModifieELL( int *cindices, unsigned int max_conv, unsigned int post_sn_num,
                            unsigned int neuron_num, unsigned int pre_sn_start, unsigned int pre_sn_end, unsigned int pre_sn_base,
                            unsigned int pre_neuron_num, unsigned int pre_pre_sn_start, unsigned int pre_pre_sn_base,
                            unsigned int next_neuron_num, unsigned int next_pre_sn_start, unsigned int next_pre_sn_base, unsigned int *tmp_spike ){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if( tid < max_conv*post_sn_num && ( cindices[tid] >= 0) ){
        unsigned int tmp = cindices[tid];
        if( cindices[tid] < pre_sn_start){
             cindices[tid] = (neuron_num + pre_pre_sn_base + (cindices[tid] - pre_pre_sn_start)) - pre_sn_base;
            if(tmp != tmp_spike[ pre_sn_base + cindices[tid] ]){
                printf("pre: ans: %d, res[pre_sn_base + %d]=%d\n", tmp, cindices[tid], tmp_spike[pre_sn_base + cindices[tid]]);
                 assert(0);
            }
        }else if( cindices[tid] >= pre_sn_start && cindices[tid] < pre_sn_end ){
             cindices[tid] = cindices[tid] - pre_sn_start;
            if(tmp != tmp_spike[ pre_sn_base + cindices[tid] ]){
                printf("own: ans: %d, res[pre_sn_base + %d]=%d\n", tmp, cindices[tid], tmp_spike[pre_sn_base + cindices[tid]]);
                 assert(0);
            }
        }else if( cindices[tid] >= pre_sn_end){
            cindices[tid] = neuron_num + pre_neuron_num + next_pre_sn_base + (cindices[tid] - next_pre_sn_start) - pre_sn_base;
            if(tmp != tmp_spike[ pre_sn_base + cindices[tid] ]){
                printf("nex: ans: %d, res[pre_sn_base + %d]=%d\n", tmp, cindices[tid], tmp_spike[pre_sn_base + cindices[tid]]);
                assert(0);
            }
        }
        else assert(0);       
    }
}

void Initialization( Sim_cond_lif_exp *Dev, cpu_sim_thread_val **Host_sim, pthread_t **cpu_sim_threads, int *NeuronTypeID, int *ConnectivityTypeID, Neuron *host_Neurons, Connectivity *host_Connectivities, InputFunctionsStruct *host_InputStimList, int T_print, int T_cpu_sim, int total_nn){

    time_t seed;
    time(&seed);

	int gpu_neurons_num = 0;
	int gpu_connections_num = 0;

    // P2P Setting
    
    for(int dev_id = 0;dev_id < DEV_NUM; dev_id++){
        cudaSetDevice(dev_id);
        int prev_id = (dev_id - 1);
        int next_id = (dev_id + 1);
        int canAccessPeer = 0;
        if (dev_id > 0){
            cudaDeviceCanAccessPeer( &canAccessPeer, dev_id, prev_id );
            if( canAccessPeer ){
                fprintf(stderr, "%d->%d: P2P can be enabled.\n", dev_id, prev_id);
                CUDA_SAFE_CALL( cudaDeviceEnablePeerAccess( prev_id, 0) );
                fprintf(stderr, "%d->%d: P2P enabled.\n", dev_id, prev_id);
            }
        }
        canAccessPeer = 0;
        if (next_id < DEV_NUM ){
            cudaDeviceCanAccessPeer( &canAccessPeer, dev_id, next_id );
            if( canAccessPeer ){
                fprintf(stderr, "%d->%d: P2P can be enabled.\n", dev_id, next_id);
                CUDA_SAFE_CALL( cudaDeviceEnablePeerAccess( next_id, 0) );
                fprintf(stderr, "%d->%d: P2P enabled.\n", dev_id, next_id);
            }
        }
    }
    

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
            neurons[i].El = h_n.El;
            neurons[i].dt_ref = h_n.dt_ref;
            neurons[i].Ie = h_n.Ie;
            neurons[i].Vr = h_n.Vr;
            neurons[i].Vth = h_n.Vth;
            neurons[i].gL = h_n.gL;
            neurons[i].dev_type = h_n.dev_type;
            neurons[i].c_num = h_n.c_num;

            int c_num = h_n.c_num;
            CTYPE *tmp_w = (CTYPE *)malloc(sizeof(CTYPE)*c_num);
            CTYPE *tmp_E = (CTYPE *)malloc(sizeof(CTYPE)*c_num);
            CTYPE *tmp_g_bar = (CTYPE *)malloc(sizeof(CTYPE)*c_num);
            CTYPE *tmp_tau = (CTYPE *)malloc(sizeof(CTYPE)*c_num);
            CTYPE *tmp_g = (CTYPE *)malloc(sizeof(CTYPE)*neurons[i].c_num * neurons[i].num );

            for(int unit = 0; unit < neurons[i].c_num*neurons[i].num; unit++){
                tmp_g[unit] = 0.f;
            }


            if( neurons[i].dev_type == NORMAL ){
                cudaMalloc( &neurons[i].w, sizeof(CTYPE)*c_num);
                cudaMalloc( &neurons[i].E, sizeof(CTYPE)*c_num);
                cudaMalloc( &neurons[i].g_bar, sizeof(CTYPE)*c_num);
                cudaMalloc( &neurons[i].tau, sizeof(CTYPE)*c_num);
                cudaMalloc( &neurons[i].g, sizeof(CTYPE)*neurons[i].c_num*neurons[i].num );

                cudaMemcpy( neurons[i].w, h_n.w, sizeof(CTYPE)*c_num, cudaMemcpyHostToDevice);
                cudaMemcpy( neurons[i].E, h_n.E, sizeof(CTYPE)*c_num, cudaMemcpyHostToDevice);
                cudaMemcpy( neurons[i].g_bar, h_n.g_bar, sizeof(CTYPE)*c_num, cudaMemcpyHostToDevice);
                cudaMemcpy( neurons[i].tau, h_n.tau, sizeof(CTYPE)*c_num, cudaMemcpyHostToDevice);
                cudaMemcpy( neurons[i].g, tmp_g, sizeof(CTYPE)*neurons[i].c_num*neurons[i].num, cudaMemcpyHostToDevice);

                free(tmp_w);
                free(tmp_E);
                free(tmp_g_bar);
                free(tmp_tau);
                free(tmp_g);
            }else{
                neurons[i].w     = h_n.w;
                neurons[i].E     = h_n.E;
                neurons[i].g_bar = h_n.g_bar;
                neurons[i].tau   = h_n.tau;
                neurons[i].g = h_n.g = host_Neurons[i].g = tmp_g;

                free(tmp_w);
                free(tmp_E);
                free(tmp_g_bar);
                free(tmp_tau);
            }

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
		//Dev[dev_id].gpu_neurons_num = Dev[dev_id].neurons[ gpu_neurons_num ].base_id;
		Dev[dev_id].gpu_neurons_num = neurons[ gpu_neurons_num ].base_id;
		Dev[dev_id].gpu_connections_num = gpu_connections_num;
    }

    for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
        fprintf(stderr, "Dev %d - %d neurons\n", dev_id, Dev[dev_id].neuron_num );
    }


    fprintf(stderr, "split Input Stim\n" );
    /// split InputStim
    for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
        Dev[dev_id].InputStimList = (InputFunctionsStruct *)malloc(sizeof(InputFunctionsStruct)*INPUT_STIM_NUM);
    }
    for(int stim_id = 0; stim_id < INPUT_STIM_NUM; stim_id++){
        int start = 0;
        int end = 0;
        unsigned long long seed;

        for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
            fprintf(stderr, "\tdev:%d\tstim:%d\n", dev_id, stim_id );
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

/////////////////////////////////// FOR DEBUGGING
// 
    unsigned int* tmp_local_id[DEV_NUM];
    unsigned int *host_tmp_local_id[DEV_NUM];
//
///////////////////////////////////

    fprintf(stderr, "Malloc\n" );
    // Malloc & Init Param & Create Streams
    for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
        cudaSetDevice(dev_id);
        Sim_cond_lif_exp *d = &Dev[dev_id];
        //CUDA Malloc
        CUDA_SAFE_CALL(cudaMalloc( &(d->u), sizeof(CTYPE)*Dev[dev_id].neuron_num ));
        CUDA_SAFE_CALL(cudaMalloc( &(d->dg_exc), sizeof(CTYPE)*Dev[dev_id].neuron_num));
        CUDA_SAFE_CALL(cudaMalloc( &(d->dg_inh), sizeof(CTYPE)*Dev[dev_id].neuron_num));
        CUDA_SAFE_CALL(cudaMalloc( &(d->refractory_time_left), sizeof(int)*Dev[dev_id].neuron_num));
        CUDA_SAFE_CALL(cudaMalloc( &(d->Inoise), sizeof(CTYPE)*Dev[dev_id].neuron_num));

        CUDA_SAFE_CALL(cudaMalloc( &(d->state), sizeof(curandStatePhilox4_32_10_t)*Dev[dev_id].neuron_num ));

        Philox_setup_kernel<<< (Dev[dev_id].neuron_num + 255)/256, 256 >>>((unsigned long long)time(NULL) , d->state, Dev[dev_id].neuron_num);

        CUDA_SAFE_CALL(cudaMalloc( &(d->type), sizeof(char)*d->neuron_num));
        CUDA_SAFE_CALL(cudaMalloc( &(d->spike), sizeof(char)*T_print*N_PRINT_THREDS*(d->neuron_num + d->pre_neuron_num + d->next_neuron_num)));
        CUDA_SAFE_CALL(cudaMalloc( &(tmp_local_id[dev_id]), sizeof(unsigned int)*(d->neuron_num + d->pre_neuron_num + d->next_neuron_num))); // fd


        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
        curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long int)seed );
        curandGenerateUniform(gen, Dev[dev_id].u, Dev[dev_id].neuron_num);

        char *host_type = (char *)malloc(sizeof(char)*d->neuron_num);
        host_tmp_local_id[dev_id] = (unsigned int *)malloc(sizeof(unsigned int)*d->neuron_num); // for debug;
        for(int type = 0,i=0; type < TotalNumOfCellTypes; type++){
            for(int j = 0; j < d->neurons[type].num; j++){
                host_type[i] = (char)type;
                host_tmp_local_id[dev_id][i] = j + d->start[type]; // fd
                i++;
            }
        }
        CUDA_SAFE_CALL( cudaMemcpy( d->type, host_type, sizeof(char)*d->neuron_num, cudaMemcpyHostToDevice));
                
        InitParams<<< (Dev[dev_id].neuron_num + 127)/128, 128>>>( Dev[dev_id].u, Dev[dev_id].dg_exc, Dev[dev_id].dg_inh, Dev[dev_id].refractory_time_left, Dev[dev_id].spike, Dev[dev_id].dev_neurons, Dev[dev_id].type, Dev[dev_id].state, Dev[dev_id].neuron_num);



        //Stream
        int streamSize = TotalNumOfCellTypes+5;
        //int UpdateStream = streamSize - 4;
        int P2PTransfer_prev = streamSize - 3;
        int P2PTransfer_next = streamSize - 2;
        int TransferStreamId = streamSize - 1;
        fprintf(stderr, "Stream Size = %d\n", streamSize);

        d->streams = (cudaStream_t *)malloc(sizeof(cudaStream_t)*streamSize);
        for(int i=0;i< streamSize-3;i++){
            cudaStreamCreate( &(d->streams[i]) );
            fprintf(stderr, "Stream%d created\n", i);
        }
        cudaStreamCreateWithFlags( &(d->streams[TransferStreamId]) ,cudaStreamNonBlocking);
        cudaStreamCreateWithFlags( &(d->streams[P2PTransfer_prev]) ,cudaStreamNonBlocking);
        cudaStreamCreateWithFlags( &(d->streams[P2PTransfer_next]) ,cudaStreamNonBlocking);


        // pthread
        if(PRINT){
        
            d->print_thread = (pthread_t *)malloc( sizeof(pthread_t)*N_PRINT_THREDS );
            d->print_arg = (pthread_val *)malloc( sizeof(pthread_val)*N_PRINT_THREDS );
        
            CUDA_SAFE_CALL(cudaMallocHost( &(d->host_spike[0]) ,sizeof(char)*T_print*N_PRINT_THREDS*(d->pre_neuron_num + d->neuron_num + d->next_neuron_num)));
            fprintf(stderr, "host malloc - %p\n",d->host_spike[0]);
          	d->host_spike[1] = &(d->host_spike[0][ T_print*(d->pre_neuron_num + d->neuron_num + d->next_neuron_num) ]);

			char *own_spike = (char *)malloc(sizeof(char)*T_print*N_PRINT_THREDS*(d->pre_neuron_num + d->neuron_num + d->next_neuron_num));
        
            for(int i=0;i<N_PRINT_THREDS;i++){
                char file_name[50];
                sprintf(file_name, "spike_output/spike_dev%d_%d.dat", dev_id,i);
                d->print_arg[i].fp = fopen(file_name, "w");

                d->print_arg[i].type = host_type;
                d->print_arg[i].T_print = T_print;
                d->print_arg[i].T_cpu_sim = T_cpu_sim;
                d->print_arg[i].total_nn = d->pre_neuron_num + d->neuron_num + d->next_neuron_num;
                d->print_arg[i].own_neuron_num = d->neuron_num;
                d->print_arg[i].spike = &own_spike[ (i%N_PRINT_THREDS)*T_print*(d->pre_neuron_num + d->neuron_num + d->next_neuron_num)];
		        d->print_arg[i].host_spike = d->host_spike[ i % N_PRINT_THREDS ];
		        d->print_arg[i].from_spike = &d->spike[ (i%N_PRINT_THREDS)*T_print*(d->pre_neuron_num + d->neuron_num + d->next_neuron_num)  ];
                d->print_arg[i].stream = d->streams[TransferStreamId];
		        d->print_arg[i].NeuronTypeID = ReverseNeuronTypeID;

				d->print_arg[i].id = i;
				d->print_arg[i].dev_id = dev_id;
                d->print_arg[i].invocations = 0;

                d->print_arg[i].neurons = d->neurons;
                d->print_arg[i].host_neurons = host_Neurons;
                d->print_arg[i].start = d->start;

                //CUDA_SAFE_CALL(cudaMallocHost( &d->arg[i].spike ,sizeof(char)*T_print*( d->pre_neuron_num + d->neuron_num + d->next_neuron_num )/2));
            }
        }
    }

    for(int dev_id = 0; dev_id < DEV_NUM; dev_id ++){
        ////////////////////////// fd
        CUDA_SAFE_CALL( cudaMemcpy( tmp_local_id[dev_id], host_tmp_local_id[dev_id], sizeof(unsigned int)*Dev[dev_id].neuron_num, cudaMemcpyHostToDevice)); //fd
        if(dev_id > 0)           CUDA_SAFE_CALL( cudaMemcpyPeer( &tmp_local_id[dev_id - 1][ Dev[dev_id - 1].neuron_num + Dev[dev_id -1].pre_neuron_num], dev_id - 1,   tmp_local_id[dev_id], dev_id, sizeof(unsigned int)*Dev[dev_id].neuron_num ));
        if(dev_id < DEV_NUM - 1) CUDA_SAFE_CALL( cudaMemcpyPeer( &tmp_local_id[dev_id + 1][ Dev[dev_id + 1].neuron_num                                ], dev_id + 1,   tmp_local_id[dev_id], dev_id, sizeof(unsigned int)*Dev[dev_id].neuron_num ) );
        ////////////////////////// fd
    }

	*cpu_sim_threads = (pthread_t *)malloc(sizeof(pthread_t)*2);
	cpu_sim_thread_val *Host_sim_tmp;
    
    Host_sim_tmp = (cpu_sim_thread_val *)malloc(sizeof(cpu_sim_thread_val)*2);

    int tmp_neuron_num = 0;
	for(int i = gpu_neurons_num; i < TotalNumOfCellTypes; i++){
		tmp_neuron_num += (host_Neurons[i].dev_type == OUTPUT)? host_Neurons[i].num: 0;
	}
	CTYPE *tmp_u = (CTYPE *)malloc(sizeof(CTYPE)*tmp_neuron_num);
	CTYPE *tmp_dg_exc =  (CTYPE *)malloc(sizeof(CTYPE)*tmp_neuron_num);
	CTYPE *tmp_dg_inh =  (CTYPE *)malloc(sizeof(CTYPE)*tmp_neuron_num);
	int *tmp_refractory_time_left = (int *)malloc(sizeof(int)*tmp_neuron_num);
	CTYPE *tmp_Inoise =  (CTYPE *)malloc(sizeof(CTYPE)*tmp_neuron_num);
	char *tmp_type =  (char *)malloc(sizeof(char)*tmp_neuron_num);
	char *tmp_spike =  (char *)malloc(sizeof(char)*tmp_neuron_num*T_cpu_sim);



	for(int th = 0; th < 2; th++){
        Sim_cond_lif_exp *h = &Host_sim_tmp[th].Host;
		
		memcpy( &h->neurons[0], host_Neurons, sizeof(Neuron)*TotalNumOfCellTypes);
		h->gpu_neurons_num = gpu_neurons_num;
		h->gpu_connections_num = gpu_connections_num;
		Host_sim_tmp[th].Dev = Dev;

		h->neuron_num = 0;
		for(int i = gpu_neurons_num; i < TotalNumOfCellTypes; i++){
			h->neuron_num += (h->neurons[i].dev_type == OUTPUT)? h->neurons[i].num: 0;
		}

		h->u = tmp_u;
		h->dg_exc = tmp_dg_exc;
		h->dg_inh = tmp_dg_inh;
		h->refractory_time_left = tmp_refractory_time_left;
		h->Inoise = tmp_Inoise;
		h->type =  tmp_type;
		h->spike = tmp_spike;

		for(int i = gpu_neurons_num, base_id = 0; i < TotalNumOfCellTypes; i++){
            if(h->neurons[i].dev_type != OUTPUT) continue;
			h->neurons[i].base_id = base_id;
			for( int j = base_id; j < h->neurons[i].num + base_id; j++) h->type[j] = i;
			base_id += h->neurons[i].num;
		}

		srand( (unsigned)time(NULL) );
		for(int i = 0 ; i < h->neuron_num; i++){
			h->u[i] = host_Neurons[h->type[i]].Vr + (host_Neurons[h->type[i]].Vth - host_Neurons[h->type[i]].Vr)*(  (rand() + 1.0)/(2.0 + RAND_MAX)  );
			h->dg_exc[i] = 0.f;
			h->dg_inh[i] = 0.f;
			h->refractory_time_left[i] = 0;
			h->spike[i] = 0;
		}

		for(int i = gpu_connections_num; i < TotalNumOfConnectivityTypes; i++){
            int *ell_cindices;
			CTYPE *ell_val;

            unsigned int width = host_Connectivities[i].max_conv*host_Connectivities[i].postNum;
			ell_cindices = (int *)malloc(sizeof(unsigned int)*width );
			ell_val = (CTYPE *)malloc(sizeof(CTYPE)*width );
			cudaMemcpy( ell_cindices, host_Connectivities[i].ELL_cindices, sizeof(int)*width, cudaMemcpyDeviceToHost );
			cudaMemcpy( ell_val, host_Connectivities[i].ELL_val, sizeof(CTYPE)*width, cudaMemcpyDeviceToHost );

			h->connectivities[i].type = host_Connectivities[i].type;
			h->connectivities[i].preNum = host_Connectivities[i].preNum;
			h->connectivities[i].postNum = host_Connectivities[i].postNum;
			h->connectivities[i].preType = host_Connectivities[i].preType;
			h->connectivities[i].postType = host_Connectivities[i].postType;
			h->connectivities[i].initial_weight = host_Connectivities[i].initial_weight;
			h->connectivities[i].delay = host_Connectivities[i].delay;
			h->connectivities[i].max_conv = host_Connectivities[i].max_conv;
			h->connectivities[i].pr = host_Connectivities[i].pr;

			h->connectivities[i].ELL_cindices = ell_cindices;
			h->connectivities[i].ELL_val = ell_val;
		}




		Host_sim_tmp[th].T_print = T_print;
		Host_sim_tmp[th].T_cpu_sim = T_cpu_sim;
		Host_sim_tmp[th].fp = NULL;
		Host_sim_tmp[th].id = th;
		Host_sim_tmp[th].NeuronTypeID = ReverseNeuronTypeID;

		Host_sim_tmp[th].Dev_num = DEV_NUM;

		pthread_barrier_init(&ready[th], NULL, DEV_NUM*2+1);
		pthread_barrier_init(&A_done[th], NULL, DEV_NUM+1);
		pthread_barrier_init(&B_done[th], NULL, DEV_NUM*2+1);
	}

	*Host_sim = Host_sim_tmp;

    fprintf(stderr, "split connectiities\n");
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
			}else if( con->max_conv > 1000 ){
			    for(int pow2 = 1; con->postNum*pow2 < CUDA_CORES; pow2*=2 )con->pr = pow2;
			}
            fprintf(stderr, "con->type:%d i:%d: con->max_conv:%d con->pr:%d\n", con->type, i, con->max_conv, con->pr);
                        
            end = start + con->postNum;

            //Malloc memory for ParallelReduction
            if( con->pr == -1 ){
                CUDA_SAFE_CALL( cudaMalloc( &(con->pr_out), sizeof(CTYPE)*(Dev[dev_id].neurons[con->preType].num)) );
                CUDA_SAFE_CALL( cudaMalloc( &(con->tmp), sizeof(CTYPE)*(Dev[dev_id].neurons[con->postType].num)) );
            }
            
            // ELL
            CUDA_SAFE_CALL( cudaMalloc( &(con->ELL_cindices), sizeof(int)*(con->max_conv*con->postNum) ) );
            CUDA_SAFE_CALL( cudaMalloc( &(con->ELL_val), sizeof(CTYPE)*(con->max_conv*con->postNum) ) );

            CUDA_SAFE_CALL( cudaMemcpy( con->ELL_cindices, &host_Connectivities[i].ELL_cindices[ start*con->max_conv], sizeof(int)*con->max_conv*con->postNum, cudaMemcpyDeviceToDevice ));
            CUDA_SAFE_CALL( cudaMemcpy( con->ELL_val, &host_Connectivities[i].ELL_val[ start*con->max_conv], sizeof(CTYPE)*con->max_conv*con->postNum, cudaMemcpyDeviceToDevice ));

            // treat IDs

                        
            int pre_presn_neuron_start = (dev_id > 0)?Dev[dev_id - 1].start[con->preType]:0;
            int pre_presn_neuron_base_id = (dev_id > 0)?Dev[dev_id-1].neurons[con->preType].base_id:0;
            int next_presn_neuron_start = (dev_id < DEV_NUM-1)?Dev[dev_id+1].start[con->preType]:0;
            int next_presn_neuron_base_id = (dev_id < DEV_NUM-1)?Dev[dev_id+1].neurons[con->preType].base_id:0;

            int m = (width > con->postNum)?width:con->postNum;
            if(PROGRESS){
                fprintf(stderr,"%d - %d - %d\n", dev_id, ConnectivityTypeID[con->type], m);

                fprintf(stderr, "\tpre_presn_neuron_start:%d pre_presn_neuron_base_id:%d\n next_presn_neuron_start:%d next_presn_neuron_base_id:%d\n",pre_presn_neuron_start, pre_presn_neuron_base_id, next_presn_neuron_start, next_presn_neuron_base_id);

                fprintf(stderr, "ELL_cindices: %p\n", con->ELL_cindices);
            }
            ModifieELL<<< (con->postNum*con->max_conv+127)/128 ,128>>>( con->ELL_cindices, con->max_conv, con->postNum,
                                        Dev[dev_id].neuron_num, Dev[dev_id].start[ con->preType ], Dev[dev_id].end[con->preType], Dev[dev_id].neurons[con->preType].base_id,
                                        Dev[dev_id].pre_neuron_num, pre_presn_neuron_start, pre_presn_neuron_base_id,
                                        Dev[dev_id].next_neuron_num, next_presn_neuron_start,next_presn_neuron_base_id,
                                        tmp_local_id[dev_id] );


            cudaDeviceSynchronize();
            start = end;
        }
    }

   for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
   	for(int i = 0; i < 2; i++)
               pthread_create( &Dev[dev_id].print_thread[i], NULL, print_spike_train, (void *)&Dev[dev_id].print_arg[i]);
   }
   pthread_create(& ((*cpu_sim_threads)[0]), NULL, SimulationOnCPU, (void *)(& (*Host_sim)[0]) );
   pthread_create(& ((*cpu_sim_threads)[1]), NULL, SimulationOnCPU, (void *)(& (*Host_sim)[1]) );


}


void CreateNetworkEnv(  NetworkEnvironment *env, int num_of_dev, int num_of_neuron_types, int num_of_connectivity_types, int num_of_inputs, int num_of_plasticities){

    // Host: allocate memory
    env->Host_Neurons = (Neuron *) malloc ( sizeof(Neuron)*num_of_neuron_types );
    env->NeuronTypeID = (int *) malloc ( sizeof(int)*num_of_neuron_types );
    env->ReverseNeuronTypeID = (int *) malloc ( sizeof(int)*num_of_neuron_types );

    env->Host_Connectivities = (Connectivity *) malloc ( sizeof(Connectivity) * num_of_connectivity_types );
    env->ConnectivityTypeID = (int *) malloc ( sizeof(int) * num_of_connectivity_types );
    
    env->Host_InputStimList = (InputFunctionsStruct *) malloc ( sizeof(InputFunctionsStruct) * num_of_inputs );

    env->Host_Plasticities = (STDP_PLASTICITY *) malloc ( sizeof(STDP_PLASTICITY) * num_of_plasticities );

    env->Dev = (Sim_cond_lif_exp *) malloc ( sizeof(Sim_cond_lif_exp) * num_of_dev );

    // Host: assigne values
    env->num_of_dev = num_of_dev;
    env->num_of_neuron_types = num_of_neuron_types;
    env->num_of_connectivity_types = num_of_connectivity_types;
    env->num_of_inputs = num_of_inputs;
    env->num_of_plasticities = num_of_plasticities;

};

void InitializeNetworkEnv( NetworkEnvironment *env, int T_print, int T_cpu_sim ){
    //Init Neurons
    init_neurons_params( env->Host_Neurons, env->NeuronTypeID );
    env->total_nn = set_base_id(env->Host_Neurons); // envに保持する必要ない気がするけど、デバッグとかで使うかもしれないので維持

    //Init Connectivities
    init_connectivity_params( env->Host_Connectivities, env->Host_Neurons, env->NeuronTypeID, env->ConnectivityTypeID );
    
    //Init Input Stim
    InitInputStimulation( env->Host_InputStimList, env->Host_Neurons, env->NeuronTypeID );

    //Init Plasticities
    Init_Plasticity( &env->Host_Plasticities, /*plasticity num*/  env->ConnectivityTypeID );

    //set T_print.
    CTYPE max_delay=0;
    for(int i=0; i< env->num_of_connectivity_types; i++){
            max_delay = (env->Host_Connectivities[i].delay > max_delay)? env->Host_Connectivities[i].delay : max_delay;
    }
    env->T_print = ( T_print > max_delay )? T_print : max_delay;
    env->T_cpu_sim = T_cpu_sim;

    if(env->T_print < env->T_cpu_sim || (env->T_print % env->T_cpu_sim) != 0){
        fprintf(stderr, "Consistency between T_print and T_cpu_sim is corrupt.\n T_print must be multiples of T_cpu_sim\n");
        exit(1);
    }

    //Split Network and init variables
    Initialization( env->Dev, &(env->Host_sim_val), &(env->cpu_sim_threads), env->NeuronTypeID, env->ConnectivityTypeID, env->Host_Neurons, env->Host_Connectivities, env->Host_InputStimList, env->T_print, env->T_cpu_sim, env->total_nn );

    ResetNetworkEnv(env, 0);

}

void loop_n_steps( NetworkEnvironment *env, int start_step, int n_steps, int trial, char **output_spike_train ){
        int T_print = env->T_print;
        int T_cpu_sim = env->T_cpu_sim;

        static int pthread_count[DEV_NUM] = {0};

        int streamSize = TotalNumOfCellTypes+5;
        int P2PTransfer_prev = streamSize - 3;
        int P2PTransfer_next = streamSize - 2;


        #pragma omp parallel num_threads( DEV_NUM )
        {
            int dev_id = omp_get_thread_num();
            Sim_cond_lif_exp *d;
            cudaSetDevice(dev_id);
            d = &( env->Dev[dev_id] );
            int n=start_step;
            int target_row = start_step % (T_print*N_PRINT_THREDS);
            pthread_count[dev_id] = (start_step == 0)? 0: pthread_count[dev_id];

            cudaDeviceSynchronize();
            
            ///////////////////////////////////   start
            while(n < start_step + n_steps){
                if(PROGRESS && dev_id == 0 ) fprintf(stderr,"n:%d\n",n);

                for(int i = 0; i < env->num_of_inputs; i++ ){
                    InputFunctionsStruct *i_struct = &(d->InputStimList[i]);
    			    if(i_struct->num > 0){
                         InputStimulation<<< (i_struct->num+127)/128, 128, 0, d->streams[ i_struct->type ]>>>( n, d->spike, i_struct->state, i_struct->freq, i_struct->num, i_struct->base_id, i_struct->IdList, target_row*d->total_neuron_num, d->neuron_num, i_struct->func_id);
                    }
        		}
                for(int i=0;i < d->gpu_connections_num ;i++){ 
                    Connectivity *c = &d->connectivities[i];
                    int preType = c->preType;
                    int postType = c->postType;
                    Neuron *preNeuron = &d->neurons[preType];
                    Neuron *postNeuron = &d->neurons[postType];
                    CTYPE *dg = (c->initial_weight > 0)?d->dg_exc:d->dg_inh;
                    CTYPE w_bar = (c->initial_weight > 0)?c->initial_weight: c->initial_weight * (-1);
                    int delay = (int)(c->delay);
                    int read_row = (target_row - delay >= 0)?target_row - delay: N_PRINT_THREDS*T_print + target_row - delay ;

                    if( c->pr == -1  ){
                        spike_propagation_PR( c->pr_out, c->tmp, c->max_conv, preNeuron->base_id, postNeuron->num, postNeuron->base_id, dg, c->ELL_cindices, c->ELL_val, w_bar, d->spike, read_row*(d->pre_neuron_num+d->neuron_num+d->next_neuron_num), d->neuron_num, d->streams[ postType ]);
                    }else if( c->pr > 1 ){
                        spike_propagation_mThreads<<<(postNeuron->num*c->pr+127)/128, 128, 0, d->streams[postType]>>>( postNeuron->base_id, postNeuron->num, dg, c->max_conv, c->ELL_cindices, c->ELL_val, w_bar, d->spike, read_row*(d->pre_neuron_num+d->neuron_num+d->next_neuron_num) + preNeuron->base_id, c->pr );
                    }else{
                        spike_propagation<<<(postNeuron->num+127)/128, 128, 0, d->streams[postType]>>>( postNeuron->base_id, postNeuron->num, dg, c->max_conv, c->ELL_cindices, c->ELL_val, w_bar, d->spike, read_row*(d->pre_neuron_num+d->neuron_num+d->next_neuron_num) + preNeuron->base_id );
                    }
                }
    
                for( int i = 0; i < env->num_of_neuron_types; i++){
                    Neuron *target = &d->neurons[i];
                    if( target->dev_type == NORMAL){
                            update_lif_ch<<< (target->num+127)/128, 128, sizeof(CTYPE)*target->c_num*128, d->streams[ target->type ]>>>( d->u, d->dg_exc, d->dg_inh, d->Inoise, d->spike, d->refractory_time_left, d->neurons[i], target_row*(d->pre_neuron_num+d->neuron_num+d->next_neuron_num));
                    }
                }


                invoke_stdp_plasticity(d->spike, d->neurons, d->connectivities, env->Host_Plasticities, target_row, T_print*N_PRINT_THREDS, d->total_neuron_num, d->streams);

                cudaDeviceSynchronize();

                ///// Sync & Memory Transfer
                if(dev_id > 0)           CUDA_SAFE_CALL( cudaMemcpyPeerAsync( &(env->Dev[dev_id - 1].spike[ target_row*(env->Dev[dev_id - 1].total_neuron_num) + (env->Dev[dev_id - 1].neuron_num) + (env->Dev[dev_id -1].pre_neuron_num)]), dev_id - 1,   &(env->Dev[dev_id].spike[target_row*env->Dev[dev_id].total_neuron_num]), dev_id, sizeof(char)*env->Dev[dev_id].neuron_num, d->streams[ P2PTransfer_prev ] ) );
                if(dev_id < DEV_NUM - 1) CUDA_SAFE_CALL( cudaMemcpyPeerAsync( &(env->Dev[dev_id + 1].spike[ target_row*env->Dev[dev_id + 1].total_neuron_num + env->Dev[dev_id + 1].neuron_num                                ]), dev_id + 1,   &(env->Dev[dev_id].spike[target_row*env->Dev[dev_id].total_neuron_num]), dev_id, sizeof(char)*env->Dev[dev_id].neuron_num, d->streams[ P2PTransfer_next ] ) );


                #pragma omp barrier
                n++;
                target_row = (target_row < N_PRINT_THREDS*T_print-1)?target_row+1:0;


                if( n!= 0 &&  target_row % T_cpu_sim == 0 ) {
                    int pthread_id = (pthread_count[dev_id] / (T_print/T_cpu_sim) )%N_PRINT_THREDS;

    			    if(d->print_arg[pthread_id].invocations > 0){
                         pthread_barrier_wait(&B_done[ pthread_id ]);
                    }
                    //d = &Dev[dev_id];
                    d->print_arg[pthread_id].invocations++;
                    d->print_arg[ pthread_id ].n = n;
                    *output_spike_train = env->Host_sim_val[pthread_id].Host.spike;
    			    if(dev_id == 0){
                         env->Host_sim_val[pthread_id].n = n;
                    }
    			    pthread_barrier_wait(&ready[pthread_id]);
                    pthread_count[dev_id]++;
                }

            }
            ///////////////////////////////////   end of simulation loop
        }
        ////////////////////// end of omp brock
    
        return;
}

/*
enum RSMG_reset_depth {
    RSMG_Trials,
    RSMG_Spikes,
    RSMG_NeuronalParam,
    RSMG_SynapticWeight
};
*/

void ResetNetworkEnv(NetworkEnvironment *env, int trial){
    if(PRINT && env->Host_sim_val[0].fp != NULL){
        

        #pragma omp parallel for num_threads( DEV_NUM )
        for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
            pthread_barrier_wait(&B_done[0]);
            pthread_barrier_wait(&B_done[1]);
        }

        for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
            Sim_cond_lif_exp *d = &( env->Dev[dev_id] );
            for(int i = 0; i < N_PRINT_THREDS; i++){
                fclose(d->print_arg[i].fp);
            }
        }
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


    for(int i=0; i < N_PRINT_THREDS; i++){
        char file_name[50];
        sprintf(file_name, "spike_output/spike_host%d.dat", i);
        env->Host_sim_val[i].fp = fopen(file_name, "w");
    }

    for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
        cudaSetDevice(dev_id);
        cudaDeviceSynchronize();
        Sim_cond_lif_exp *d = &( env->Dev[dev_id] );

        if(PRINT){
            for(int i = 0; i < N_PRINT_THREDS; i++){
                char file_name[128]; // file name -> dev_id_ThreadsId + start_step_n_steps とか？
                sprintf(file_name, "spike_output/spike_dev%d_%d.dat", dev_id,i);
                d->print_arg[i].fp = fopen(file_name, "w");
                d->print_arg[i].invocations = 0;
            }
        }

        InitParams<<< ( d->neuron_num + 127)/128, 128>>>( d->u, d->dg_exc, d->dg_inh, d->refractory_time_left, d->spike, d->dev_neurons, d->type, d->state, d->neuron_num);
    }
    return;
}

void FinalizeNetworkEnv( NetworkEnvironment *env ){
    for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
        cudaSetDevice(dev_id);
        cudaDeviceSynchronize();
        Sim_cond_lif_exp *d = &( env->Dev[dev_id] );


        //////////// fclose
        for(int i = 0; i < N_PRINT_THREDS; i++){
            fclose(d->print_arg[i].fp);
        }

        //////////// destroy streams
        for(int i =0;i<5;i++) cudaStreamDestroy(d->streams[i]);

        //////////// cancel threads
        for(int i = 0; i < 2; i ++){
	  	    pthread_cancel( d->print_thread[i] );
	        printf("kill thread %d-%d\n", dev_id, i);
        }
    }

    pthread_cancel( env->cpu_sim_threads[0] );
    pthread_cancel( env->cpu_sim_threads[1] );
}


void SetInputFreq(NetworkEnvironment *env, const float *freq_list){
    for(int dev_id = 0; dev_id < DEV_NUM; dev_id++){
        for(int i = 0; i < env->num_of_inputs; i++){
            env->Dev[dev_id].InputStimList[i].freq = freq_list[i];
        }
    }
    return;
}
