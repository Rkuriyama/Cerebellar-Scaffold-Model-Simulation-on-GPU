NVCC=nvcc
#CUDAFLAGS = --gpu-architecture=compute_75 --gpu-code=compute_75 -O3 -lm -lcurand -lpthread -lineinfo
#CUDAFLAGS = --gpu-architecture=compute_75 --gpu-code=compute_75 -O3 -lm -lcurand -lineinfo
CUDAFLAGS = --gpu-architecture=compute_70 --gpu-code=compute_70 -O3 -lm -lcurand -lpthread -lineinfo -g
OPENMPFLAGS =  -Xcompiler -fopenmp -lgomp
SFMTC = -DSFMT_MEXP=19937 /usr/local/include/SFMT-src/SFMT.c

all: mains

mains: main

main: main.cu GPUScaffoldSimulation.o init_params.o Simulation_header.o InputStimuli.o SynapticPlasticity.o UserDefined.o
	$(NVCC) $(CUDAFLAGS) $(OPENMPFLAGS) -o $@ $^

GPUScaffoldSimulation.o: GPUScaffoldSimulation.cu 
	$(NVCC) $(CUDAFLAGS) $(OPENMPFLAGS) -c $^


init_params.o: init_params.cu init_params.h struct_enum_def.h
	$(NVCC) $(CUDAFLAGS) -c init_params.cu

InputStimuli.o: InputStimuli.cu struct_enum_def.h
	$(NVCC) $(CUDAFLAGS) -c $<

Simulation_header.o: Simulation_header.cu Simulation_header.h struct_enum_def.h
	$(NVCC) $(CUDAFLAGS) -c Simulation_header.cu

SynapticPlasticity.o: SynapticPlasticity.cu SynapticPlasticity.h struct_enum_def.h
	$(NVCC) $(CUDAFLAGS) -c SynapticPlasticity.cu


UserDefined.o: UserDefined.cu UserDefined.h struct_enum_def.h
	$(NVCC) $(CUDAFLAGS) -c UserDefined.cu

main.cu: option.h

clean:
	rm *.o main
