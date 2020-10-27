NVCC=nvcc
#CUDAFLAGS = --gpu-architecture=compute_75 --gpu-code=compute_75 -O3 -lm -lcurand -lpthread -lineinfo
#CUDAFLAGS = --gpu-architecture=compute_75 --gpu-code=compute_75 -O3 -lm -lcurand -lineinfo
CUDAFLAGS = --gpu-architecture=compute_70 --gpu-code=compute_70 -O3 -lm -lcurand -lpthread -lineinfo -g
OPENMPFLAGS =  -Xcompiler -fopenmp -lgomp
SFMTC = -DSFMT_MEXP=19937 /usr/local/include/SFMT-src/SFMT.c

all: Simulation_header.o init_params.o mains

mains: main


main: main.cu init_params.o Simulation_header.o UserInputFunctions.o SynapticPlasticity.o
	$(NVCC) $(CUDAFLAGS) $(OPENMPFLAGS) -o main main.cu init_params.o Simulation_header.o UserInputFunctions.o SynapticPlasticity.o

init_params.o: init_params.cu init_params.h
	$(NVCC) $(CUDAFLAGS) -c init_params.cu

UserInputFunctions.o: UserInputFunctions.cu struct_enum_def.h
	$(NVCC) $(CUDAFLAGS) -c UserInputFunctions.cu

Simulation_header.o: Simulation_header.cu Simulation_header.h
	$(NVCC) $(CUDAFLAGS) -c Simulation_header.cu

SynapticPlasticity.o: SynapticPlasticity.cu SynapticPlasticity.h struct_enum_def.h
	$(NVCC) $(CUDAFLAGS) -c SynapticPlasticity.cu

main.cu: option.h

clean:
	rm *.o main
