NVCC=nvcc
#CUDAFLAGS = --gpu-architecture=compute_75 --gpu-code=compute_75 -O3 -lm -lcurand -lpthread -lineinfo
#CUDAFLAGS = --gpu-architecture=compute_75 --gpu-code=compute_75 -O3 -lm -lcurand -lineinfo
CUDAFLAGS = --gpu-architecture=compute_70 --gpu-code=compute_70 -O3 -lm -lcurand -lpthread -lineinfo -g
OPENMPFLAGS =  -Xcompiler -fopenmp -lgomp
SFMTC = -DSFMT_MEXP=19937 /usr/local/include/SFMT-src/SFMT.c
LINKPATH = ./src/
GSS_OBJ = GPUScaffoldSimulation.o init_params.o Simulation_header.o SynapticPlasticity.o UserInputFunctions.o
modifiedGSS_OBJ = GPUScaffoldSimulation.o  UserDefined.o init_params.o Simulation_header.o SynapticPlasticity.o InputStimuli.o

GSS_OBJ_P = $(addprefix ./src/GSS/, $(GSS_OBJ))
modifiedGSS_OBJ_P = $(addprefix ./src/modifiedGSS/, $(modifiedGSS_OBJ))
CGYM_OBJ = cgym.o
CGYM_OBJ_P = $(addprefix ./src/cgym/, $(CGYM_OBJ))

all: main param_modify modifiedParam_modify

main: main.o $(CGYM_OBJ_P) $(GSS_OBJ_P)
	$(NVCC) $(CUDAFLAGS) $(OPENMPFLAGS) -o $@ $^

modifiedParam_modify: param_modify.o $(modifiedGSS_OBJ_P) 
	$(NVCC) $(CUDAFLAGS) $(OPENMPFLAGS) -o $@ $^

param_modify: param_modify.o $(GSS_OBJ_P) 
	$(NVCC) $(CUDAFLAGS) $(OPENMPFLAGS) -o $@ $^

main.o: main.cu
	$(NVCC) $(CUDAFLAGS) $(OPENMPFLAGS) -c $<

param_modify.o: param_modify.cu
	$(NVCC) $(CUDAFLAGS) $(OPENMPFLAGS) -c $<

clean:
	rm *.o main param_modify
