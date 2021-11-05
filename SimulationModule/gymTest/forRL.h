#include <stdio.h>
#include <stdlib.h>

#include <curand_kernel.h>

#include "struct_enum_def.h"


enum RL_Layer {
    RL_INPUT,
    RL_Q,
    RL_V,
    RL_A,
    RL_ACTION,
    RL_REWARD,
    RL_LAYER_TYPES
};

enum RL_Loc { // location
    RL_HOST,
    RL_DEV
};

enum RL_Val_type {
    RL_SPIKE,
    RL_CURRENT
};

typedef struct {
    enum RL_Val_type type;
    enum RL_Loc location;
    int population_size; // = neuron per population
    int output_dim; // size of data = output_dim * values
    void *data;
    float *values,
          *d_values;
    
} RL_Data;

typedef struct {

} RL_Set;
