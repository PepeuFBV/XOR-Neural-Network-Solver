#ifndef TRAIN_H
#define TRAIN_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Neuron {
    float activation;
    float *out_weights;
    float bias;
    float z;

    float dactv;
    float *dw;
    float dbias;
    float dz;
} Neuron;

typedef struct layer {
    int size;
    Neuron *neurons;
} layer;

int get_inputs(int num_training, int num_inputs, double *(*inputs));
int get_outputs(int num_training, int num_outputs, double *(*outputs));
void forward_prop(int num_layers, int *num_neurons, layer *lay);
void backwards(int num_layers, int *num_neurons, layer *lay, double *output);
void update_weights(int num_layers, int *num_neurons, layer *lay, double learning_rate);
layer create_layer(int size);
Neuron create_neuron(int num_weights);
int initialize_weights();
int create_architecture(int num_layers, int *sizes, int *hidden_size);
void free_memory(int num_layers, layer *lay);
void train(int num_layers, int *num_neurons, layer *lay, double *inputs, double *outputs, int num_training, double learning_rate);

#endif