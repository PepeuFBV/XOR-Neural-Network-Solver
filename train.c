#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "extra.c"

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

int get_inputs(int num_training, int num_inputs, double *(*inputs)) {
    for (int i = 0; i < num_training; i++) {
        printf("Enter the inputs for training %d: ", i + 1);

        for (int j = 0; j < num_inputs; j++) {
            scanf("%lf", &inputs[i][j]);
        }
    }
    return 0;
}

int get_outputs(int num_training, int num_outputs, double *(*outputs)) {
    for (int i = 0; i < num_training; i++) {
        printf("Enter the outputs for training %d: ", i + 1);

        for (int j = 0; j < num_outputs; j++) {
            scanf("%lf", &outputs[i][j]);
        }
    }
    return 0;
}

void forward_prop(int num_layers, int *num_neurons, layer *lay) {
    for (int i = 1; i < num_layers; i++) {
        for (int j = 0; j < num_neurons[i]; j++) {
            lay[i].neurons[j].z = lay[i].neurons[j].bias;

            for (int k = 0; k < num_neurons[i - 1]; k++) {
                lay[i].neurons[j].z += lay[i - 1].neurons[k].out_weights[j] * lay[i - 1].neurons[k].activation;
            }

            // Relu Activation Function for Hidden Layers
            if (i < num_layers - 1) {
                if (lay[i].neurons[j].z < 0) {
                    lay[i].neurons[j].activation = 0;
                } else {
                    lay[i].neurons[j].activation = lay[i].neurons[j].z;
                }
            } else {
                // Sigmoid Activation function for Output Layer
                lay[i].neurons[j].activation = 1 / (1 + exp(-lay[i].neurons[j].z));
                printf("OUTPUT: %d\n", (int)round(lay[i].neurons[j].activation));
                printf("\n");
            }
        }
    }
}

void backwards(int num_layers, int *num_neurons, layer *lay, double *output) {
    for (int i = num_layers - 1; i > 0; i--) {
        for (int j = 0; j < num_neurons[i]; j++) {
            if (i == num_layers - 1) {
                lay[i].neurons[j].dactv = lay[i].neurons[j].activation - output[j];
            } else {
                lay[i].neurons[j].dactv = 0;
                for (int k = 0; k < num_neurons[i + 1]; k++) {
                    lay[i].neurons[j].dactv += lay[i + 1].neurons[k].dactv * lay[i + 1].neurons[k].out_weights[j];
                }
            }

            lay[i].neurons[j].dz = lay[i].neurons[j].dactv * sigmoid_derivative(lay[i].neurons[j].activation);
            lay[i].neurons[j].dbias = lay[i].neurons[j].dz;
            for (int k = 0; k < num_neurons[i - 1]; k++) {
                lay[i].neurons[j].dw[k] = lay[i].neurons[j].dz * lay[i - 1].neurons[k].activation;
            }
        }
    }
}

void update_weights(int num_layers, int *num_neurons, layer *lay, double learning_rate) {
    for (int i = 1; i < num_layers; i++) {
        for (int j = 0; j < num_neurons[i]; j++) {
            lay[i].neurons[j].bias -= learning_rate * lay[i].neurons[j].dbias;
            for (int k = 0; k < num_neurons[i - 1]; k++) {
                lay[i].neurons[j].out_weights[k] -= learning_rate * lay[i].neurons[j].dw[k];
            }
        }
    }
}

layer create_layer(int size) {
    layer l;
    l.size = size;
    l.neurons = (Neuron *)malloc(size * sizeof(Neuron));
    return l;
}

Neuron create_neuron(int num_weights) {
    Neuron n;
    n.out_weights = (float *)malloc(num_weights * sizeof(float));
    n.dw = (float *)malloc(num_weights * sizeof(float));
    return n;
}

int initialize_weights() {
    // random weights between -1 and 1
    srand(time(NULL));
    for (int i = 0; i < 100; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        r = r * 2 - 1;
        // printf("%f\n", r);
    }
    return 0;  // SUCCESS_INIT_WEIGHTS
}

int create_architecture(int num_layers, int *sizes, int *hidden_size) {
    layer *layers = (layer *)malloc(num_layers * sizeof(layer));

    for (int i = 0; i < num_layers; i++) {
        layers[i] = create_layer(sizes[i]);
        printf("Layer %d has %d neurons\n", i + 1, layers[i].size);

        for (int j = 0; j < sizes[i]; j++) {
            if (i < (num_layers - 1)) {
                layers[i].neurons[j] = create_neuron(hidden_size[i + 1]);
            }
            printf("Neuron %d has %d weights\n", j + 1, i + 1);
        }
    }

    if (initialize_weights() != 0) {  // SUCCESS_INIT_WEIGHTS
        return -1;                    // ERROR_CREATE_ARCHITECTURE
    }

    return 0;  // SUCCESS_CREATE_ARCHITECTURE
}

void free_memory(int num_layers, layer *lay) {
    for (int i = 0; i < num_layers; i++) {
        free(lay[i].neurons);
    }
    free(lay);
}

void train(int num_layers, int *num_neurons, layer *lay, double *inputs, double *outputs, int num_training, double learning_rate) {
    for (int i = 0; i < num_training; i++) {
        for (int j = 0; j < num_neurons[0]; j++) {
            lay[0].neurons[j].activation = inputs[i * num_neurons[0] + j];
        }

        forward_prop(num_layers, num_neurons, lay);
        backwards(num_layers, num_neurons, lay, outputs + i * num_neurons[num_layers - 1]);
        update_weights(num_layers, num_neurons, lay, learning_rate);
        printf("Training %d done\n", i + 1);
    }
}