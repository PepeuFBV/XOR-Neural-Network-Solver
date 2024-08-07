#include <stdio.h>
#include <stdlib.h>

#include "train.h"

int main() {
    int num_layers = 4;
    int num_neurons[] = {2, 2, 2, 1};  // Example sizes for input, hidden, and output layers
    int hidden_size[] = {2, 2};        // Example sizes for hidden layers

    // Create the architecture
    if (create_architecture(num_layers, num_neurons, hidden_size) != 0) {
        fprintf(stderr, "Error creating architecture\n");
        return 1;
    }

    // Allocate memory for layers
    layer *layers = (layer *)malloc(num_layers * sizeof(layer));
    for (int i = 0; i < num_layers; i++) {
        layers[i] = create_layer(num_neurons[i]);
    }

    // Example training data
    int num_training = 2;
    double inputs[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};  // Example inputs
    double outputs[4] = {0.7, 0.8, 0.9, 1.0};           // Example outputs
    double learning_rate = 0.01;

    // Train the network
    train(num_layers, num_neurons, layers, inputs, outputs, num_training, learning_rate);

    // Free allocated memory
    free_memory(num_layers, layers);

    return 0;
}