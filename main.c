#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "train.c"

int main() {
    srand(time(NULL));  // Seed for random numbers

    // XOR problem
    double input[4 * 2] = {0, 0, 0, 1, 1, 0, 1, 1};
    double output[4 * 1] = {0, 1, 1, 0};

    int input_size = 2;
    int hidden_size = 2;
    int output_size = 1;
    printf("How many epochs do you want to run? ");
    int epochs = 10000;  // Default value
    scanf("%d", &epochs);
    printf("What is the learning rate? ");
    double learning_rate = 0.1;  // Default value
    scanf("%lf", &learning_rate);

    // Initialize weights and bias
    double weights_input_hidden[2 * 2];
    double bias_hidden[2];
    double weights_hidden_output[2 * 1];
    double bias_output[1];
    initialize_weights(weights_input_hidden, 2 * 2);
    initialize_weights(weights_hidden_output, 2 * 1);
    initialize_weights(bias_hidden, 2);
    initialize_weights(bias_output, 1);

    train(input, output, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, input_size, hidden_size, output_size, epochs, learning_rate);

    printf("\nPredictions:\n");
    for (int i = 0; i < 4; ++i) {
        double *hidden_layer = malloc(hidden_size * sizeof(double));
        double *output_layer = malloc(output_size * sizeof(double));
        double *hidden_layer_activation = malloc(hidden_size * sizeof(double));
        double *output_layer_activation = malloc(output_size * sizeof(double));

        for (int j = 0; j < hidden_size; ++j) {
            hidden_layer_activation[j] = 0.0;
            for (int k = 0; k < input_size; ++k) {
                hidden_layer_activation[j] += input[i * input_size + k] * weights_input_hidden[j * input_size + k];
            }
            hidden_layer_activation[j] += bias_hidden[j];
            hidden_layer[j] = sigmoid(hidden_layer_activation[j]);
        }

        for (int j = 0; j < output_size; ++j) {
            output_layer_activation[j] = 0.0;
            for (int k = 0; k < hidden_size; ++k) {
                output_layer_activation[j] += hidden_layer[k] * weights_hidden_output[j * hidden_size + k];
            }
            output_layer_activation[j] += bias_output[j];
            output_layer[j] = sigmoid(output_layer_activation[j]);
        }

        printf("Input: %f, %f -> Expected: %f Output: %f\n", input[i * input_size], input[i * input_size + 1], output[i], output_layer[0]);

        free(hidden_layer);
        free(output_layer);
        free(hidden_layer_activation);
        free(output_layer_activation);
    }

    return 0;
}
