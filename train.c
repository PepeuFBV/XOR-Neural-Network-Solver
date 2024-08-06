#include "extra.c"

// Initialize weights with random values between -1 and 1
void initialize_weights(double *weights, int size) {
    for (int i = 0; i < size; ++i) {
        weights[i] = (double)rand() / (double)RAND_MAX * 2.0 -
                     1.0;  // Random value between -1 and 1
    }
}

// Training the neural network
void train(double *input, double *output, double *weights_input_hidden, double *bias_hidden, double *weights_hidden_output, double *bias_output, int input_size, int hidden_size, int output_size, int epochs, double learning_rate) {
    double *hidden_layer = malloc(hidden_size * sizeof(double));
    double *output_layer = malloc(output_size * sizeof(double));
    double *hidden_layer_activation = malloc(hidden_size * sizeof(double));
    double *output_layer_activation = malloc(output_size * sizeof(double));
    double *hidden_layer_error = malloc(hidden_size * sizeof(double));
    double *output_layer_error = malloc(output_size * sizeof(double));
    double *hidden_layer_delta = malloc(hidden_size * sizeof(double));
    double *output_layer_delta = malloc(output_size * sizeof(double));

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (int i = 0; i < 4; ++i) {  // 4 samples for XOR problem

            // Forward pass
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

            // Calcula o erro
            for (int j = 0; j < output_size; ++j) {
                output_layer_error[j] = output[i * output_size + j] - output_layer[j];
                output_layer_delta[j] = output_layer_error[j] * sigmoid_derivative(output_layer[j]);
                total_loss += output_layer_error[j] * output_layer_error[j];
            }

            // Calculate hidden layer error
            for (int j = 0; j < hidden_size; ++j) {
                hidden_layer_error[j] = 0.0;
                for (int k = 0; k < output_size; ++k) {
                    hidden_layer_error[j] += output_layer_delta[k] * weights_hidden_output[k * hidden_size + j];
                }
                hidden_layer_delta[j] = hidden_layer_error[j] * sigmoid_derivative(hidden_layer[j]);
            }

            // Update weights and bias
            for (int j = 0; j < hidden_size; ++j) {
                for (int k = 0; k < input_size; ++k) {
                    weights_input_hidden[j * input_size + k] += learning_rate * hidden_layer_delta[j] * input[i * input_size + k];
                }
                bias_hidden[j] += learning_rate * hidden_layer_delta[j];
            }

            for (int j = 0; j < output_size; ++j) {
                for (int k = 0; k < hidden_size; ++k) {
                    weights_hidden_output[j * hidden_size + k] += learning_rate * output_layer_delta[j] * hidden_layer[k];
                }
                bias_output[j] += learning_rate * output_layer_delta[j];
            }
        }

        // Print loss every n/25  n = number of epochs
        if ((epoch + 1) % (epochs / 25) == 0) {
            printf("Epoch %d, Loss: %f\n", epoch + 1, total_loss / 4.0);
        }
    }

    free(hidden_layer);
    free(output_layer);
    free(hidden_layer_activation);
    free(output_layer_activation);
    free(hidden_layer_error);
    free(output_layer_error);
    free(hidden_layer_delta);
    free(output_layer_delta);
}