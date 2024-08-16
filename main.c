#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

inline static float rand_float(float a)
{
	return (float)rand() / (float)(RAND_MAX / a);
}

inline static float tanh_activation(float x)
{
	return (1.0f - expf(-2 * x)) / (1.0f + expf(-2 * x));
}

inline static float tanh_derivative(float x)
{
	float tanh_x = tanh_activation(x);
	return 1 - tanh_x * tanh_x;
}

int main()
{
	srand((unsigned int)time(NULL));

	// XOR inputs and targets
	float inputs[4][2] = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};
	float targets[4] = { 0, 1, 1, 0 }; // XOR outputs

	// Network parameters
	float hidden_layer[4] = { 0 };		// 4 hidden neurons
	float output_layer = 0;				// 1 output neuron
	float input_weights[2][4] = { 0 };  // weights between input and hidden
	float output_weights[4] = { 0 };    // weights between hidden and output

	// Initialize weights randomly between 0 and 1
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			input_weights[i][j] = rand_float(1.0f);

			if (i == 0) // Initialize output weights once
				output_weights[j] = rand_float(1.0f);
		}
	}

	float learning_rate = 0.1f;
	int epochs = 1000; // number of training iterations

	// Training loop
	for (int epoch = 0; epoch < epochs; epoch++)
	{
		float total_error = 0;

		// For each input in the XOR dataset
		for (int n = 0; n < 4; n++)
		{
			// Forward propagation
			// 1. Calculate activations in hidden layer
			for (int i = 0; i < 4; i++)
			{
				float sum = 0;
				for (int j = 0; j < 2; j++)
				{
					sum += inputs[n][j] * input_weights[j][i];
				}
				hidden_layer[i] = tanh_activation(sum);
			}

			// 2. Calculate activation in output layer
			float sum = 0;
			for (int i = 0; i < 4; i++)
			{
				sum += hidden_layer[i] * output_weights[i];
			}
			output_layer = tanh_activation(sum);

			// Calculate error
			float error = targets[n] - output_layer;
			total_error += error * error; // squared error

			// Backpropagation
			// 1. Calculate output layer error and update output weights
			float output_delta = error * tanh_derivative(output_layer);
			for (int i = 0; i < 4; i++)
			{
				output_weights[i] += learning_rate * output_delta * hidden_layer[i];
			}

			// 2. Calculate hidden layer error and update input weights
			float hidden_error[4] = { 0 };
			for (int i = 0; i < 4; i++)
			{
				hidden_error[i] = output_weights[i] * output_delta * tanh_derivative(hidden_layer[i]);
			}

			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					input_weights[i][j] += learning_rate * hidden_error[j] * inputs[n][i];
				}
			}
		}

		// Optionally print the error every 1000 epochs to track progress
		/*if (epoch % 1000 == 0)
		{*/
			printf("Epoch %d, Total Error: %.6f\n", epoch, total_error);
		//}
	}

	// After training, test the network on XOR inputs
	printf("\nTrained network results:\n");
	for (int n = 0; n < 4; n++)
	{
		// Forward propagation for testing
		for (int i = 0; i < 4; i++)
		{
			float sum = 0;
			for (int j = 0; j < 2; j++)
			{
				sum += inputs[n][j] * input_weights[j][i];
			}
			hidden_layer[i] = tanh_activation(sum);
		}

		float sum = 0;
		for (int i = 0; i < 4; i++)
		{
			sum += hidden_layer[i] * output_weights[i];
		}
		output_layer = tanh_activation(sum);

		printf("Input: %.0f %.0f, Output: %.2f (Expected: %.0f)\n", inputs[n][0], inputs[n][1], output_layer, targets[n]);
	}

	return 0;
}
