# XOR-Neural-Network-Solver

Implementation of a perceptron neural network with multiple layers and backpropagation to solve the XOR problem, in the c programming language.

## How to run

- Clone the repository with the following command:

```bash
git clone https://github.com/PepeuFBV/XOR-Neural-Network-Solver
```

- Navigate to the project's directory with the following command:

```bash
cd XOR-Neural-Network-Solver
```

- Compile the main.c file with the following command:

```bash
gcc main.c -o main
```

- Run the compiled file with the following command:

```bash
./main
```

## How to use

- The program will train the neural network to solve the XOR problem and then test it with the following inputs:

```c
{0, 0}
{0, 1}
{1, 0}
{1, 1}
```

- The output will be the result of the XOR operation for each input.

```c
0
1
1
0
```

## How it works

The network is defined with an input layer of 2 neurons, a hidden layer of 4 neurons, and an output layer of 1 neuron.

The tanh_activation function is used as the activation function for the neurons, and its derivative is computed using tanh_derivative to assist in the backpropagation process.

Weights between the input layer and hidden layer (input_weights), and between the hidden layer and output layer (output_weights), are initialized randomly.

The training process involves looping through a specified number of epochs. In each epoch, the network processes each input pair in the XOR dataset:

- Forward propagation is performed to compute the activations in the hidden and output layers.
- The error is calculated as the difference between the target output and the actual output.
- Backpropagation is used to adjust the weights based on the computed error, using the learning rate.

The network is tested after training to evaluate its performance on the XOR inputs, and the results are printed to compare the network's output with the expected XOR values.
