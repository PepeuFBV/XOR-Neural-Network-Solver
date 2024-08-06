#include <stdio.h>
#include <stdlib.h>

// Função de ativação sigmoid
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// Derivada da função sigmoid
double sigmoid_derivative(double x)
{
    return x * (1.0 - x);
}