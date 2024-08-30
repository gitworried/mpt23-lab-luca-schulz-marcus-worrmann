#include <omp.h>
#include <stdlib.h>
#include "mpt_nn.h"
#include "mpt_nn_utility.h"
#include "math.h"

// Utility function for sigmoid
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// Utility function for sigmoid derivative
double dSigmoid(double x)
{
    return x * (1.0 - x);
}

// Sequential inference implementation
void forward_pass_sequential(double inputs[], double hiddenLayer[], double outputLayer[],
                             double hiddenLayerBias[], double outputLayerBias[],
                             double **hiddenWeights, double **outputWeights,
                             int numInputs, int numHiddenNodes, int numOutputs)
{
    for (int i = 0; i < numHiddenNodes; i++)
    {
        double activation = hiddenLayerBias[i];
        for (int j = 0; j < numInputs; j++)
        {
            activation += inputs[j] * hiddenWeights[j][i];
        }
        hiddenLayer[i] = sigmoid(activation);
    }

    for (int i = 0; i < numOutputs; i++)
    {
        double activation = outputLayerBias[i];
        for (int j = 0; j < numHiddenNodes; j++)
        {
            activation += hiddenLayer[j] * outputWeights[j][i];
        }
        outputLayer[i] = sigmoid(activation);
    }
}

// Parallel inference implementation using OpenMP
void forward_pass_parallel(double inputs[], double hiddenLayer[], double outputLayer[],
                           double hiddenLayerBias[], double outputLayerBias[],
                           double **hiddenWeights, double **outputWeights,
                           int numInputs, int numHiddenNodes, int numOutputs)
{
#pragma omp parallel for
    for (int i = 0; i < numHiddenNodes; i++)
    {
        double activation = hiddenLayerBias[i];
        for (int j = 0; j < numInputs; j++)
        {
            activation += inputs[j] * hiddenWeights[j][i];
        }
        hiddenLayer[i] = sigmoid(activation);
    }

#pragma omp parallel for
    for (int i = 0; i < numOutputs; i++)
    {
        double activation = outputLayerBias[i];
        for (int j = 0; j < numHiddenNodes; j++)
        {
            activation += hiddenLayer[j] * outputWeights[j][i];
        }
        outputLayer[i] = sigmoid(activation);
    }
}

// SIMD inference implementation using OpenMP
void forward_pass_simd(double inputs[], double hiddenLayer[], double outputLayer[],
                       double hiddenLayerBias[], double outputLayerBias[],
                       double **hiddenWeights, double **outputWeights,
                       int numInputs, int numHiddenNodes, int numOutputs)
{
#pragma omp parallel for simd
    for (int i = 0; i < numHiddenNodes; i++)
    {
        double activation = hiddenLayerBias[i];
#pragma omp simd
        for (int j = 0; j < numInputs; j++)
        {
            activation += inputs[j] * hiddenWeights[j][i];
        }
        hiddenLayer[i] = sigmoid(activation);
    }

#pragma omp parallel for simd
    for (int i = 0; i < numOutputs; i++)
    {
        double activation = outputLayerBias[i];
#pragma omp simd
        for (int j = 0; j < numHiddenNodes; j++)
        {
            activation += hiddenLayer[j] * outputWeights[j][i];
        }
        outputLayer[i] = sigmoid(activation);
    }
}

// Sequential backpropagation implementation
void backpropagation_sequential(double inputs[], double target[], double hiddenLayer[], double outputLayer[],
                                double hiddenLayerBias[], double outputLayerBias[],
                                double **hiddenWeights, double **outputWeights,
                                double lr, int numInputs, int numHiddenNodes, int numOutputs)
{
    double deltaOutput[numOutputs];
    double deltaHidden[numHiddenNodes];

    // Compute delta for output layer
    for (int i = 0; i < numOutputs; i++)
    {
        double error = target[i] - outputLayer[i];
        deltaOutput[i] = error * dSigmoid(outputLayer[i]);
    }

    // Compute delta for hidden layer
    for (int i = 0; i < numHiddenNodes; i++)
    {
        double error = 0.0;
        for (int j = 0; j < numOutputs; j++)
        {
            error += deltaOutput[j] * outputWeights[i][j];
        }
        deltaHidden[i] = error * dSigmoid(hiddenLayer[i]);
    }

    // Update output weights and biases
    for (int i = 0; i < numOutputs; i++)
    {
        outputLayerBias[i] += deltaOutput[i] * lr;
        for (int j = 0; j < numHiddenNodes; j++)
        {
            outputWeights[j][i] += hiddenLayer[j] * deltaOutput[i] * lr;
        }
    }

    // Update hidden weights and biases
    for (int i = 0; i < numHiddenNodes; i++)
    {
        hiddenLayerBias[i] += deltaHidden[i] * lr;
        for (int j = 0; j < numInputs; j++)
        {
            hiddenWeights[j][i] += inputs[j] * deltaHidden[i] * lr;
        }
    }
}

// Parallel backpropagation using OpenMP
void backpropagation_parallel(double inputs[], double target[], double hiddenLayer[], double outputLayer[],
                              double hiddenLayerBias[], double outputLayerBias[],
                              double **hiddenWeights, double **outputWeights,
                              double lr, int numInputs, int numHiddenNodes, int numOutputs)
{
    double deltaOutput[numOutputs];
    double deltaHidden[numHiddenNodes];

// Compute delta for output layer in parallel
#pragma omp parallel for
    for (int i = 0; i < numOutputs; i++)
    {
        double error = target[i] - outputLayer[i];
        deltaOutput[i] = error * dSigmoid(outputLayer[i]);
    }

// Compute delta for hidden layer in parallel
#pragma omp parallel for
    for (int i = 0; i < numHiddenNodes; i++)
    {
        double error = 0.0;
        for (int j = 0; j < numOutputs; j++)
        {
            error += deltaOutput[j] * outputWeights[i][j];
        }
        deltaHidden[i] = error * dSigmoid(hiddenLayer[i]);
    }

// Update output weights and biases in parallel
#pragma omp parallel for
    for (int i = 0; i < numOutputs; i++)
    {
        outputLayerBias[i] += deltaOutput[i] * lr;
        for (int j = 0; j < numHiddenNodes; j++)
        {
            outputWeights[j][i] += hiddenLayer[j] * deltaOutput[i] * lr;
        }
    }

// Update hidden weights and biases in parallel
#pragma omp parallel for
    for (int i = 0; i < numHiddenNodes; i++)
    {
        hiddenLayerBias[i] += deltaHidden[i] * lr;
        for (int j = 0; j < numInputs; j++)
        {
            hiddenWeights[j][i] += inputs[j] * deltaHidden[i] * lr;
        }
    }
}

// SIMD backpropagation using OpenMP
void backpropagation_simd(double inputs[], double target[], double hiddenLayer[], double outputLayer[],
                          double hiddenLayerBias[], double outputLayerBias[],
                          double **hiddenWeights, double **outputWeights,
                          double lr, int numInputs, int numHiddenNodes, int numOutputs)
{
    double deltaOutput[numOutputs];
    double deltaHidden[numHiddenNodes];

// Compute delta for output layer with SIMD
#pragma omp parallel for simd
    for (int i = 0; i < numOutputs; i++)
    {
        double error = target[i] - outputLayer[i];
        deltaOutput[i] = error * dSigmoid(outputLayer[i]);
    }

// Compute delta for hidden layer with SIMD
#pragma omp parallel for simd
    for (int i = 0; i < numHiddenNodes; i++)
    {
        double error = 0.0;
#pragma omp simd
        for (int j = 0; j < numOutputs; j++)
        {
            error += deltaOutput[j] * outputWeights[i][j];
        }
        deltaHidden[i] = error * dSigmoid(hiddenLayer[i]);
    }

// Update output weights and biases with SIMD
#pragma omp parallel for simd
    for (int i = 0; i < numOutputs; i++)
    {
        outputLayerBias[i] += deltaOutput[i] * lr;
#pragma omp simd
        for (int j = 0; j < numHiddenNodes; j++)
        {
            outputWeights[j][i] += hiddenLayer[j] * deltaOutput[i] * lr;
        }
    }

// Update hidden weights and biases with SIMD
#pragma omp parallel for simd
    for (int i = 0; i < numHiddenNodes; i++)
    {
        hiddenLayerBias[i] += deltaHidden[i] * lr;
#pragma omp simd
        for (int j = 0; j < numInputs; j++)
        {
            hiddenWeights[j][i] += inputs[j] * deltaHidden[i] * lr;
        }
    }
}

