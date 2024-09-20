#include <omp.h>
#include <stdlib.h>
#include "mpt_nn.h"
#include "mpt_nn_utility.h"
#include "math.h"

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double dSigmoid(double x)
{
    return x * (1.0 - x);
}

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

void forward_pass_parallel(double inputs[], double hiddenLayer[], double outputLayer[],
                           double hiddenLayerBias[], double outputLayerBias[],
                           double **hiddenWeights, double **outputWeights,
                           int numInputs, int numHiddenNodes, int numOutputs)
{
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numHiddenNodes; i++)
    {
        double activation = hiddenLayerBias[i];
        for (int j = 0; j < numInputs; j++)
        {
            activation += inputs[j] * hiddenWeights[j][i];
        }
        hiddenLayer[i] = sigmoid(activation);
    }

#pragma omp parallel for schedule(dynamic)
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

void forward_pass_simd(double inputs[], double hiddenLayer[], double outputLayer[],
                       double hiddenLayerBias[], double outputLayerBias[],
                       double **hiddenWeights, double **outputWeights,
                       int numInputs, int numHiddenNodes, int numOutputs)
{
#pragma omp parallel for simd schedule(dynamic)
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

#pragma omp parallel for simd schedule(dynamic)
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

void backpropagation_sequential(double inputs[], double target[], double hiddenLayer[], double outputLayer[],
                                double hiddenLayerBias[], double outputLayerBias[],
                                double **hiddenWeights, double **outputWeights,
                                double lr, int numInputs, int numHiddenNodes, int numOutputs)
{
    double deltaOutput[numOutputs];
    double deltaHidden[numHiddenNodes];

    for (int i = 0; i < numOutputs; i++)
    {
        double error = target[i] - outputLayer[i];
        deltaOutput[i] = error * dSigmoid(outputLayer[i]);
    }

    for (int i = 0; i < numHiddenNodes; i++)
    {
        double error = 0.0;
        for (int j = 0; j < numOutputs; j++)
        {
            error += deltaOutput[j] * outputWeights[i][j];
        }
        deltaHidden[i] = error * dSigmoid(hiddenLayer[i]);
    }

    for (int i = 0; i < numOutputs; i++)
    {
        outputLayerBias[i] += deltaOutput[i] * lr;
        for (int j = 0; j < numHiddenNodes; j++)
        {
            outputWeights[j][i] += hiddenLayer[j] * deltaOutput[i] * lr;
        }
    }

    for (int i = 0; i < numHiddenNodes; i++)
    {
        hiddenLayerBias[i] += deltaHidden[i] * lr;
        for (int j = 0; j < numInputs; j++)
        {
            hiddenWeights[j][i] += inputs[j] * deltaHidden[i] * lr;
        }
    }
}

void backpropagation_parallel(double inputs[], double target[], double hiddenLayer[], double outputLayer[],
                              double hiddenLayerBias[], double outputLayerBias[],
                              double **hiddenWeights, double **outputWeights,
                              double lr, int numInputs, int numHiddenNodes, int numOutputs)
{
    double deltaOutput[numOutputs];
    double deltaHidden[numHiddenNodes];

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numOutputs; i++)
    {
        double error = target[i] - outputLayer[i];
        deltaOutput[i] = error * dSigmoid(outputLayer[i]);
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numHiddenNodes; i++)
    {
        double error = 0.0;
        for (int j = 0; j < numOutputs; j++)
        {
            error += deltaOutput[j] * outputWeights[i][j];
        }
        deltaHidden[i] = error * dSigmoid(hiddenLayer[i]);
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numHiddenNodes; i++)
    {
        for (int j = 0; j < numOutputs; j++)
        {
            double local_update = hiddenLayer[i] * deltaOutput[j] * lr;
            {
                outputWeights[i][j] += local_update;
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numOutputs; i++)
    {
        outputLayerBias[i] += deltaOutput[i] * lr;
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numInputs; i++)
    {
        for (int j = 0; j < numHiddenNodes; j++)
        {
            double local_update = inputs[i] * deltaHidden[j] * lr;
            {
                hiddenWeights[i][j] += local_update;
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numHiddenNodes; i++)
    {
        hiddenLayerBias[i] += deltaHidden[i] * lr;
    }
}

void backpropagation_simd(double inputs[], double target[], double hiddenLayer[], double outputLayer[],
                          double hiddenLayerBias[], double outputLayerBias[],
                          double **hiddenWeights, double **outputWeights,
                          double lr, int numInputs, int numHiddenNodes, int numOutputs)
{
    double deltaOutput[numOutputs];
    double deltaHidden[numHiddenNodes];

#pragma omp parallel for simd schedule(dynamic)
    for (int i = 0; i < numOutputs; i++)
    {
        double error = target[i] - outputLayer[i];
        deltaOutput[i] = error * dSigmoid(outputLayer[i]);
    }

#pragma omp parallel for simd schedule(dynamic)
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

#pragma omp parallel for simd schedule(dynamic)
    for (int i = 0; i < numHiddenNodes; i++)
    {
#pragma omp simd
        for (int j = 0; j < numOutputs; j++)
        {
            double local_update = hiddenLayer[i] * deltaOutput[j] * lr;
            outputWeights[i][j] += local_update;
        }
    }

#pragma omp parallel for simd schedule(dynamic)
    for (int i = 0; i < numOutputs; i++)
    {
        outputLayerBias[i] += deltaOutput[i] * lr;
    }

#pragma omp parallel for simd schedule(dynamic)
    for (int i = 0; i < numInputs; i++)
    {
#pragma omp simd
        for (int j = 0; j < numHiddenNodes; j++)
        {
            double local_update = inputs[i] * deltaHidden[j] * lr;
            hiddenWeights[i][j] += local_update;
        }
    }

#pragma omp parallel for simd schedule(dynamic)
    for (int i = 0; i < numHiddenNodes; i++)
    {
        hiddenLayerBias[i] += deltaHidden[i] * lr;
    }
}
