#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mpt_nn.h"
#include "mpt_nn_utility.h"

// Test the sigmoid function
static void test_sigmoid()
{
    assert(sigmoid(0) == 0.5);
    assert(sigmoid(100) > 0.999);
    assert(sigmoid(-100) < 0.001);
    printf("test_sigmoid passed.\n");
}

// Test the initialization of weights
static void test_initialize_weights()
{
    int rows = 2, cols = 3;
    double **weights = malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++)
    {
        weights[i] = malloc(cols * sizeof(double));
    }

    initialize_weights(weights, rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            assert(weights[i][j] >= -0.5 && weights[i][j] <= 0.5);
        }
        free(weights[i]);
    }
    free(weights);

    printf("test_initialize_weights passed.\n");
}

// Test the forward pass (sequential)
static void test_forward_pass()
{
    int numInputs = 2, numHiddenNodes = 2, numOutputs = 1;
    double inputs[] = {0.5, 0.5};
    double hiddenLayer[2];
    double outputLayer[1];
    double hiddenLayerBias[2] = {0.1, 0.2};
    double outputLayerBias[1] = {0.3};

    double *hiddenWeights[2];
    double *outputWeights[2];
    for (int i = 0; i < 2; i++)
    {
        hiddenWeights[i] = malloc(2 * sizeof(double));
        outputWeights[i] = malloc(1 * sizeof(double));
    }

    hiddenWeights[0][0] = 0.1;
    hiddenWeights[0][1] = 0.2;
    hiddenWeights[1][0] = 0.3;
    hiddenWeights[1][1] = 0.4;
    outputWeights[0][0] = 0.5;
    outputWeights[1][0] = 0.6;

    forward_pass_sequential(inputs, hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, numInputs, numHiddenNodes, numOutputs);

    assert(outputLayer[0] > 0 && outputLayer[0] < 1); // Basic range check

    for (int i = 0; i < 2; i++)
    {
        free(hiddenWeights[i]);
        free(outputWeights[i]);
    }

    printf("test_forward_pass passed.\n");
}

// Test backpropagation (sequential)
static void test_backpropagation()
{
    int numInputs = 2, numHiddenNodes = 2, numOutputs = 1;
    double inputs[] = {0.5, 0.5};
    double target[] = {1.0};
    double hiddenLayer[2];
    double outputLayer[1];
    double hiddenLayerBias[2] = {0.1, 0.2};
    double outputLayerBias[1] = {0.3};

    double *hiddenWeights[2];
    double *outputWeights[2];
    for (int i = 0; i < 2; i++)
    {
        hiddenWeights[i] = malloc(2 * sizeof(double));
        outputWeights[i] = malloc(1 * sizeof(double));
    }

    hiddenWeights[0][0] = 0.1;
    hiddenWeights[0][1] = 0.2;
    hiddenWeights[1][0] = 0.3;
    hiddenWeights[1][1] = 0.4;
    outputWeights[0][0] = 0.5;
    outputWeights[1][0] = 0.6;

    forward_pass_sequential(inputs, hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, numInputs, numHiddenNodes, numOutputs);
    backpropagation_sequential(inputs, target, hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, 0.1, numInputs, numHiddenNodes, numOutputs);

    // Check that the weights have been updated
    assert(hiddenWeights[0][0] != 0.1);
    assert(outputWeights[0][0] != 0.5);

    for (int i = 0; i < 2; i++)
    {
        free(hiddenWeights[i]);
        free(outputWeights[i]);
    }

    printf("test_backpropagation passed.\n");
}

// Main function to run all tests
int main()
{
    test_sigmoid();
    test_initialize_weights();
    test_forward_pass();
    test_backpropagation();
    printf("All tests passed.\n");
    return 0;
}

