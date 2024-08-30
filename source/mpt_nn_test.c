/**
 * @file mpt_nn_test.c
 * @authors Marcus Worrmann, Luca Schulz
 * @brief Unit tests for the mpt_nn functions defined in mpt_nn.h and mpt_nn_utility.h
 * @version 1.0
 * @date 2024-08-30
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include <immintrin.h>
#include "mpt_nn.h"
#include "mpt_nn_utility.h"

/**
 * @brief Allocates memory for a 2D array (weights matrix).
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return double** Pointer to the allocated 2D array.
 *
 * Allocates memory for a 2D array that will hold the weights of a neural network layer.
 */
static double **allocate_weights(int rows, int cols)
{
    double **weights = malloc(rows * sizeof(double *));
    if (!weights)
    {
        fprintf(stderr, "Failed to allocate memory for weights\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++)
    {
        weights[i] = malloc(cols * sizeof(double));
        if (!weights[i])
        {
            fprintf(stderr, "Failed to allocate memory for weights[%d]\n", i);
            for (int j = 0; j < i; j++)
            {
                free(weights[j]);
            }
            free(weights);
            exit(EXIT_FAILURE);
        }
    }
    return weights;
}

/**
 * @brief Frees the memory allocated for a 2D array (weights matrix).
 *
 * Deallocates the memory that was previously allocated for a 2D array used to store weights.
 *
 * @param weights Pointer to the 2D array to be freed.
 * @param rows Number of rows.
 *
 *
 */
static void free_weights(double **weights, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(weights[i]);
    }
    free(weights);
}

/**
 * @brief Tests the sigmoid function.
 *
 * Tests the sigmoid function with known input values and asserts the correctness of the output.
 * E.g sigmoid of 0 should be 0.5
 */
static void test_sigmoid()
{
    assert(sigmoid(0) == 0.5);
    assert(sigmoid(100) > 0.999);
    assert(sigmoid(-100) < 0.001);
    printf("test_sigmoid passed.\n");
}

/**
 * @brief Tests the initialize_weights function.
 *
 * Verifies that the initialize_weights function correctly initializes the weights to small random values in an expected range.
 *
 * Initializes a small matrix and checks, if all values are initialized to non zero small values.
 * Asserts the correct allocation of the matrix
 */
static void test_initialize_weights()
{
    int rows = 2, cols = 3;
    double **weights = allocate_weights(rows, cols);

    initialize_weights(weights, rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            assert(weights[i][j] >= -0.5 && weights[i][j] <= 0.5);
        }
    }
    free_weights(weights, rows);

    printf("test_initialize_weights passed.\n");
}

/**
 * @brief Tests the forward_pass_sequential function.
 *
 * Checks that the forward_pass_sequential function correctly computes the output of the mpt_nn neural network.
 *
 * Initializes values for input, hidden layer etc....
 * allocates memory for weigths.
 * Calls forward_pass_sequential.
 * Asserts that the output layer containes the correct values in a specific range.
 *
 */
static void test_forward_pass()
{
    int numInputs = 2, numHiddenNodes = 2, numOutputs = 1;
    double inputs[] = {0.5, 0.5};
    double hiddenLayer[2];
    double outputLayer[1];
    double hiddenLayerBias[2] = {0.1, 0.2};
    double outputLayerBias[1] = {0.3};

    double **hiddenWeights = allocate_weights(numInputs, numHiddenNodes);
    double **outputWeights = allocate_weights(numHiddenNodes, numOutputs);

    hiddenWeights[0][0] = 0.1;
    hiddenWeights[0][1] = 0.2;
    hiddenWeights[1][0] = 0.3;
    hiddenWeights[1][1] = 0.4;
    outputWeights[0][0] = 0.5;
    outputWeights[1][0] = 0.6;

    forward_pass_sequential(inputs, hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, numInputs, numHiddenNodes, numOutputs);

    assert(outputLayer[0] > 0 && outputLayer[0] < 1);
    free_weights(hiddenWeights, numInputs);
    free_weights(outputWeights, numHiddenNodes);

    printf("test_forward_pass (sequential) passed.\n");
}

/**
 * @brief Tests the forward_pass_parallel function.
 *
 * Cecks that the forward_pass_parallel function correctly computes the output of the mpt_nn neural network in parallel.
 *
 * Initializes values for input, hidden layer etc....
 * allocates memory for weigths.
 * Calls forward_pass_parallel.
 * Asserts that the output layer containes the correct values in a specific range.
 */
static void test_forward_pass_parallel()
{
    int numInputs = 2, numHiddenNodes = 2, numOutputs = 1;
    double inputs[] = {0.5, 0.5};
    double hiddenLayer[2];
    double outputLayer[1];
    double hiddenLayerBias[2] = {0.1, 0.2};
    double outputLayerBias[1] = {0.3};

    double **hiddenWeights = allocate_weights(numInputs, numHiddenNodes);
    double **outputWeights = allocate_weights(numHiddenNodes, numOutputs);

    hiddenWeights[0][0] = 0.1;
    hiddenWeights[0][1] = 0.2;
    hiddenWeights[1][0] = 0.3;
    hiddenWeights[1][1] = 0.4;
    outputWeights[0][0] = 0.5;
    outputWeights[1][0] = 0.6;

    forward_pass_parallel(inputs, hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, numInputs, numHiddenNodes, numOutputs);

    assert(outputLayer[0] > 0 && outputLayer[0] < 1);

    free_weights(hiddenWeights, numInputs);
    free_weights(outputWeights, numHiddenNodes);

    printf("test_forward_pass (parallel) passed.\n");
}

/**
 * @brief Tests the forward_pass_simd function.
 *
 * Checks that the forward_pass_simd function correctly computes the output of the mpt_nn neural network using SIMD.
 *
 * Initializes values for input, hidden layer etc....
 * allocates memory for weigths.
 * Calls forward_pass_simd.
 * Asserts that the output layer containes the correct values in a specific range.
 *
 */
static void test_forward_pass_simd()
{
    int numInputs = 2, numHiddenNodes = 2, numOutputs = 1;
    double inputs[] = {0.5, 0.5};
    double hiddenLayer[2];
    double outputLayer[1];
    double hiddenLayerBias[2] = {0.1, 0.2};
    double outputLayerBias[1] = {0.3};

    double **hiddenWeights = allocate_weights(numInputs, numHiddenNodes);
    double **outputWeights = allocate_weights(numHiddenNodes, numOutputs);

    hiddenWeights[0][0] = 0.1;
    hiddenWeights[0][1] = 0.2;
    hiddenWeights[1][0] = 0.3;
    hiddenWeights[1][1] = 0.4;
    outputWeights[0][0] = 0.5;
    outputWeights[1][0] = 0.6;

    forward_pass_simd(inputs, hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, numInputs, numHiddenNodes, numOutputs);

    assert(outputLayer[0] > 0 && outputLayer[0] < 1);

    free_weights(hiddenWeights, numInputs);
    free_weights(outputWeights, numHiddenNodes);

    printf("test_forward_pass (SIMD) passed.\n");
}

/**
 * @brief Tests the backpropagation_sequential function.
 *
 * Checks that the backpropagation_sequential function correctly updates the weights and biases of the mpt_nn neural network.
 *
 * Initializes values for input, hidden layer etc....
 * allocates memory for weigths.
 * Calls forward_pass_sequential and baclpropagation_sequential.
 * Asserts that the weigths and biases are updated correctly
 *
 */
static void test_backpropagation()
{
    int numInputs = 2, numHiddenNodes = 2, numOutputs = 1;
    double inputs[] = {0.5, 0.5};
    double target[] = {1.0};
    double hiddenLayer[2];
    double outputLayer[1];
    double hiddenLayerBias[2] = {0.1, 0.2};
    double outputLayerBias[1] = {0.3};

    double **hiddenWeights = allocate_weights(numInputs, numHiddenNodes);
    double **outputWeights = allocate_weights(numHiddenNodes, numOutputs);

    hiddenWeights[0][0] = 0.1;
    hiddenWeights[0][1] = 0.2;
    hiddenWeights[1][0] = 0.3;
    hiddenWeights[1][1] = 0.4;
    outputWeights[0][0] = 0.5;
    outputWeights[1][0] = 0.6;

    forward_pass_sequential(inputs, hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, numInputs, numHiddenNodes, numOutputs);
    backpropagation_sequential(inputs, target, hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, 0.1, numInputs, numHiddenNodes, numOutputs);

    assert(hiddenWeights[0][0] != 0.1);
    assert(outputWeights[0][0] != 0.5);

    free_weights(hiddenWeights, numInputs);
    free_weights(outputWeights, numHiddenNodes);

    printf("test_backpropagation (sequential) passed.\n");
}

/**
 * @brief Tests the backpropagation_parallel function.
 *
 * Checks that the backpropagation_parallel function correctly updates the weights and biases of the mpt_nn neural network in parallel.
 *
 * Initializes values for input, hidden layer etc....
 * allocates memory for weigths.
 * Calls forward_pass_parallel and backpasspropagation_parallel.
 * Asserts that the weigths and biases are updated correctly
 */
static void test_backpropagation_parallel()
{
    int numInputs = 2, numHiddenNodes = 2, numOutputs = 1;
    double inputs[] = {0.5, 0.5};
    double target[] = {1.0};
    double hiddenLayer[2];
    double outputLayer[1];
    double hiddenLayerBias[2] = {0.1, 0.2};
    double outputLayerBias[1] = {0.3};

    double **hiddenWeights = allocate_weights(numInputs, numHiddenNodes);
    double **outputWeights = allocate_weights(numHiddenNodes, numOutputs);

    hiddenWeights[0][0] = 0.1;
    hiddenWeights[0][1] = 0.2;
    hiddenWeights[1][0] = 0.3;
    hiddenWeights[1][1] = 0.4;
    outputWeights[0][0] = 0.5;
    outputWeights[1][0] = 0.6;

    forward_pass_parallel(inputs, hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, numInputs, numHiddenNodes, numOutputs);
    backpropagation_parallel(inputs, target, hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, 0.1, numInputs, numHiddenNodes, numOutputs);

    assert(hiddenWeights[0][0] != 0.1);
    assert(outputWeights[0][0] != 0.5);

    free_weights(hiddenWeights, numInputs);
    free_weights(outputWeights, numHiddenNodes);

    printf("test_backpropagation (parallel) passed.\n");
}

/**
 * @brief Tests the backpropagation_simd function.
 *
 * This function checks that the backpropagation_simd function correctly updates the weights and biases of a simple neural network using SIMD.
 *
 * Initializes values for input, hidden layer etc....
 * allocates memory for weigths.
 * Calls forward_pass_simd and backpropagation_simd.
 * Asserts that the weigths and biases are updated correctly
 */
static void test_backpropagation_simd()
{
    int numInputs = 2, numHiddenNodes = 2, numOutputs = 1;
    double inputs[] = {0.5, 0.5};
    double target[] = {1.0};
    double hiddenLayer[2];
    double outputLayer[1];
    double hiddenLayerBias[2] = {0.1, 0.2};
    double outputLayerBias[1] = {0.3};

    double **hiddenWeights = allocate_weights(numInputs, numHiddenNodes);
    double **outputWeights = allocate_weights(numHiddenNodes, numOutputs);

    hiddenWeights[0][0] = 0.1;
    hiddenWeights[0][1] = 0.2;
    hiddenWeights[1][0] = 0.3;
    hiddenWeights[1][1] = 0.4;
    outputWeights[0][0] = 0.5;
    outputWeights[1][0] = 0.6;

    forward_pass_simd(inputs, hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, numInputs, numHiddenNodes, numOutputs);
    backpropagation_simd(inputs, target, hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, 0.1, numInputs, numHiddenNodes, numOutputs);

    assert(hiddenWeights[0][0] != 0.1);
    assert(outputWeights[0][0] != 0.5);

    free_weights(hiddenWeights, numInputs);
    free_weights(outputWeights, numHiddenNodes);

    printf("test_backpropagation (SIMD) passed.\n");
}

/**
 * @brief Main function for running all unit tests.
 *
 * Calls all test functions defined above and reports the results.
 *
 * @return Returns 0 if all tests pass.
 */
int main()
{
    test_sigmoid();
    test_initialize_weights();
    test_forward_pass();
    test_forward_pass_parallel();
    test_forward_pass_simd();
    test_backpropagation();
    test_backpropagation_parallel();
    test_backpropagation_simd();
    printf("All tests passed.\n");
    return 0;
}
