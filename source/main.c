/**
 * @file main.c
 * @authors Marcus Worrmann, Luca Schulz
 * @brief main function file using all the functions declared in mpt_nn.h and mpt_utility.h to create a functioning running neural network thats trained with the MNIST data set.
 * @version 1.0
 * @date 2024-08-30
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpt_nn.h"
#include "mpt_nn_utility.h"

int main(int argc, char *argv[])
{
    int visualize = 0;

    if (argc > 1 && strcmp(argv[1], "-v") == 0)
    {
        visualize = 1;
        argc--;
        argv++;
    }

    if (argc < 7)
    {
        fprintf(stderr, "Usage: %s [-v] <mode> <numTrainingSets> <numInputs> <numHiddenNodes> <numOutputs> <epochs> <learningRate>\n", argv[0]);
        fprintf(stderr, "Modes: sequential, parallel, simd\n");
        return EXIT_FAILURE;
    }

    int mode = 0;
    if (strcmp(argv[1], "sequential") == 0)
    {
        mode = 1;
    }
    else if (strcmp(argv[1], "parallel") == 0)
    {
        mode = 2;
    }
    else if (strcmp(argv[1], "simd") == 0)
    {
        mode = 3;
    }
    else
    {
        fprintf(stderr, "Invalid mode specified. Use 'sequential', 'parallel', or 'simd'.\n");
        return EXIT_FAILURE;
    }

    int numTrainingSets = atoi(argv[2]);
    int numInputs = atoi(argv[3]);
    int numHiddenNodes = atoi(argv[4]);
    int numOutputs = atoi(argv[5]);
    int epochs = atoi(argv[6]);
    double learningRate = atof(argv[7]);

    double *hiddenLayer = malloc(numHiddenNodes * sizeof(double));
    double *outputLayer = malloc(numOutputs * sizeof(double));
    double *hiddenLayerBias = malloc(numHiddenNodes * sizeof(double));
    double *outputLayerBias = malloc(numOutputs * sizeof(double));
    double **hiddenWeights = malloc(numInputs * sizeof(double *));
    double **outputWeights = malloc(numHiddenNodes * sizeof(double *));
    double **training_inputs = malloc(numTrainingSets * sizeof(double *));
    double **training_outputs = malloc(numTrainingSets * sizeof(double *));

    for (int i = 0; i < numInputs; i++)
        hiddenWeights[i] = malloc(numHiddenNodes * sizeof(double));
    for (int i = 0; i < numHiddenNodes; i++)
        outputWeights[i] = malloc(numOutputs * sizeof(double));
    for (int i = 0; i < numTrainingSets; i++)
        training_inputs[i] = malloc(numInputs * sizeof(double));
    for (int i = 0; i < numTrainingSets; i++)
        training_outputs[i] = malloc(numOutputs * sizeof(double));

    load_mnist(training_inputs, training_outputs, numTrainingSets, numInputs, numOutputs);

    initialize_weights(hiddenWeights, numInputs, numHiddenNodes);
    initialize_weights(outputWeights, numHiddenNodes, numOutputs);
    initialize_bias(hiddenLayerBias, numHiddenNodes);
    initialize_bias(outputLayerBias, numOutputs);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double totalLoss = 0.0;
        int correctPredictions = 0;

        for (int i = 0; i < numTrainingSets; i++)
        {
            int expectedLabel = 0;
            for (int j = 1; j < numOutputs; j++)
            {
                if (training_outputs[i][j] > training_outputs[i][expectedLabel])
                {
                    expectedLabel = j;
                }
            }

            if (visualize)
            {
                printf("Training on image %d (Epoch %d) - Expected output: %d\n", i + 1, epoch + 1, expectedLabel);
                visualize_mnist_digit(training_inputs[i], numInputs);
            }

            // Forward pass
            if (mode == 1)
            {
                forward_pass_sequential(training_inputs[i], hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, numInputs, numHiddenNodes, numOutputs);
            }
            else if (mode == 2)
            {
                forward_pass_parallel(training_inputs[i], hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, numInputs, numHiddenNodes, numOutputs);
            }
            else if (mode == 3)
            {
                forward_pass_simd(training_inputs[i], hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, numInputs, numHiddenNodes, numOutputs);
            }

            double loss = 0.0;
            for (int j = 0; j < numOutputs; j++)
            {
                loss += pow(training_outputs[i][j] - outputLayer[j], 2);
            }
            totalLoss += loss;

            int predictedLabel = 0;
            for (int j = 1; j < numOutputs; j++)
            {
                if (outputLayer[j] > outputLayer[predictedLabel])
                {
                    predictedLabel = j;
                }
            }
            int actualLabel = 0;
            for (int j = 1; j < numOutputs; j++)
            {
                if (training_outputs[i][j] > training_outputs[i][actualLabel])
                {
                    actualLabel = j;
                }
            }
            if (predictedLabel == actualLabel)
            {
                correctPredictions++;
            }

            if (mode == 1)
            {
                backpropagation_sequential(training_inputs[i], training_outputs[i], hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, learningRate, numInputs, numHiddenNodes, numOutputs);
            }
            else if (mode == 2)
            {
                backpropagation_parallel(training_inputs[i], training_outputs[i], hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, learningRate, numInputs, numHiddenNodes, numOutputs);
            }
            else if (mode == 3)
            {
                backpropagation_simd(training_inputs[i], training_outputs[i], hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, learningRate, numInputs, numHiddenNodes, numOutputs);
            }
        }

        double averageLoss = totalLoss / numTrainingSets;
        double accuracy = (double)correctPredictions / numTrainingSets * 100.0;
        printf("Epoch %d/%d - Loss: %.6f - Accuracy: %.2f%% (%d/%d)\n", epoch + 1, epochs, averageLoss, accuracy, correctPredictions, numTrainingSets);
    }

    free(hiddenLayer);
    free(outputLayer);
    free(hiddenLayerBias);
    free(outputLayerBias);

    for (int i = 0; i < numInputs; i++)
        free(hiddenWeights[i]);
    for (int i = 0; i < numHiddenNodes; i++)
        free(outputWeights[i]);
    for (int i = 0; i < numTrainingSets; i++)
        free(training_inputs[i]);
    for (int i = 0; i < numTrainingSets; i++)
        free(training_outputs[i]);

    free(hiddenWeights);
    free(outputWeights);
    free(training_inputs);
    free(training_outputs);

    return 0;
}
