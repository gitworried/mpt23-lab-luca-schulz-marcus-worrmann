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
#include <string.h>
#include <stdbool.h>
#include <getopt.h>
#include "mpt_nn.h"
#include "mpt_nn_utility.h"

int main(int argc, char *argv[])
{
    int opt;

    int visualize = 0;
    int mode = 1;
    int numTrainingSets = 10000;
    int numInputs = 784;
    int numHiddenNodes = 10;
    int numOutputs = 10;
    int epochs = 10;
    int numThreads = 1;
    double learningRate = 0.01;
    double dropoutRate = 0.0;
    size_t counter = 0;

    bool nProvided = false;
    bool dProvided = false;

    struct option longopt[] =
        {
            {"help", no_argument, NULL, '?'},
            {"defaultParams", no_argument, NULL, 'D'},
            {"epochs", required_argument, NULL, 'e'},
            {"hidden", required_argument, NULL, 'h'},
            {"inputs", required_argument, NULL, 'i'},
            {"outputs", required_argument, NULL, 'o'},
            {"mode", required_argument, NULL, 'm'},
            {"dropOut", required_argument, NULL, 'd'},
            {"learning", required_argument, NULL, 'l'},
            {"numThreads", required_argument, NULL, 'n'},
            {"trainsets", required_argument, NULL, 't'},
            {"visualize", no_argument, NULL, 'v'},
            {0, 0, 0, 0}};

    const char *optstring = "Dd:e:h:i:l:m:n:o:t:v";

    opterr = 0;

    while ((opt = getopt_long(argc, argv, optstring, longopt, NULL)) != -1)
    {
        switch (opt)
        {
        case 'D':
            mode = 1;
            numTrainingSets = 10000;
            numInputs = 784;
            numHiddenNodes = 128;
            numOutputs = 10;
            epochs = 10;
            learningRate = 0.01;
            dropoutRate = 0.0;
            printf("\033[1;33m************************** INFO ***************************\n");
            printf("* Training mpt_nn with default parameters                 *\n");
            printf("* %-25s %-29s *\n", "Mode[1]:", "sequential");
            printf("* %-25s %-29d *\n", "Training sets:", numTrainingSets);
            printf("* %-25s %-29d *\n", "Input nodes:", numInputs);
            printf("* %-25s %-29d *\n", "Hidden nodes:", numHiddenNodes);
            printf("* %-25s %-29d *\n", "Output nodes:", numOutputs);
            printf("* %-25s %-29d *\n", "Epochs:", epochs);
            printf("* %-25s %-29.6f *\n", "Learning rate:", learningRate);
            printf("* %-25s %-29.6f *\n", "Dropout rate:", dropoutRate);
            if (nProvided)
            {
                printf("* %-25s %-25d *\n", "Number of Threads:", numThreads);
            }
            printf("***********************************************************\033[0m\n");
            dProvided = true;
            break;
        case 'd':
            dropoutRate = atof(optarg);
            break;
        case 'e':
            epochs = atoi(optarg);
            counter++;
            break;
        case 'h':
            numHiddenNodes = atoi(optarg);
            counter++;
            break;
        case 'i':
            numInputs = atoi(optarg);
            counter++;
            break;
        case 'l':
            learningRate = atof(optarg);
            counter++;
            break;
        case 'n':
            numThreads = atoi(optarg);
            nProvided = true;
            break;
        case 'm':
            mode = atoi(optarg);
            counter++;
            break;
        case 'o':
            numOutputs = atoi(optarg);
            counter++;
            break;
        case 't':
            numTrainingSets = atoi(optarg);
            counter++;
            break;
        case 'v':
            visualize = 1;
            break;
        case '?':
            print_options();
            exit(EXIT_FAILURE);
        default:
            exit(EXIT_FAILURE);
        }
    }

    printf("%ld\n", counter);

    if (!dProvided && counter < 7)
    {
        printf("\033[1;31mMissing arguments. Please select -D for default parameters or set them yourself with the available options.\n");
        printf("-? or --help to display all available options.\033[0m\n");
        print_options();
        exit(EXIT_FAILURE);
    }

    if (!dProvided)
    {
        printf("\033[1;33m************************** INFO ***************************\n");
        printf("* Training mpt_nn with parameters:                        *\n");

        switch (mode)
        {
        case 1:
            printf("* %-25s %-29s *\n", "Mode[1]:", "sequential");
            break;
        case 2:
            printf("* %-25s %-29s *\n", "Mode[2]:", "parallel");
            break;
        case 3:
            printf("* %-25s %-29s *\n", "Mode[3]:", "SIMD");
            break;
        default:
            printf("* %-25s %-29d *\n", "Mode:", mode);
            break;
        }

        printf("* %-25s %-29d *\n", "Training sets:", numTrainingSets);
        printf("* %-25s %-29d *\n", "Input nodes:", numInputs);
        printf("* %-25s %-29d *\n", "Hidden nodes:", numHiddenNodes);
        printf("* %-25s %-29d *\n", "Output nodes:", numOutputs);
        printf("* %-25s %-29d *\n", "Epochs:", epochs);
        printf("* %-25s %-29.6f *\n", "Learning rate:", learningRate);
        printf("* %-25s %-29.6f *\n", "Dropout rate:", dropoutRate);
        if (nProvided)
        {
            printf("* %-25s %-29d *\n", "Number of Threads:", numThreads);
        }
        printf("***********************************************************\n\033[0m");
    }

    if (nProvided)
    {
        omp_set_num_threads(numThreads);
    }

    double *hiddenLayer = malloc(numHiddenNodes * sizeof(double));
    double *outputLayer = malloc(numOutputs * sizeof(double));
    double *hiddenLayerBias = malloc(numHiddenNodes * sizeof(double));
    double *outputLayerBias = malloc(numOutputs * sizeof(double));
    double **hiddenWeights = malloc(numInputs * sizeof(double *));
    double **outputWeights = malloc(numHiddenNodes * sizeof(double *));
    double **training_inputs = malloc(numTrainingSets * sizeof(double *));
    double **training_outputs = malloc(numTrainingSets * sizeof(double *));

    for (int i = 0; i < numHiddenNodes; i++)
        hiddenLayer[i] = 0.0;
    for (int i = 0; i < numOutputs; i++)
        outputLayer[i] = 0.0;
    for (int i = 0; i < numHiddenNodes; i++)
        hiddenLayerBias[i] = 0.0;
    for (int i = 0; i < numOutputs; i++)
        outputLayerBias[i] = 0.0;

    for (int i = 0; i < numInputs; i++)
    {
        hiddenWeights[i] = malloc(numHiddenNodes * sizeof(double));
        for (int j = 0; j < numHiddenNodes; j++)
        {
            hiddenWeights[i][j] = 0.0;
        }
    }
    for (int i = 0; i < numHiddenNodes; i++)
    {
        outputWeights[i] = malloc(numOutputs * sizeof(double));
        for (int j = 0; j < numOutputs; j++)
        {
            outputWeights[i][j] = 0.0;
        }
    }
    for (int i = 0; i < numTrainingSets; i++)
    {
        training_inputs[i] = malloc(numInputs * sizeof(double));
        for (int j = 0; j < numInputs; j++)
        {
            training_inputs[i][j] = 0.0;
        }
    }
    for (int i = 0; i < numTrainingSets; i++)
    {
        training_outputs[i] = malloc(numOutputs * sizeof(double));
        for (int j = 0; j < numOutputs; j++)
        {
            training_outputs[i][j] = 0.0;
        }
    }

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

            if (mode == 1)
            {
                forward_pass_sequential(training_inputs[i], hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, numInputs, numHiddenNodes, numOutputs, dropoutRate);
            }
            else if (mode == 2)
            {
                forward_pass_parallel(training_inputs[i], hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, numInputs, numHiddenNodes, numOutputs, dropoutRate);
            }
            else if (mode == 3)
            {
                forward_pass_simd(training_inputs[i], hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, numInputs, numHiddenNodes, numOutputs, dropoutRate);
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
                backpropagation_sequential(training_inputs[i], training_outputs[i], hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, learningRate, numInputs, numHiddenNodes, numOutputs, dropoutRate);
            }
            else if (mode == 2)
            {
                backpropagation_parallel(training_inputs[i], training_outputs[i], hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, learningRate, numInputs, numHiddenNodes, numOutputs, dropoutRate);
            }
            else if (mode == 3)
            {
                backpropagation_simd(training_inputs[i], training_outputs[i], hiddenLayer, outputLayer, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, learningRate, numInputs, numHiddenNodes, numOutputs, dropoutRate);
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
