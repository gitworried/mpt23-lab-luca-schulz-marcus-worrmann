#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpt_nn_utility.h"

void load_mnist(double **training_inputs, double **training_outputs, int numTrainingSets, int numInputs, int numOutputs)
{
    FILE *imageFile = fopen("train-images.idx3-ubyte", "rb");
    FILE *labelFile = fopen("train-labels.idx1-ubyte", "rb");

    if (imageFile == NULL || labelFile == NULL)
    {
        perror("Error opening MNIST files");
        exit(1);
    }

    fseek(imageFile, 16, SEEK_SET);
    fseek(labelFile, 8, SEEK_SET);

    for (int i = 0; i < numTrainingSets; i++)
    {
        for (int j = 0; j < numInputs; j++)
        {
            unsigned char pixel = 0;
            if (fread(&pixel, sizeof(unsigned char), 1, imageFile) != 1)
            {
                perror("Error reading image file");
                exit(1);
            }
            training_inputs[i][j] = pixel / 255.0;
        }

        unsigned char label = 0;
        if (fread(&label, sizeof(unsigned char), 1, labelFile) != 1)
        {
            perror("Error reading label file");
            exit(1);
        }

        for (int k = 0; k < numOutputs; k++)
        {
            training_outputs[i][k] = (label == k) ? 1.0 : 0.0;
        }
    }

    fclose(imageFile);
    fclose(labelFile);
}

void initialize_weights(double **weights, int rows, int cols)
{
    srand((unsigned int)time(NULL));
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            weights[i][j] = (rand() / (double)RAND_MAX) - 0.5;
        }
    }
}

void initialize_bias(double bias[], int size)
{
    srand((unsigned int)time(NULL));
    for (int i = 0; i < size; i++)
    {
        bias[i] = (rand() / (double)RAND_MAX) - 0.5;
    }
}

void save_weights_and_biases(double **hiddenWeights, double **outputWeights,
                             double hiddenLayerBias[], double outputLayerBias[],
                             int numInputs, int numHiddenNodes, int numOutputs)
{
    FILE *file = fopen("weights_biases.txt", "w");

    if (file == NULL)
    {
        printf("Error opening file to save weights and biases.\n");
        return;
    }

    fprintf(file, "Hidden Weights:\n");
    for (int i = 0; i < numInputs; i++)
    {
        for (int j = 0; j < numHiddenNodes; j++)
        {
            fprintf(file, "%f ", hiddenWeights[i][j]);
        }
        fprintf(file, "\n");
    }

    fprintf(file, "\nOutput Weights:\n");
    for (int i = 0; i < numHiddenNodes; i++)
    {
        for (int j = 0; j < numOutputs; j++)
        {
            fprintf(file, "%f ", outputWeights[i][j]);
        }
        fprintf(file, "\n");
    }

    fprintf(file, "\nHidden Layer Biases:\n");
    for (int i = 0; i < numHiddenNodes; i++)
    {
        fprintf(file, "%f ", hiddenLayerBias[i]);
    }

    fprintf(file, "\nOutput Layer Biases:\n");
    for (int i = 0; i < numOutputs; i++)
    {
        fprintf(file, "%f ", outputLayerBias[i]);
    }

    fclose(file);
}

void visualize_mnist_digit(double *input, int numInputs)
{
    for (int i = 0; i < numInputs; i++)
    {
        if (i % 28 == 0)
        {
            printf("\n");
        }
        if (input[i] > 0.5)
        {
            printf("#");
        }
        else if (input[i] > 0.2)
        {
            printf("+");
        }
        else
        {
            printf(".");
        }
    }
    printf("\n\n");
}
