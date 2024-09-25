#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpt_nn_utility.h"

void load_mnist(double **training_inputs, double **training_outputs, int numTrainingSets, int numInputs, int numOutputs)
{
    FILE *imageFile = fopen("data/train-images.idx3-ubyte", "rb");
    FILE *labelFile = fopen("data/train-labels.idx1-ubyte", "rb");

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

void print_options(void)
{
    printf("\033[1;33mINFO:"
           " If default parameters are not set with -D, options -m, -t, -i, -h, -o, -e and -l are mandatory and require an argument\033[0m\n");
    printf("Available options:\n");
    printf("  -d, --dropOut     <dropOutRate>        Set the droput rate [Between 0.0 - 1.0]\n");
    printf("  -D, --defaultParams                    Set default paramaters for training\n");
    printf("  -e, --epochs      <numEpochs>          Set the number of epochs for training\n");
    printf("  -h, --hidden      <numHiddenNodes>     Set the number of hidden nodes\n");
    printf("  -i, --inputs      <numInputs>          Set the number of input nodes [784 for MNIST]\n");
    printf("  -l, --learning    <learningRate>       Set the learning rate [Between 0.0 - 1.0]\n");
    printf("  -m, --mode        <mode>               Set the mode [1: sequential][2: parallel][3: simd]\n");
    printf("  -n, --numThreads  <numThreads>         Set the number of threads to be used while executing a parallel region\n");
    printf("  -o, --outputs     <numOutput>          Set the number of output nodes[10 for MNIST]\n");
    printf("  -t, --trainsets   <numTrainingSets>    Set the number of training sets[max. 70000 for MNIST]\n");
    printf("  -v, --visualize                        Enable visualization\n");
    printf("  --help                                 Display this help and exit\n");
}

void apply_dropout(double *layer, int size, double dropout_rate)
{
    for (int i = 0; i < size; i++)
    {
        double random_val = (double)rand() / RAND_MAX;
        if (random_val < dropout_rate)
        {
            layer[i] = 0;
        }
        else
        {
            layer[i] *= 1.0 / (1.0 - dropout_rate);
        }
    }
}
