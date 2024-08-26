#ifndef MPT_NN_UTILITY_H
#define MPT_NN_UTILITY_H

#include "mpt_nn.h"

// Function prototypes for utility functions
void load_mnist(double **training_inputs, double **training_outputs, int numTrainingSets, int numInputs, int numOutputs);
void initialize_weights(double **weights, int rows, int cols);
void initialize_bias(double bias[], int size);
void save_weights_and_biases(double **hiddenWeights, double **outputWeights,
                             double hiddenLayerBias[], double outputLayerBias[],
                             int numInputs, int numHiddenNodes, int numOutputs);
void visualize_mnist_digit(double *input, int numInputs);

#endif
