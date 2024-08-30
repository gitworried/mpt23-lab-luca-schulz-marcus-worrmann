/**
 * @file mpt_nn_utility.h
 * @authors Marcus Worrmann, Luca Schulz
 * @brief Header file for utility functions used for the mpt_nn.
 * @version 1.0
 * @date 2024-08-30
 *
 * @copyright Copyright (c) 2024
 *
 *
 * This file contains the declarations for various utility functions
 * used to support the mpt_nn operations, such as loading data,
 * initializing weights and biases, saving model parameters, and visualizing data.
 */
#ifndef MPT_NN_UTILITY_H
#define MPT_NN_UTILITY_H

#include "mpt_nn.h"

/**
 * @brief Loads the MNIST dataset into the mpt_nn input and output arrays.
 *
 * Reads MNIST images and lables from the data files.
 * Normalizes image pixel values between 0 and 1.
 * Converts the labels into a one-hot encoded format for training
 *
 * @param training_inputs 2D array to store the input data for training.
 * @param training_outputs 2D array to store the expected output data for training.
 * @param numTrainingSets Number of training examples to load.
 * @param numInputs Number of input nodes (pixels per image).
 * @param numOutputs Number of output nodes (number of classes).
 */
void load_mnist(double **training_inputs, double **training_outputs, int numTrainingSets, int numInputs, int numOutputs);

/**
 * @brief Initializes the weights of the mpt_nn with random values.
 *
 * Each weight is assigned a small random value.
 * Ensures that the neural network beginns with a diverse set of parameters preventing errors and allowing effective learning during training.
 *
 * @param weights 2D array to store the weights.
 * @param rows Number of rows in the weight matrix (input nodes).
 * @param cols Number of columns in the weight matrix (output nodes).
 */
void initialize_weights(double **weights, int rows, int cols);

/**
 * @brief Initializes the biases of the mpt_nnth random values.
 *
 * Each bias is initialized with a small random value.
 * Ensures that neural networks begins with a diverse set of biases also preventing errors and allowing effective learning during training.
 *
 * @param bias Array to store the biases.
 * @param size Number of biases to initialize.
 */
void initialize_bias(double bias[], int size);

/**
 * @brief Saves the weights and biases of the mpt_nn to a file.
 *
 * This function is not used at the moment.
 * It allows to persist the Parameters of the mpt_nn after training for future use if needed.
 *
 * @param hiddenWeights 2D array containing the weights between the input and hidden layers.
 * @param outputWeights 2D array containing the weights between the hidden and output layers.
 * @param hiddenLayerBias Array containing the biases for the hidden layer.
 * @param outputLayerBias Array containing the biases for the output layer.
 * @param numInputs Number of input nodes.
 * @param numHiddenNodes Number of nodes in the hidden layer.
 * @param numOutputs Number of output nodes.
 */
void save_weights_and_biases(double **hiddenWeights, double **outputWeights,
                             double hiddenLayerBias[], double outputLayerBias[],
                             int numInputs, int numHiddenNodes, int numOutputs);

/**
 * @brief Visualizes an MNIST digit by printing it to the console.
 *
 * Interprets the pixel values as a 28x28 grid.
 * Speciall characters are used for different pixel valuse
 * # for darker values
 * + for middle values
 * . for lighter values
 *
 * @param input Array containing the pixel values of the MNIST digit.
 * @param numInputs Number of input nodes (pixels per image).
 */
void visualize_mnist_digit(double *input, int numInputs);

#endif // MPT_NN_UTILITY_H
