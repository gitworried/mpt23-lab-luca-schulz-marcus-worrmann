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

/**
 * @brief Prints the awailable command line options to the terminal.
 *
 */
void print_options(void);
/**
 * @brief Applys a dropout to a layer of neurons.
 *
 * This function randomly drops out (sets to 0) a portion of the neurons in a layer
 * during training, based on the specified dropout rate. The remaining active neurons
 * are scaled by a factor of `1 / (1 - dropout_rate)` to maintain the overall output
 * distribution.
 *
 * INFO: Amount of hidden nodes has to be taken into considaration when choosing the dropout rate.
 * For example: When having 128 hidden nodes a max. dropout rate of 0.2 should be choosen to achieve optimal results.
 * anything higher will lead to too many dropouts for such an amount of hidden nodes.
 *
 * @param layer Pointer to the array representing the layer's neuron activations.
 * @param size Number of neurons in the layer.
 * @param dropout_rate Probability of dropping a neuron (value between 0.0 and 1.0).
 */
void apply_dropout(double *layer, int size, double dropout_rate);

#endif // MPT_NN_UTILITY_H
