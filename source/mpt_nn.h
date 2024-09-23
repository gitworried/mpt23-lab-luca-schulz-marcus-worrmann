/**
 * @file mpt_nn.h
 * @authors Marcus Worrmann, Luca Schulz
 * @brief Header file for the core mpt_nn functions.
 * @version 1.0
 * @date 2024-08-30
 *
 * @copyright Copyright (c) 2024
 *
 * This file contains the declarations for the forward and backward pass
 * functions used for the mpt_nn neural network. The functions are implemented
 * in sequential, parallel, and SIMD modes using OpenMP for parallelization.
 */

#ifndef MPT_NN_H
#define MPT_NN_H

#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include "mpt_nn.h"
#include "mpt_nn_utility.h"
/**
 * @brief Defines the sigmoid activation function.
 *
 * Takes an input x and maps it to an output between 0 and 1 using the sigmoid formula.
 * Introduces a non-literal to the neural network allowing it to learn faster
 *
 * @param x input value.
 * @return result of applying the sigmoid function.
 */
double sigmoid(double x);

/**
 * @brief Defines the derivative of the sigmoid function.
 *
 * Calculates the derivative of the Sigmoid function.
 * Is essential during the backpropagation process. Calculates the gradient of the loss function with respect to the weights.
 *
 * @param x Input value.
 * @return Derivative of the sigmoid function.
 */
double dSigmoid(double x);

/**
 * @brief Performs a forward pass through the neural network sequentially.
 *
 * Computes the output of the hidden and output layers based on the input data, weights and biases (without parallelisation)
 *
 * @param inputs Input data for the neural network.
 * @param hiddenLayer Array storing the activations of the hidden layer.
 * @param outputLayer Array storing activations of the output layer.
 * @param hiddenLayerBias Array containing the biases for the hidden layer.
 * @param outputLayerBias Array containing the biases for the output layer.
 * @param hiddenWeights 2D array containing the weights between input and hidden layers.
 * @param outputWeights 2D array containing the weights between hidden and output layers.
 * @param numInputs Number of input nodes.
 * @param numHiddenNodes Number of hidden layer nodes.
 * @param numOutputs Number of output nodes.
 * @param dropout_rate Dropout rate for random neuron dropouts.
 */
void forward_pass_sequential(double inputs[], double hiddenLayer[], double outputLayer[],
                             double hiddenLayerBias[], double outputLayerBias[],
                             double **hiddenWeights, double **outputWeights,
                             int numInputs, int numHiddenNodes, int numOutputs, double dropout_rate);

/**
 * @brief Performs a forward pass through the neural network using OpenMP for parallelization.
 *
 * Essentially the same as forward_pass_sequential but uses OpenMP for parallelisation.
 * Paralleizes the computation of the activations for hidden and outout layers
 *
 * @param inputs Input data for the neural network.
 * @param hiddenLayer Array storing the activations of the hidden layer.
 * @param outputLayer Array storing activations of the output layer.
 * @param hiddenLayerBias Array containing the biases for the hidden layer.
 * @param outputLayerBias Array containing the biases for the output layer.
 * @param hiddenWeights 2D array containing the weights between input and hidden layers.
 * @param outputWeights 2D array containing the weights between hidden and output layers.
 * @param numInputs Number of input nodes.
 * @param numHiddenNodes Number of hidden layer nodes.
 * @param numOutputs Number of output nodes.
 * @param dropout_rate Dropout rate for random neuron dropouts.
 */
void forward_pass_parallel(double inputs[], double hiddenLayer[], double outputLayer[],
                           double hiddenLayerBias[], double outputLayerBias[],
                           double **hiddenWeights, double **outputWeights,
                           int numInputs, int numHiddenNodes, int numOutputs, double dropout_rate);

/**
 * @brief Performs a forward pass through the neural network using SIMD and OpenMP.
 *
 * Essentially the same as forward_pass_parallel but with furhter optimization utilizing SIMD(Single Instruction, Multiple Data) via OpenMP.
 * SIMD allows the CPU to perform the same operation on multiple data points
 *
 * @param inputs Input data for the neural network.
 * @param hiddenLayer Array storing the activations of the hidden layer.
 * @param outputLayer Array storing activations of the output layer.
 * @param hiddenLayerBias Array containing the biases for the hidden layer.
 * @param outputLayerBias Array containing the biases for the output layer.
 * @param hiddenWeights 2D array containing the weights between input and hidden layers.
 * @param outputWeights 2D array containing the weights between hidden and output layers.
 * @param numInputs Number of input nodes.
 * @param numHiddenNodes Number of hidden layer nodes.
 * @param numOutputs Number of output nodes.
 * @param dropout_rate Dropout rate for random neuron dropouts.
 */
void forward_pass_simd(double inputs[], double hiddenLayer[], double outputLayer[],
                       double hiddenLayerBias[], double outputLayerBias[],
                       double **hiddenWeights, double **outputWeights,
                       int numInputs, int numHiddenNodes, int numOutputs, double dropout_rate);

/**
 * @brief Performs a backpropagation through the neural network sequentially.
 *
 * Implements the backwardpropagation algorithm to update the weights and biases of the neural network sequentially
 * Calculates the error for each output node by comparing the actual output to the target output using the dsigmoid to compute the gradient.
 * Propagates the error backwards to the hidden layer using dsigmoid.
 * Adjusts the weights and biases for both layers using the gradient and learning rate.
 *
 *
 * @param inputs Input data for the neural network.
 * @param target The target output data for the neural network.
 * @param hiddenLayer Array storing the activations of the hidden layer.
 * @param outputLayer Array storing activations of the output layer.
 * @param hiddenLayerBias Array containing the biases for the hidden layer.
 * @param outputLayerBias Array containing the biases for the output layer.
 * @param hiddenWeights 2D array containing the weights between input and hidden layers.
 * @param outputWeights 2D array containing the weights between hidden and output layers.
 * @param lr Learning rate used for weight updates.
 * @param numInputs Number of input nodes.
 * @param numHiddenNodes Number of nodes in the hidden layer.
 * @param numOutputs Number of output nodes.
 * @param dropout_rate Dropout rate for random neuron dropouts.
 */
void backpropagation_sequential(double inputs[], double target[], double hiddenLayer[], double outputLayer[],
                                double hiddenLayerBias[], double outputLayerBias[],
                                double **hiddenWeights, double **outputWeights,
                                double lr, int numInputs, int numHiddenNodes, int numOutputs, double dropout_rate);

/**
 * @brief Performs a backpropagation through the neural network using OpenMP for parallelization.
 *
 * Essentially the same as backpropagation_sequential but now errors for the output layer are calculatet in parallel across multiple threads.
 * Hidden Layer errors are also calculated in parallel.
 * Final adjustments to weights and biases are distributed across multiple threads.
 *
 * @param inputs Input data for the neural network.
 * @param target The target output data for the neural network.
 * @param hiddenLayer Array storing the activations of the hidden layer.
 * @param outputLayer Array storing activations of the output layer.
 * @param hiddenLayerBias Array containing the biases for the hidden layer.
 * @param outputLayerBias Array containing the biases for the output layer.
 * @param hiddenWeights 2D array containing the weights between input and hidden layers.
 * @param outputWeights 2D array containing the weights between hidden and output layers.
 * @param lr Learning rate used for weight updates.
 * @param numInputs Number of input nodes.
 * @param numHiddenNodes Number of nodes in the hidden layer.
 * @param numOutputs Number of output nodes.
 * @param dropout_rate Dropout rate for random neuron dropouts.
 */
void backpropagation_parallel(double inputs[], double target[], double hiddenLayer[], double outputLayer[],
                              double hiddenLayerBias[], double outputLayerBias[],
                              double **hiddenWeights, double **outputWeights,
                              double lr, int numInputs, int numHiddenNodes, int numOutputs, double dropout_rate);

/**
 * @brief Performs a backward pass (backpropagation) through the neural network using SIMD and OpenMP.
 *
 * Essentially the same as backpropagation_parallel but with furhter optimization utilizing SIMD(Single Instruction, Multiple Data) via OpenMP.
 * Output layer errors are calculated in parallel across multiple threads and optimized usind SIMD.
 * Hidden layer errors are calculated in parallel across multiple threads and ptimized usind SIMD.
 * Weights and biases are updated in parrallel across multiple threads and optimized using SIMD.
 *
 *
 * @param inputs Input data for the neural network.
 * @param target The target output data for the neural network.
 * @param hiddenLayer Array storing the activations of the hidden layer.
 * @param outputLayer Array storing activations of the output layer.
 * @param hiddenLayerBias Array containing the biases for the hidden layer.
 * @param outputLayerBias Array containing the biases for the output layer.
 * @param hiddenWeights 2D array containing the weights between input and hidden layers.
 * @param outputWeights 2D array containing the weights between hidden and output layers.
 * @param lr Learning rate used for weight updates.
 * @param numInputs Number of input nodes.
 * @param numHiddenNodes Number of nodes in the hidden layer.
 * @param numOutputs Number of output nodes.
 * @param dropout_rate Dropout rate for random neuron dropouts.
 */
void backpropagation_simd(double inputs[], double target[], double hiddenLayer[], double outputLayer[],
                          double hiddenLayerBias[], double outputLayerBias[],
                          double **hiddenWeights, double **outputWeights,
                          double lr, int numInputs, int numHiddenNodes, int numOutputs, double dropout_rate);

#endif // MPT_NN_H
