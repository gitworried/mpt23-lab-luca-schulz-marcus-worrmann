#ifndef MPT_NN_H
#define MPT_NN_H

double sigmoid(double x);

double dSigmoid(double x);

// Sequential inference
void forward_pass_sequential(double inputs[], double hiddenLayer[], double outputLayer[],
                             double hiddenLayerBias[], double outputLayerBias[],
                             double **hiddenWeights, double **outputWeights,
                             int numInputs, int numHiddenNodes, int numOutputs);

// Parallel inference using OpenMP
void forward_pass_parallel(double inputs[], double hiddenLayer[], double outputLayer[],
                           double hiddenLayerBias[], double outputLayerBias[],
                           double **hiddenWeights, double **outputWeights,
                           int numInputs, int numHiddenNodes, int numOutputs);

// SIMD inference using OpenMP
void forward_pass_simd(double inputs[], double hiddenLayer[], double outputLayer[],
                       double hiddenLayerBias[], double outputLayerBias[],
                       double **hiddenWeights, double **outputWeights,
                       int numInputs, int numHiddenNodes, int numOutputs);

// Sequential backpropagation
void backpropagation_sequential(double inputs[], double target[], double hiddenLayer[], double outputLayer[],
                                double hiddenLayerBias[], double outputLayerBias[],
                                double **hiddenWeights, double **outputWeights,
                                double lr, int numInputs, int numHiddenNodes, int numOutputs);

// Parallel backpropagation using OpenMP
void backpropagation_parallel(double inputs[], double target[], double hiddenLayer[], double outputLayer[],
                              double hiddenLayerBias[], double outputLayerBias[],
                              double **hiddenWeights, double **outputWeights,
                              double lr, int numInputs, int numHiddenNodes, int numOutputs);

// SIMD backpropagation using OpenMP
void backpropagation_simd(double inputs[], double target[], double hiddenLayer[], double outputLayer[],
                          double hiddenLayerBias[], double outputLayerBias[],
                          double **hiddenWeights, double **outputWeights,
                          double lr, int numInputs, int numHiddenNodes, int numOutputs);

#endif
