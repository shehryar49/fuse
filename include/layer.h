#ifndef FUSE_LAYER_H_
#define FUSE_LAYER_H_

#include "matrix.h"
#include <random>

typedef void(*MatOp)(Matrix&,Matrix&);
typedef double(*LossFunction)(const Matrix&,const Matrix&);
typedef void(*LossFunctionDeriv)(Matrix&,const Matrix&,const Matrix&);



void sigmoid(Matrix&,Matrix&);
void sigmoid_derivative(Matrix&,Matrix&);
double MSE(const Matrix&,const Matrix&);
void MSE_derivative(Matrix&,const Matrix&,const Matrix&);

class Layer
{
public:
    Matrix weights;
    Matrix bias;
    MatOp fn;
    MatOp fnd;
    LossFunctionDeriv errfnd;
    //Auxiliary space to avoid allocating memory during training
    Matrix aL;
    Matrix dels; //deltas
    Matrix aL_dup; // auxiliary space to perform some operations on aL
    Matrix dout_dup;
    Matrix dw_transpose;
    Matrix deltas_chosen;
    Matrix inputsT_repeated;
    Layer(size_t num_neurons,size_t num_inputs,MatOp activation_fn,MatOp fnd,LossFunctionDeriv errfn);
    Layer();
    const Matrix& forward(const Matrix& inputs);
    void adapt_aux(size_t num_inputs,size_t next_neurons); // adapt auxiliary variables to sizes
    // deltas at output layer
    const Matrix& deltas_output(const Matrix& y);
    Matrix& deltas(const Matrix& dout,const Matrix& dw); // dw are weights of next layer
    void backpropagate(const Matrix& a,const Matrix& del,size_t idx,double lr = 0.1);
    void backpropagate_all(const Matrix& a,const Matrix& del,double lr = 0.1);  
};

#endif