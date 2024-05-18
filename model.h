#ifndef FUSE_MODEL_H_
#define FUSE_MODEL_H_

#include "layer.h"
#include "matrix.h"

extern MatOp sigmoid_activation[2];
extern void* mse_loss[2];

class Model
{
private:
    std::vector<Layer> layers;
    LossFunction lossfn;
    LossFunctionDeriv lossfnderiv;
public:
    Model(void* loss_fun[2] = mse_loss);
    void add_layer(size_t num_neurons,size_t num_inputs,MatOp activation[2]);
    void fit(const Matrix& inputs,const Matrix& y,size_t epochs,double lr = 0.1);
};

#endif