#include <iostream>
#include "model.h"


size_t max_idx(double* arr,size_t n)
{
	double max_val = 0;
	double idx = 0;
	for(size_t i=0;i<n;i++)
	{
		if(arr[i] > max_val || i == 0)
		{
			idx = i;
			max_val = arr[i];
		}
	}
	return idx;
}
double accuracy(const Matrix& pred,const Matrix& y)
{
	int correct = 0;
	for(size_t i=0;i<y.rows;i++)
	{
		correct += (max_idx(pred.matrix[i],pred.cols) == max_idx(y.matrix[i],y.cols));
	}
	return (double)correct / y.rows;
}

MatOp sigmoid_activation[2] = {sigmoid,sigmoid_derivative};
void* mse_loss[2] = {(void*)&MSE,(void*)&MSE_derivative};


Model::Model(void* loss_fun[2])
{
    lossfn = (LossFunction)loss_fun[0];
    lossfnderiv = (LossFunctionDeriv)loss_fun[1];
}
void Model::add_layer(size_t num_neurons,size_t num_inputs,MatOp activation[2])
{
    layers.push_back(Layer(num_neurons,num_inputs,activation[0],activation[1],lossfnderiv));
}
void Model::fit(const Matrix& inputs,const Matrix& y,size_t epochs,double lr)
{
    double loss;
    for(size_t k = 0;k<layers.size();k++)
    {
        if(k == layers.size()-1)
            layers[k].adapt_aux(y.rows,1);
        else
            layers[k].adapt_aux(y.rows, layers[k+1].weights.cols);
    }
    for(size_t i=1;i<=epochs;i++)
    {
        for(size_t i=0;i<layers.size();i++)
        {
            if(i > 0)
                layers[i].forward(layers[i-1].aL);
            else
                layers[i].forward(inputs);
        }
        loss = lossfn(layers.back().aL,y);
        //std::cout << "Loss: "<<loss<<"  Accuracy:"<<accuracy(layers.back().aL,y) <<std::endl;
        //backpropagate
        size_t idx = rand()%y.rows;
        const Matrix* last_delta = nullptr;
        for(size_t i=1;i<=layers.size();i++)
        {
            size_t curr = layers.size() - i;
            if(i == 1)
            {
                const Matrix& deltas = layers[curr].deltas_output(y);
                last_delta = &deltas;
            }
            else 
            {
                const Matrix& deltas = layers[curr].deltas(*last_delta, layers[curr+1].weights);
                last_delta = &deltas;
            }

            if(curr == 0)
                layers[curr].backpropagate(inputs,*last_delta,idx,lr);
            else
                layers[curr].backpropagate(layers[curr-1].aL,*last_delta,idx,lr);
        }
    }
    loss = lossfn(layers.back().aL,y);
    std::cout << "Loss: "<<loss<<"  Accuracy:"<<accuracy(layers.back().aL,y) <<std::endl;
}
