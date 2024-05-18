#include <cstdio>
#include <iostream>
#include <cmath>
#include <string.h>
#include <random>
#include "matrix.h"


using namespace std;



typedef void(*Callback)(Matrix&,Matrix&);
typedef void(*ErrorFun)(Matrix&,const Matrix&,const Matrix&);


void sigmoid(Matrix& mat,Matrix& result)
{
	size_t rows = mat.rows;
	size_t cols = mat.cols;
	for(size_t i=0;i<rows;i++)
	{
		for(size_t j=0;j<cols;j++)
		  result.matrix[i][j] = 1 / (1 + std::pow(EULER_NUMBER,-mat.matrix[i][j]));
	}
}
void sigmoid_derivative(Matrix& mat,Matrix& result)
{
	size_t rows = mat.rows;
	size_t cols = mat.cols;
	for(size_t i=0;i<rows;i++)
	{
		for(size_t j=0;j<cols;j++)
		  result.matrix[i][j] = mat.matrix[i][j]*(1.0 - mat.matrix[i][j]);
	}
}
void MSE_derivative(Matrix& result,const Matrix& aL,const Matrix& y)
{
    for(size_t i=0;i<aL.rows;i++)
	{
		for(size_t j=0;j<aL.cols;j++)
		{
			result.matrix[i][j] = (aL.matrix[i][j] - y.matrix[i][j]) * 2;
		}
	}
}
double MSE(const Matrix& aL,const Matrix& y)
{
	double ans = 0;
    for(size_t i=0;i<aL.rows;i++)
	{
		for(size_t j=0;j<aL.cols;j++)
		{
			ans +=  pow((aL.matrix[i][j] - y.matrix[i][j]) , 2);
		}
	}
	return ans;
}

class Layer
{
public:
	Matrix weights;
	Matrix bias;
	Callback fn;
	Callback fnd;
	ErrorFun errfnd;
    Matrix aL;
    Matrix dels; //deltas
	Matrix aL_dup; // auxiliary space to perform some operations on aL
    Matrix dout_dup;
    Matrix dw_transpose;
    Layer(size_t num_neurons,size_t num_inputs,Callback activation_fn,Callback fnd,ErrorFun errfn) : weights(num_inputs,num_neurons),
	   bias(1,num_neurons),aL(1,num_neurons),dels(1,1),aL_dup(1,1),dout_dup(1,1),dw_transpose(1,1)
	{
		//assign random weights
        double lower_bound = -1.0;
   		double upper_bound = 1.0;
   		std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
        std::random_device rd;
        std::default_random_engine re(rd()); 

        for(size_t i=0;i<num_inputs;i++)
		{
			for(size_t j=0;j<num_neurons;j++)
				weights[i][j] = unif(re);
		}
		for(size_t i=0;i<num_neurons;i++)
			bias[0][i] = unif(re);

        fn = (Callback)activation_fn;
		this->fnd = (Callback)fnd;
        errfnd = (ErrorFun)errfn;
	}
    Layer() : weights(1,1), bias(1,1),aL(1,1),dels(1,1),aL_dup(1,1),dout_dup(1,1),dw_transpose(1,1)
    {

    }
    const Matrix& forward(const Matrix& inputs)
	{
		aL.resize(inputs.rows,weights.cols);
		inputs.matmul(weights,aL);
		aL.addrow(bias);
		if(fn)
			fn(aL,aL); //activate output
		return aL;
	}
    // deltas at output layer
    const Matrix& deltas_output(const Matrix& y)
	{
        dels.resize(y.rows,weights.cols);
        aL_dup.resize(aL.rows,aL.cols);
		if(!fnd)
			errfnd(dels,aL,y);
		else 
		{
			fnd(aL,aL_dup);
			errfnd(dels,aL,y);
			aL_dup.mul(dels);
			return aL_dup;
		}
		return dels;
	}  
    Matrix& deltas(const Matrix& dout,const Matrix& dw) // dw are weights of next layer
	{
		dout_dup.resize(dout.rows,dw.rows);
        dw_transpose.resize(dw.cols,dw.rows);
        aL_dup.resize(aL.rows,aL.cols);

        dw.transpose(dw_transpose);
		dout.matmul(dw_transpose,dout_dup);
        
		if(fnd)
		{

			fnd(aL,aL_dup);
			dout_dup.mul(aL_dup);
		}
		return dout_dup;
	}
    void backpropagate(const Matrix& a,const Matrix& del,size_t idx,double lr = 0.1)
	{
        // stochastic gradient descent
		Matrix inputs(1,a.cols);
		Matrix deltas(1,del.cols);

		inputs.setrow(0,a.matrix[idx]);
        deltas.setrow(0,del.matrix[idx]);
		inputs = inputs.transpose();
		inputs = inputs.repeat_cols(deltas.cols);
		inputs.mulrow(deltas);
		//inputs is now delta weight
		Matrix& delta_weight = inputs;
		Matrix d1 = delta_weight * lr;
		Matrix d2 = deltas * lr;
		weights.sub(d1);
		bias.sub(d2);
	}
    void backpropagate_all(const Matrix& a,const Matrix& del,double lr = 0.1)
	{
        // shahryar's gradient descent
        size_t all = a.rows;
        for(size_t idx=0;idx<all;idx++)
        {
		    Matrix inputs(1,a.cols);
		    Matrix deltas(1,del.cols);
		    inputs.setrow(0,a.matrix[idx]);
            deltas.setrow(0,del.matrix[idx]);
		    inputs = inputs.transpose();
		    inputs = inputs.repeat_cols(deltas.cols);
		    inputs.mulrow(deltas);
		    //inputs is now delta weight
		    Matrix& delta_weight = inputs;
		    Matrix d1 = delta_weight * lr;
		    Matrix d2 = deltas * lr;
		    weights.sub(d1);
		    bias.sub(d2);
        }
	}
    
};
class Model
{
private:
    std::vector<Layer> layers;
public:
    void add_layer(const Layer& l)
    {
        layers.push_back(l);
    }
    void fit(const Matrix& inputs,const Matrix& y,size_t epochs,double lr = 0.1)
    {
        double loss;
        vector<const Matrix*> dels;
	    for(size_t i=1;i<=epochs;i++)
	    {
            for(size_t i=0;i<layers.size();i++)
            {
                if(i > 0)
                    layers[i].forward(layers[i-1].aL);
                else
                    layers[i].forward(inputs);
            }

		    
            loss = MSE(layers.back().aL,y);
            //backpropagate
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
                    layers[curr].backpropagate_all(inputs,*last_delta,lr);
                else
                    layers[curr].backpropagate_all(layers[curr-1].aL,*last_delta,lr);
            }
	    }
        cout << "Loss: "<<loss<<endl;
        cout << layers.back().aL;
    }
};
int main()
{
	//Training XOR network
	Matrix inputs(4,2);
	inputs[0][0] = 0;
	inputs[0][1] = 0;
	
	inputs[1][0] = 0;
	inputs[1][1] = 1;
	
	inputs[2][0] = 1;
	inputs[2][1] = 0;
	
	inputs[3][0] = 1;
	inputs[3][1] = 1;
	
	Matrix y(4,1);
	y[0][0] = 0;
	y[1][0] = 1;
	y[2][0] = 1;
	y[3][0] = 0;

    Model model;
	model.add_layer(Layer(2,2,&sigmoid,&sigmoid_derivative,&MSE_derivative));
	model.add_layer(Layer(1,2,sigmoid,sigmoid_derivative,MSE_derivative));

    size_t epochs = 10000;

    model.fit(inputs,y,epochs,0.1);
}