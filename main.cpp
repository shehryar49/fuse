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
	//Auxiliary space to avoid allocating memory during training
    Matrix aL;
    Matrix dels; //deltas
	Matrix aL_dup; // auxiliary space to perform some operations on aL
    Matrix dout_dup;
    Matrix dw_transpose;
	Matrix deltas_chosen;
	Matrix inputsT_repeated;

    Layer(size_t num_neurons,size_t num_inputs,Callback activation_fn,Callback fnd,ErrorFun errfn) : weights(num_inputs,num_neurons),
	   bias(1,num_neurons),aL(1,num_neurons),dels(1,1),aL_dup(1,1),dout_dup(1,1),dw_transpose(1,1),
	   deltas_chosen(1, num_neurons),inputsT_repeated(1,1)
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
    Layer() : weights(1,1), bias(1,1),aL(1,1),dels(1,1),aL_dup(1,1),dout_dup(1,1),dw_transpose(1,1),
	  deltas_chosen(1,1),
	  inputsT_repeated(1,1)
    {

    }
    const Matrix& forward(const Matrix& inputs)
	{
		inputs.matmul(weights,aL);
		aL.addrow(bias);
		if(fn)
			fn(aL,aL); //activate output
		return aL;
	}
	void adapt_aux(size_t num_inputs,size_t next_neurons) // adapt auxiliary variables to sizes
	{
		//Number of inputs here is not the number of inputs from previous layer
		//It is the total samples we are training on

		//weights.cols tells the number of neurons
		//it is also the number of inputs next layer will receive
		aL.resize(num_inputs,weights.cols);
		aL_dup.resize(num_inputs,weights.cols);
		dels.resize(num_inputs,weights.cols);
		dout_dup.resize(num_inputs,next_neurons);
		dw_transpose.resize(next_neurons,weights.cols);
        //
		inputsT_repeated.resize(weights.rows,weights.cols);

	}
    // deltas at output layer
    const Matrix& deltas_output(const Matrix& y)
	{
        //dels.resize(y.rows,weights.cols);
        //aL_dup.resize(aL.rows,aL.cols);
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
        deltas_chosen.setrow(0,del.matrix[idx]);
		inputsT_repeated.setcols(a.matrix[idx]);
		inputsT_repeated.mulrow(deltas_chosen);
		inputsT_repeated.mul(lr);
		deltas_chosen.mul(lr);
		weights.sub(inputsT_repeated);
		bias.sub(deltas_chosen);
	}
    void backpropagate_all(const Matrix& a,const Matrix& del,double lr = 0.1)
	{
        // shahryar's gradient descent
        size_t all = a.rows;
        for(size_t idx=0;idx<all;idx++)
        {
			deltas_chosen.setrow(0,del.matrix[idx]);
			inputsT_repeated.setcols(a.matrix[idx]);
			inputsT_repeated.mulrow(deltas_chosen);
			inputsT_repeated.mul(lr);
			deltas_chosen.mul(lr);
			weights.sub(inputsT_repeated);
			bias.sub(deltas_chosen);
        }
	}
    
};
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
            loss = MSE(layers.back().aL,y);
	        cout << "Loss: "<<loss<<"  Accuracy:"<<accuracy(layers.back().aL,y) <<endl;
            //backpropagate
			size_t idx = rand()%150;
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
		loss = MSE(layers.back().aL,y);
		cout << "Loss: "<<loss<<"  Accuracy:"<<accuracy(layers.back().aL,y) <<endl;
    }
};
void XOR_demo()
{
	//Training XOR
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
	/*model.add_layer(Layer(2,2,&sigmoid,&sigmoid_derivative,&MSE_derivative));
	model.add_layer(Layer(1,2,sigmoid,sigmoid_derivative,MSE_derivative));*/
	
	model.add_layer(Layer(128,2,sigmoid,sigmoid_derivative,MSE_derivative));
	model.add_layer(Layer(64,128,sigmoid,sigmoid_derivative,MSE_derivative));
	model.add_layer(Layer(1,64,sigmoid,sigmoid_derivative,MSE_derivative));
	
    size_t epochs = 1000;
    model.fit(inputs,y,epochs,0.1);	
}
void IRIS_demo()
{
	double inputs_arr[150][4] = {{5.1, 3.5, 1.4, 0.2}, {4.9, 3.0, 1.4, 0.2}, {4.7, 3.2, 1.3, 0.2}, {4.6, 3.1, 1.5, 0.2}, {5.0, 3.6, 1.4, 0.2}, {5.4, 3.9, 1.7, 0.4}, {4.6, 3.4, 1.4, 0.3}, {5.0, 3.4, 1.5, 0.2}, {4.4, 2.9, 1.4, 0.2}, {4.9, 3.1, 1.5, 0.1}, {5.4, 3.7, 1.5, 0.2}, {4.8, 3.4, 1.6, 0.2}, {4.8, 3.0, 1.4, 0.1}, {4.3, 3.0, 1.1, 0.1}, {5.8, 4.0, 1.2, 0.2}, {5.7, 4.4, 1.5, 0.4}, {5.4, 3.9, 1.3, 0.4}, {5.1, 3.5, 1.4, 0.3}, {5.7, 3.8, 1.7, 0.3}, {5.1, 3.8, 1.5, 0.3}, {5.4, 3.4, 1.7, 0.2}, {5.1, 3.7, 1.5, 0.4}, {4.6, 3.6, 1.0, 0.2}, {5.1, 3.3, 1.7, 0.5}, {4.8, 3.4, 1.9, 0.2}, {5.0, 3.0, 1.6, 0.2}, {5.0, 3.4, 1.6, 0.4}, {5.2, 3.5, 1.5, 0.2}, {5.2, 3.4, 1.4, 0.2}, {4.7, 3.2, 1.6, 0.2}, {4.8, 3.1, 1.6, 0.2}, {5.4, 3.4, 1.5, 0.4}, {5.2, 4.1, 1.5, 0.1}, {5.5, 4.2, 1.4, 0.2}, {4.9, 3.1, 1.5, 0.1}, {5.0, 3.2, 1.2, 0.2}, {5.5, 3.5, 1.3, 0.2}, {4.9, 3.1, 1.5, 0.1}, {4.4, 3.0, 1.3, 0.2}, {5.1, 3.4, 1.5, 0.2}, {5.0, 3.5, 1.3, 0.3}, {4.5, 2.3, 1.3, 0.3}, {4.4, 3.2, 1.3, 0.2}, {5.0, 3.5, 1.6, 0.6}, {5.1, 3.8, 1.9, 0.4}, {4.8, 3.0, 1.4, 0.3}, {5.1, 3.8, 1.6, 0.2}, {4.6, 3.2, 1.4, 0.2}, {5.3, 3.7, 1.5, 0.2}, {5.0, 3.3, 1.4, 0.2}, {7.0, 3.2, 4.7, 1.4}, {6.4, 3.2, 4.5, 1.5}, {6.9, 3.1, 4.9, 1.5}, {5.5, 2.3, 4.0, 1.3}, {6.5, 2.8, 4.6, 1.5}, {5.7, 2.8, 4.5, 1.3}, {6.3, 3.3, 4.7, 1.6}, {4.9, 2.4, 3.3, 1.0}, {6.6, 2.9, 4.6, 1.3}, {5.2, 2.7, 3.9, 1.4}, {5.0, 2.0, 3.5, 1.0}, {5.9, 3.0, 4.2, 1.5}, {6.0, 2.2, 4.0, 1.0}, {6.1, 2.9, 4.7, 1.4}, {5.6, 2.9, 3.6, 1.3}, {6.7, 3.1, 4.4, 1.4}, {5.6, 3.0, 4.5, 1.5}, {5.8, 2.7, 4.1, 1.0}, {6.2, 2.2, 4.5, 1.5}, {5.6, 2.5, 3.9, 1.1}, {5.9, 3.2, 4.8, 1.8}, {6.1, 2.8, 4.0, 1.3}, {6.3, 2.5, 4.9, 1.5}, {6.1, 2.8, 4.7, 1.2}, {6.4, 2.9, 4.3, 1.3}, {6.6, 3.0, 4.4, 1.4}, {6.8, 2.8, 4.8, 1.4}, {6.7, 3.0, 5.0, 1.7}, {6.0, 2.9, 4.5, 1.5}, {5.7, 2.6, 3.5, 1.0}, {5.5, 2.4, 3.8, 1.1}, {5.5, 2.4, 3.7, 1.0}, {5.8, 2.7, 3.9, 1.2}, {6.0, 2.7, 5.1, 1.6}, {5.4, 3.0, 4.5, 1.5}, {6.0, 3.4, 4.5, 1.6}, {6.7, 3.1, 4.7, 1.5}, {6.3, 2.3, 4.4, 1.3}, {5.6, 3.0, 4.1, 1.3}, {5.5, 2.5, 4.0, 1.3}, {5.5, 2.6, 4.4, 1.2}, {6.1, 3.0, 4.6, 1.4}, {5.8, 2.6, 4.0, 1.2}, {5.0, 2.3, 3.3, 1.0}, {5.6, 2.7, 4.2, 1.3}, {5.7, 3.0, 4.2, 1.2}, {5.7, 2.9, 4.2, 1.3}, {6.2, 2.9, 4.3, 1.3}, {5.1, 2.5, 3.0, 1.1}, {5.7, 2.8, 4.1, 1.3}, {6.3, 3.3, 6.0, 2.5}, {5.8, 2.7, 5.1, 1.9}, {7.1, 3.0, 5.9, 2.1}, {6.3, 2.9, 5.6, 1.8}, {6.5, 3.0, 5.8, 2.2}, {7.6, 3.0, 6.6, 2.1}, {4.9, 2.5, 4.5, 1.7}, {7.3, 2.9, 6.3, 1.8}, {6.7, 2.5, 5.8, 1.8}, {7.2, 3.6, 6.1, 2.5}, {6.5, 3.2, 5.1, 2.0}, {6.4, 2.7, 5.3, 1.9}, {6.8, 3.0, 5.5, 2.1}, {5.7, 2.5, 5.0, 2.0}, {5.8, 2.8, 5.1, 2.4}, {6.4, 3.2, 5.3, 2.3}, {6.5, 3.0, 5.5, 1.8}, {7.7, 3.8, 6.7, 2.2}, {7.7, 2.6, 6.9, 2.3}, {6.0, 2.2, 5.0, 1.5}, {6.9, 3.2, 5.7, 2.3}, {5.6, 2.8, 4.9, 2.0}, {7.7, 2.8, 6.7, 2.0}, {6.3, 2.7, 4.9, 1.8}, {6.7, 3.3, 5.7, 2.1}, {7.2, 3.2, 6.0, 1.8}, {6.2, 2.8, 4.8, 1.8}, {6.1, 3.0, 4.9, 1.8}, {6.4, 2.8, 5.6, 2.1}, {7.2, 3.0, 5.8, 1.6}, {7.4, 2.8, 6.1, 1.9}, {7.9, 3.8, 6.4, 2.0}, {6.4, 2.8, 5.6, 2.2}, {6.3, 2.8, 5.1, 1.5}, {6.1, 2.6, 5.6, 1.4}, {7.7, 3.0, 6.1, 2.3}, {6.3, 3.4, 5.6, 2.4}, {6.4, 3.1, 5.5, 1.8}, {6.0, 3.0, 4.8, 1.8}, {6.9, 3.1, 5.4, 2.1}, {6.7, 3.1, 5.6, 2.4}, {6.9, 3.1, 5.1, 2.3}, {5.8, 2.7, 5.1, 1.9}, {6.8, 3.2, 5.9, 2.3}, {6.7, 3.3, 5.7, 2.5}, {6.7, 3.0, 5.2, 2.3}, {6.3, 2.5, 5.0, 1.9}, {6.5, 3.0, 5.2, 2.0}, {6.2, 3.4, 5.4, 2.3}, {5.9, 3.0, 5.1, 1.8}};
	double y_arr[150][3] = {{1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
	
	Matrix inputs(150,4);
	Matrix y(150,3);
    
	for(size_t i=0;i<150;i++)
	{
		memcpy(inputs.matrix[i],inputs_arr[i],4*sizeof(double));
		memcpy(y.matrix[i],y_arr[i],3*sizeof(double));
	}

	Model m;
	m.add_layer(Layer(2,4,sigmoid,sigmoid_derivative,MSE_derivative));
	m.add_layer(Layer(3,2,sigmoid,sigmoid_derivative,MSE_derivative));
	
	m.fit(inputs,y,10000,0.1);

}

int main()
{
	XOR_demo();
	//IRIS_demo();
}