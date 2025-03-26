#include "layer.h"



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

Layer::Layer(size_t num_neurons,size_t num_inputs,MatOp activation_fn,MatOp fnd,LossFunctionDeriv errfn) : weights(num_inputs,num_neurons),
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

    fn = activation_fn;
    this->fnd = fnd;
    errfnd = errfn;
}
Layer::Layer() : weights(1,1), bias(1,1),aL(1,1),dels(1,1),aL_dup(1,1),dout_dup(1,1),dw_transpose(1,1),
    deltas_chosen(1,1),
    inputsT_repeated(1,1)
{

}
const Matrix& Layer::forward(const Matrix& inputs)
{
    inputs.matmul(weights,aL);
    aL.addrow(bias);
    if(fn)
        fn(aL,aL); //activate output
    return aL;
}
void Layer::adapt_aux(size_t num_inputs,size_t next_neurons) // adapt auxiliary variables to sizes
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
const Matrix& Layer::deltas_output(const Matrix& y)
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
Matrix& Layer::deltas(const Matrix& dout,const Matrix& dw) // dw are weights of next layer
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
void Layer::backpropagate(const Matrix& a,const Matrix& del,size_t idx,double lr)
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
void Layer::backpropagate_all(const Matrix& a,const Matrix& del,double lr )
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
