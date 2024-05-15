#include <cstdio>
#include <iostream>
#include <cmath>
#include <string.h>
#include <random>
#include <chrono>

#define EULER_NUMBER 2.718281828459045
using namespace std;

class Matrix
{
public:
    double** matrix = nullptr;
    size_t rows = 0;
    size_t cols = 0;

    Matrix(size_t r,size_t c)
    {
        rows = r;
        cols = c;
        matrix = new double*[rows];
        //no initialization to save performance
        //matrices in neural network are assigned random values anyway
        for(size_t i=0;i<rows;i++)
          matrix[i] = new double[cols];
    }
	Matrix(size_t r,size_t c,double val)
	{
		rows = r;
        cols = c;
        matrix = new double*[rows];
		for(size_t i=0;i<rows;i++)
        {
          	matrix[i] = new double[cols];
          	for(size_t j=0;j<cols;j++)
		    	matrix[i][j] = val;
        }
	}
    Matrix(const Matrix& obj)
    {
        rows = obj.rows;
        cols = obj.cols;
        matrix = new double*[rows];
        for(size_t i=0;i<rows;i++)
        {
          matrix[i] = new double[cols];
          memcpy(matrix[i],obj.matrix[i],sizeof(double)*cols);
        }
    }
    Matrix& operator=(const Matrix& obj)
    {
        if(&obj == this) return *this;
        for(size_t i=0;i<rows;i++)
            delete[] matrix[i];
        delete[] matrix;
        rows = obj.rows;
        cols = obj.cols;
        matrix = new double*[rows];
        for(size_t i=0;i<rows;i++)
        {
            matrix[i] = new double[cols];
            memcpy(matrix[i],obj.matrix[i],sizeof(double)*cols);
        }
        return *this;
    }
    double*& operator[](size_t idx)
    {
        return matrix[idx];
    }
    //Inplace operations
    void sub(const Matrix& rhs)
    {
        if(rows != rhs.rows || cols!=rhs.cols)
            return;
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<cols;j++)
            {
                matrix[i][j] -= rhs.matrix[i][j];
            }
        }
    }
    void add(const Matrix& rhs)
    {
        if(rows != rhs.rows || cols!=rhs.cols)
            return;
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<cols;j++)
            {
                matrix[i][j] += rhs.matrix[i][j];
            }
        }
    }
    void mul(const Matrix& rhs)
    {
        if(rows != rhs.rows || cols!=rhs.cols)
            return;
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<cols;j++)
            {
                matrix[i][j] *= rhs.matrix[i][j];
            }
        }
    }
    void div(const Matrix& rhs)
    {
        if(rows != rhs.rows || cols!=rhs.cols)
            return;
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<cols;j++)
            {
                matrix[i][j] /= rhs.matrix[i][j]; // divide by zero? user's responsibility, performance is first priority
            }
        }
    }
    //
    void addrow(const Matrix& rhs)
    {
        if(rhs.rows != 1 || cols!=rhs.cols)
            return;
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<cols;j++)
            {
                matrix[i][j] += rhs.matrix[0][j];
            }
        }
    }
    void mulrow(const Matrix& rhs)
    {
        if(rhs.rows != 1 || cols!=rhs.cols)
            return;
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<cols;j++)
            {
                matrix[i][j] *= rhs.matrix[0][j];
            }
        }
    }
    //
    void negate()
    {
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<cols;j++)
            {
                matrix[i][j] *= -1;
            }
        }
    }
    void add(double val)
    {
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<cols;j++)
            {
                matrix[i][j] += val;
            }
        }
    }
    void sub(double val)
    {
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<cols;j++)
            {
                matrix[i][j] -= val;
            }
        }
    }
	void mul(double val)
    {
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<cols;j++)
            {
                matrix[i][j] *= val;
            }
        }
    }
    void lsub(double val)
    {
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<cols;j++)
            {
                matrix[i][j] = val - matrix[i][j];
            }
        }
    }
    void exp()
    {
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<cols;j++)
            {
                matrix[i][j] = std::pow(EULER_NUMBER,matrix[i][j]);
            }
        }
    }
	void inverse()
    {
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<cols;j++)
            {
                matrix[i][j] = 1.0/matrix[i][j];
            }
        }
    }
	void square()
    {
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<cols;j++)
            {
                matrix[i][j] *= matrix[i][j];
            }
        }
    }
	double sum()
	{
		double s = 0;
		for(size_t i=0;i<rows;i++)
		{
			for(size_t j=0;j<cols;j++)
			  s += matrix[i][j];
		}
		return s;
	}
    //
    void matmul(const Matrix& rhs,Matrix& result)const
    {
        if(cols != rhs.rows || result.rows!=rows || result.cols != rhs.cols) //multiplication not possible
        {
            fprintf(stderr,"WARNING: Matrix multiplication not possible for orders (%zu,%zu) and (%zu,%zu)\n",rows,cols,rhs.rows,rhs.cols);
            return;
        }
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<rhs.cols;j++)
            {
                double sum = 0;
                for(size_t k=0;k<cols;k++)
                    sum += matrix[i][k] * rhs.matrix[k][j];
                result.matrix[i][j] = sum;
            }
        }
    }
	//Non inplace operations or methods that allocate memory
	Matrix repeat_cols(size_t n)
    {
		Matrix res(rows,cols*n);
		for(size_t i = 0;i<rows;i++)
		{
            size_t m = 0;
			for(size_t j=1;j<=n;j++)
			{
				for(size_t k=0;k<cols;k++)
				{
					res.matrix[i][m++] = matrix[i][k];
				}
			}
		}
		return res;
	}
	Matrix transpose()
	{
		Matrix res(cols,rows);
		for(size_t i=0;i<rows;i++)
		{
			for(size_t j=0;j<cols;j++)
			  res.matrix[j][i] = matrix[i][j];
		}
		return res;
	}
	void transpose(Matrix& res) const
	{
        if(res.rows != cols || res.cols != rows)
		  return;
		for(size_t i=0;i<rows;i++)
		{
			for(size_t j=0;j<cols;j++)
			  res.matrix[j][i] = matrix[i][j];
		}
	}
    Matrix row_argmax() // returns column idx of max element in each row
	{
    	Matrix res(rows,1);
		for(size_t i=0;i<rows;i++)
		{
			double max_val = 0;
			double max_idx = 0;
			for(size_t j=0;j<cols;j++)
			{
				if(matrix[i][j] > max_val || j == 0)
				{
					max_val = matrix[i][j];
					max_idx = j;
				}
			}
			res.matrix[i][0] = max_idx;
		}
		return res;
	}
	void row_argmax(Matrix& res) // returns column idx of max element in each row
	{
		if(res.rows != rows || res.cols != 1)
		  return;
		for(size_t i=0;i<rows;i++)
		{
			double max_val = 0;
			double max_idx = 0;
			for(size_t j=0;j<cols;j++)
			{
				if(matrix[i][j] > max_val || j == 0)
				{
					max_val = matrix[i][j];
					max_idx = j;
				}
			}
			res.matrix[i][0] = max_idx;
		}
	}
	Matrix copy() const
	{
		return Matrix(*this);
	}
	size_t num_rows() const
	{
		return rows;
	}
    size_t num_cols() const
	{
		return cols;
	}
	void setrow(size_t idx,double* row)
	{
		if(idx >= rows)
		  return;
		memcpy(matrix[idx],row, cols*sizeof(double));
	}
	Matrix operator*(double val)
    {
		Matrix result(rows,cols);
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<cols;j++)
            {
                result[i][j] = matrix[i][j]*val;
            }
        }
		return result;
    }
	void resize(size_t r,size_t c)
	{
		if(rows == r && cols == c)
		  return;
        for(size_t i=0;i<rows;i++)
            delete[] matrix[i];
        delete[] matrix;
        rows = r;
        cols = c;
        matrix = new double*[rows];
        for(size_t i=0;i<rows;i++)
        {
            matrix[i] = new double[cols];
        }
	}
    friend std::ostream& operator<<(std::ostream& out,const Matrix& );
    ~Matrix()
    {
        for(size_t i=0;i<rows;i++)
          delete[] matrix[i];
        delete[] matrix;
    }
};
std::ostream& operator<<(std::ostream& out,const Matrix& matrix)
{
    for(size_t i=0;i<matrix.rows;i++)
    {
        for(size_t j=0;j<matrix.cols;j++)
          out << matrix.matrix[i][j]<< " ";
        out << std::endl;        
    }
    return out;
}

typedef void(*Callback)(Matrix&,Matrix&);
typedef void(*ErrorFun)(Matrix&,const Matrix&,const Matrix&);


void sigmoid(Matrix& mat,Matrix& result)
{
	size_t rows = mat.num_rows();
	size_t cols = mat.num_cols();
	for(size_t i=0;i<rows;i++)
	{
		for(size_t j=0;j<cols;j++)
		  result.matrix[i][j] = 1 / (1 + std::pow(EULER_NUMBER,-mat.matrix[i][j]));
	}
}
void sigmoid_derivative(Matrix& mat,Matrix& result)
{
	size_t rows = mat.num_rows();
	size_t cols = mat.num_cols();
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
    Matrix ld;
	Matrix aL_dup; // auxiliary space to perform some operations on aL
    Matrix dout_dup;
    Matrix dw_transpose;
    Layer(size_t num_neurons,size_t num_inputs,Callback activation_fn,Callback fnd,ErrorFun errfn) : weights(num_inputs,num_neurons),
	   bias(1,num_neurons),aL(1,num_neurons),ld(1,1),aL_dup(1,1),dout_dup(1,1),dw_transpose(1,1)
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
        ld.resize(y.rows,y.cols);
        aL_dup.resize(aL.rows,aL.cols);
		if(!fnd)
			errfnd(ld,aL,y);
		else 
		{
			fnd(aL,aL_dup);
			errfnd(ld,aL,y);
			aL_dup.mul(ld);
			return aL_dup;
		}
		return ld;
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


	Layer hidden(2,2,&sigmoid,&sigmoid_derivative,&MSE_derivative);
	Layer output(1,2,sigmoid,sigmoid_derivative,MSE_derivative);
    size_t epochs = 100000;
    double loss;
	for(size_t i=1;i<=epochs;i++)
	{
		const Matrix& a1 = hidden.forward(inputs);
        const Matrix& a2 = output.forward(a1);
        loss = MSE(a2,y);
    	const Matrix& dout = output.deltas_output(y);
    	const Matrix& dh = hidden.deltas(dout,output.weights);
		size_t idx = rand()%4; // pick a random idx
    	output.backpropagate(a1,dout,idx,0.1);
    	hidden.backpropagate(inputs,dh,idx,0.1);
	}
    cout << "Loss: "<<loss<<endl;
    cout << output.aL;
}