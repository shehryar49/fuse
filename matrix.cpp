#include "matrix.h"
#include <cstdio>


Matrix::Matrix(size_t r,size_t c)
{
    rows = r;
    cols = c;
    matrix = new double*[rows];
    //no initialization to save performance
    //matrices in neural network are assigned random values anyway
    for(size_t i=0;i<rows;i++)
        matrix[i] = new double[cols];
}
Matrix::Matrix(size_t r,size_t c,double val)
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
Matrix::Matrix(const Matrix& obj)
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
Matrix& Matrix::operator=(const Matrix& obj)
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
double*& Matrix::operator[](size_t idx)
{
    return matrix[idx];
}
//Inplace operations
void Matrix::sub(const Matrix& rhs)
{
    if(rows != rhs.rows || cols!=rhs.cols)
    {
        fprintf(stderr,"WARNING: Incompatible dimensions");
        return;
    }
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            matrix[i][j] -= rhs.matrix[i][j];
        }
    }
}
void Matrix::add(const Matrix& rhs)
{
    if(rows != rhs.rows || cols!=rhs.cols)
    {
        fprintf(stderr,"WARNING: Incompatible dimensions");
        return;
    }
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            matrix[i][j] += rhs.matrix[i][j];
        }
    }
}
void Matrix::mul(const Matrix& rhs)
{
    if(rows != rhs.rows || cols!=rhs.cols)
    {
        fprintf(stderr,"WARNING: Incompatible dimensions");
        return;
    }
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            matrix[i][j] *= rhs.matrix[i][j];
        }
    }
}
void Matrix::div(const Matrix& rhs)
{
    if(rows != rhs.rows || cols!=rhs.cols)
    {
        fprintf(stderr,"WARNING: Incompatible dimensions");
        return;
    }
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            matrix[i][j] /= rhs.matrix[i][j]; // divide by zero? user's responsibility, performance is first priority
        }
    }
}
//
void Matrix::addrow(const Matrix& rhs)
{
    if(rhs.rows != 1 || cols!=rhs.cols)
    {
        fprintf(stderr,"WARNING: Incompatible dimensions");
        return;
    }
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            matrix[i][j] += rhs.matrix[0][j];
        }
    }
}
void Matrix::mulrow(const Matrix& rhs)
{
    if(rhs.rows != 1 || cols!=rhs.cols)
    {
        fprintf(stderr,"WARNING: Incompatible dimensions");
        return;
    }
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            matrix[i][j] *= rhs.matrix[0][j];
        }
    }
}
//
void Matrix::negate()
{
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            matrix[i][j] *= -1;
        }
    }
}
void Matrix::add(double val)
{
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            matrix[i][j] += val;
        }
    }
}
void Matrix::sub(double val)
{
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            matrix[i][j] -= val;
        }
    }
}
void Matrix::mul(double val)
{
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            matrix[i][j] *= val;
        }
    }
}
void Matrix::mul(double val,Matrix& result)
{
    if(result.rows != rows || result.cols!=cols)
    {
        fprintf(stderr,"WARNING: Can't store result, mul(double,Matrix&) ");
        return;
    }
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            result.matrix[i][j] = matrix[i][j]*val;
        }
    }
}

void Matrix::lsub(double val)
{
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            matrix[i][j] = val - matrix[i][j];
        }
    }
}
void Matrix::exp()
{
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            matrix[i][j] = std::pow(EULER_NUMBER,matrix[i][j]);
        }
    }
}
void Matrix::inverse()
{
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            matrix[i][j] = 1.0/matrix[i][j];
        }
    }
}
void Matrix::square()
{
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            matrix[i][j] *= matrix[i][j];
        }
    }
}
double Matrix::sum()
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
void Matrix::matmul(const Matrix& rhs,Matrix& result)const
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
Matrix Matrix::repeat_cols(size_t n)
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
void Matrix::repeat_cols(size_t n,Matrix& res)
{
    if(res.rows != rows || res.cols != cols*n)
    {
        fprintf(stderr, "WARNING: Unable to store repeat_cols result\n");
        return;
    }
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
}
Matrix Matrix::transpose()
{
    Matrix res(cols,rows);
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
            res.matrix[j][i] = matrix[i][j];
    }
    return res;
}
void Matrix::transpose(Matrix& res) const
{
    if(res.rows != cols || res.cols != rows)
    {
        fprintf(stderr,"WARNING: Can't store transpose result.");
        return;
    }
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
            res.matrix[j][i] = matrix[i][j];
    }
}
Matrix Matrix::row_argmax() // returns column idx of max element in each row
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
void Matrix::row_argmax(Matrix& res) // returns column idx of max element in each row
{
    if(res.rows != rows || res.cols != 1)
    {
        fprintf(stderr,"WARNING: Can't store argmax result");
        return;
    }
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
void Matrix::setrow(size_t idx,double* row)
{
    if(idx >= rows)
        return;
    memcpy(matrix[idx],row, cols*sizeof(double));
}
void Matrix::setcol(size_t idx,double* col)
{
    if(idx >= cols)
      return;
    size_t j = 0;
    for(size_t i=0;i<rows;i++)
    {
        matrix[i][idx] = col[j++];
    }
}
void Matrix::setcols(double* col)
{
    size_t k = 0;
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            matrix[i][j] = col[k];
        }
        k++;
    }
}
Matrix Matrix::operator*(double val)
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
void Matrix::resize(size_t r,size_t c)
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
Matrix::~Matrix()
{
    for(size_t i=0;i<rows;i++)
        delete[] matrix[i];
    delete[] matrix;
}


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