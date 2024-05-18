#ifndef FUSE_MATRIX_H_
#define FUSE_MATRIX_H_
#include <iostream>
#include <string.h>
#include <cmath>

#define EULER_NUMBER 2.718281828459045
class Matrix
{
public:
    double** matrix = nullptr;
    size_t rows = 0;
    size_t cols = 0;
    Matrix(size_t r,size_t c);
	Matrix(size_t r,size_t c,double val);
    Matrix(const Matrix& obj);
    Matrix& operator=(const Matrix& obj);
    double*& operator[](size_t idx);
    //Inplace operations
    void sub(const Matrix& rhs);
    void add(const Matrix& rhs);
    void mul(const Matrix& rhs);
    void div(const Matrix& rhs);
    //
    void addrow(const Matrix& rhs);
    void mulrow(const Matrix& rhs);
    //
    void negate();
    void add(double val);
    void sub(double val);
	void mul(double val);
    void mul(double val,Matrix& res);
    void lsub(double val);
    void exp();
	void inverse();
	void square();
	double sum();
    //
    void matmul(const Matrix& rhs,Matrix& result)const;
	//Non inplace operations or methods that allocate memory
	Matrix repeat_cols(size_t n);
    void repeat_cols(size_t n,Matrix& res);
	Matrix transpose();
	void transpose(Matrix& res) const;
    Matrix row_argmax(); // returns column idx of max element in each row
	void row_argmax(Matrix& res); // returns column idx of max element in each row
	void setrow(size_t idx,double* row);
    void setcol(size_t idx,double* col);
    void setcols(double* col);
	Matrix operator*(double val);
	void resize(size_t r,size_t c);
    friend std::ostream& operator<<(std::ostream& out,const Matrix& );
    ~Matrix();
};

#endif