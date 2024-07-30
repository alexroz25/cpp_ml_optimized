#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cassert>
#include <iostream>
#include <cmath>

#define LEAKY_RELU_HYPERPARAM 0.1

template <typename T> class Matrix {
private:
	int _rows;
	int _cols;
	std::vector<T> _values;

public:
	Matrix() : _rows(0), _cols(0), _values({}) {}

	Matrix(int rows, int cols) : _rows(rows), _cols(cols), _values(std::vector<T>(rows* cols)) { assert(rows * cols == _values.size()); }

	Matrix(int rows, int cols, std::vector<T> values) : _rows(rows), _cols(cols), _values(values) { assert(rows * cols == values.size()); }

	// read-only getters
	int rows() { return _rows; }
	int cols() { return _cols; }
	const std::vector<T>& values() const { return _values; }

	T& at(int r, int c) { return _values[r * _cols + c]; }

    Matrix<T> multiply(const float n) const;

	Matrix<T> multiply(const Matrix<T>& other) const;

	Matrix<T> multiply_transpose(const Matrix<T>& otherT) const; // slow, avoid if possible

	Matrix<T> transpose_multiply(const Matrix<T>& other) const;

    Matrix<T> square() const;

    Matrix<T> add(const Matrix<T>& other) const;

    Matrix<T> subtract(const Matrix<T>& other) const;

    Matrix<T> add_vector(const std::vector<T>& other) const;

    std::vector<T> collapse() const;

    Matrix<T> leaky_ReLU() const;

    Matrix<T> leaky_ReLU_derivative(const Matrix<T>& ref) const;

    Matrix<T> softmax() const;

	void print();

	void print_dimensions();
};

template <typename T> Matrix<T> Matrix<T>::multiply(const float n) const {
    Matrix<T> out = *this;
    int tot = _rows * _cols;
    #pragma omp parallel for
    for (int i = 0; i < tot; ++i) out._values[i] *= n;
    return out;
}

template <typename T> Matrix<T> Matrix<T>::multiply(const Matrix<T>& other) const {
	assert(this->_cols == other._rows);
	Matrix<T> out(this->_rows, other._cols);
	
    int thisRows = _rows, otherRows = other._rows, otherCols = other._cols;

	#pragma omp parallel for
	for (int i = 0; i < thisRows; ++i) {
		for (int k = 0; k < otherRows; ++k) {
			for (int j = 0; j < otherCols; ++j) {
				out._values[i * otherCols + j] += _values[i * otherRows + k] * other._values[k * otherCols + j];
			}
		}
	}

	return out;
}

template <typename T> Matrix<T> Matrix<T>::multiply_transpose(const Matrix<T>& otherT) const {
	assert(this->_cols == otherT._cols);
	Matrix<T> out(this->_rows, otherT._rows);

    int thisRows = _rows, otherTRows = otherT._rows, otherTCols = otherT._cols;

	#pragma omp parallel for
    for (int i = 0; i < thisRows; ++i) {
        for (int j = 0; j < otherTRows; ++j) {
            for (int k = 0; k < otherTCols; ++k) {
				out._values[i * otherTRows + j] += _values[i * otherTCols + k] * otherT._values[j * otherTCols + k];
			}
		}
	}

	return out;
}

template <typename T> Matrix<T> Matrix<T>::transpose_multiply(const Matrix<T>& other) const {
	assert(this->_rows == other._rows);
	Matrix<T> out(this->_cols, other._cols);

    int thisCols = _cols, otherCols = other._cols, otherRows = other._rows;

	#pragma omp parallel for
    for (int i = 0; i < thisCols; ++i) {
        for (int k = 0; k < otherRows; ++k) {
            for (int j = 0; j < otherCols; ++j) {
				out._values[i * otherCols + j] += _values[k * thisCols + i] * other._values[k * otherCols + j];
			}
		}
	}

	return out;
}

template <typename T> Matrix<T> Matrix<T>::square() const {
    Matrix<T> out = *this;
    int n = _rows * _cols;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) out._values[i] *= _values[i];
    return out;
}

template <typename T> Matrix<T> Matrix<T>::add(const Matrix<T>& other) const {
    assert(_rows == other._rows && _cols == other._cols);
    Matrix out = *this;
    int n = _rows * _cols;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) out._values[i] += other._values[i];
    return out;
}

template <typename T> Matrix<T> Matrix<T>::subtract(const Matrix<T>& other) const {
    assert(_rows == other._rows && _cols == other._cols);
    Matrix out = *this;
    int n = _rows * _cols;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) out._values[i] -= other._values[i];
    return out;
}

template <typename T> Matrix<T> Matrix<T>::add_vector(const std::vector<T>& other) const {
    assert(other.size() == _rows);
    Matrix<T> out = *this;
    int rows = _rows, cols = _cols;
    
    #pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        int rval = other[r];
        for (int c = 0; c < cols; ++c) {
            out._values[r * _cols + c] += rval;
        }
    }
    return out;
}

template <typename T> std::vector<T> Matrix<T>::collapse() const {
    std::vector<T> result(_rows);
    int rows = _rows, cols = _cols;
    #pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            result[r] += _values[r * cols + c];
        }
    }
    return result;
}

template <typename T> Matrix<T> Matrix<T>::leaky_ReLU() const {
    Matrix<T> out = *this;
    int n = _rows * _cols;

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        if (out._values[i] < 0) out._values[i] *= LEAKY_RELU_HYPERPARAM;
    }
    
    return out;
}

template <typename T> Matrix<T> Matrix<T>::leaky_ReLU_derivative(const Matrix<T>& ref) const {
    assert(_rows == ref._rows && _cols == ref._cols);
    Matrix<T> out = *this;
    int n = _rows * _cols;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        if (ref._values[i] < 0) out._values[i] *= LEAKY_RELU_HYPERPARAM;
    }
    return out;
}

template <typename T> Matrix<T> Matrix<T>::softmax() const {
        Matrix<float> out = *this;
        int rows = _rows;
        int cols = _cols;
        
        for (int c = 0; c < cols; ++c) {
            float denom = 0;
            for (int r = 0; r < rows; ++r) denom += exp(out.at(r, c));
            for (int r = 0; r < rows; ++r) out.at(r, c) = exp(out.at(r, c)) / denom;
        }
        
        return out;
    }

template <typename T> void Matrix<T>::print() {
	std::cout << _rows << " x " << _cols << ":\n";

	for (int r = 0; r < _rows; ++r) {
		for (int c = 0; c < _cols; ++c) {
			std::cout << at(r, c) << '\t';
		}
		std::cout << '\n';
	}

	std::cout << std::endl;
}

template <typename T> void Matrix<T>::print_dimensions() {
	std::cout << _rows << " x " << _cols << std::endl;
}

#endif