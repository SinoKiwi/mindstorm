#pragma once 
#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <iostream>
#include <vector>
#include <algorithm>

template <typename T>
class Tensor {
protected:
    std::vector<T> data_;
private:
    std::vector<size_t> shape_;
    std::vector<size_t> stride_;

public:
    Tensor(const std::vector<size_t>& shape) : shape_(shape) {
        size_t size = 1;
        for (size_t dim : shape_) {
            size *= dim;
        }
        data_.resize(size);
        calculateStride();
    }

    void calculateStride() {
        stride_.resize(shape_.size());
        stride_[stride_.size() - 1] = 1;
        for (int i = stride_.size() - 2; i >= 0; --i) {
            stride_[i] = stride_[i + 1] * shape_[i + 1];
        }
    }

    size_t getOffset(const std::vector<size_t>& indices) const {
        size_t offset = 0;
        for (int i = 0; i < indices.size(); ++i) {
            offset += indices[i] * stride_[i];
        }
        return offset;
    }

    T& operator[](const std::vector<size_t>& indices) {
        size_t offset = getOffset(indices);
        return data_[offset];
    }

    const T& operator[](const std::vector<size_t>& indices) const {
        size_t offset = getOffset(indices);
        return data_[offset];
    }

    std::vector<size_t> shape() const {
        return shape_;
    }

    void reshape(const std::vector<size_t>& new_shape) {
        size_t size = 1;
        for (size_t dim : new_shape) {
            size *= dim;
        }
        if (size != data_.size()) {
            throw std::runtime_error("Cannot reshape tensor to a different size.");
        }
        shape_ = new_shape;
        calculateStride();
    }

    Tensor<T> slice(const std::vector<size_t>& start, const std::vector<size_t>& end) const {
        std::vector<size_t> new_shape;
        for (int i = 0; i < shape_.size(); ++i) {
            new_shape.push_back(end[i] - start[i]);
        }
        Tensor<T> result(new_shape);

        std::vector<size_t> indices(shape_.size(), 0);
        for (int i = 0; i < result.data_.size(); ++i) {
            result.data_[i] = (*this)[indices];
            for (int j = shape_.size() - 1; j >= 0; --j) {
                if (indices[j] < end[j] - 1) {
                    indices[j]++;
                    break;
                } else {
                    indices[j] = start[j];
                }
            }
        }

        return result;
    }

    Tensor<T> transpose(const size_t dim1, const size_t dim2) const {
        if (dim1 >= shape_.size() || dim2 >= shape_.size()) {
            throw std::runtime_error("Invalid dimension indices for transpose.");
        }
        std::vector<size_t> transposed_shape = shape_;
        std::swap(transposed_shape[dim1], transposed_shape[dim2]);
        Tensor<T> transposed_tensor(transposed_shape);
        for (size_t i = 0; i < data_.size(); ++i) {
            std::vector<size_t> indices = getIndicesFromOffset(i);
            std::swap(indices[dim1], indices[dim2]);
            transposed_tensor[indices] = data_[i];
        }
        return transposed_tensor;
    }
private:
    std::vector<size_t> getIndicesFromOffset(size_t offset) const {
        std::vector<size_t> indices(shape_.size());
        for (int i = indices.size() - 1; i >= 0; --i) {
            indices[i] = offset % shape_[i];
            offset /= shape_[i];
        }
        return indices;
    }
};


template <typename T>
Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensors must have the same shape for addition.");
    }

    Tensor<T> result(a.shape());
    for (int i = 0; i < result.data_.size(); ++i) {
        result.data_[i] = a.data_[i] + b.data_[i];
    }

    return result;
}

template <typename T>
Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensors must have the same shape for subtraction.");
    }

    Tensor<T> result(a.shape());
    for (int i = 0; i < result.data_.size(); ++i) {
        result.data_[i] = a.data_[i] - b.data_[i];
    }

    return result;
}

template <typename T>
Tensor<T> operator*(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensors must have the same shape for element-wise multiplication.");
    }

    Tensor<T> result(a.shape());
    for (int i = 0; i < result.data_.size(); ++i) {
        result.data_[i] = a.data_[i] * b.data_[i];
    }

    return result;
}

template <typename T>
Tensor<T> operator/(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensors must have the same shape for element-wise division.");
    }

    Tensor<T> result(a.shape());
    for (int i = 0; i < result.data_.size(); ++i) {
        result.data_[i] = a.data_[i] / b.data_[i];
    }

    return result;
}


#endif
