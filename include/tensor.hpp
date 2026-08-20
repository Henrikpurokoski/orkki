#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <type_traits>
#include <vector>

#include "math_components.hpp"
#include "node.hpp"

// a separate storage struct: saves just the tensor data
struct Storage {
  // dator
  std::vector<float> data_;

  // constructoror
  explicit Storage(const std::size_t size, float value) : data_(size, value) {}
};

// tensor implementation: saves tensor metadata and points to a storage struct
struct TensorImpl {
  // pointer to storage
  std::shared_ptr<Storage> storage_;
  // pointer to gradient implementation
  std::shared_ptr<TensorImpl> grad_ = nullptr;
  // node this tensor is a parent of
  std::shared_ptr<Node> child_node_;
  // shape of tensor
  const std::vector<std::size_t> shape_;
  // how many slots have to be skipped for each dimension
  const std::vector<std::size_t> strides_;
  // offset for later to use in slicing etc
  const std::size_t offset_ = 0;
  // requires grad flag
  const bool requires_grad_;

  // getters
  float* data() { return storage_->data_.data(); }
  const float* data() const { return storage_->data_.data(); }
  std::size_t size() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), std::size_t{1},
                           std::multiplies<std::size_t>());
  }
  // constructor
  explicit TensorImpl(const std::shared_ptr<Storage>& storage,
                      const std::vector<std::size_t>& shape,
                      const std::vector<std::size_t>& strides,
                      const std::size_t& offset, const bool& requires_grad,
                      const std::shared_ptr<Node>& child_node)
      : storage_(storage),
        shape_(shape),
        strides_(strides),
        offset_(offset),
        requires_grad_(requires_grad),
        child_node_(child_node) {}
};

// tensor class: API for tensor impl. Only stores a pointer to a tensor impl
class Tensor {
 public:
  // four constructors: initiate with zeros, ones, any value and with random
  static Tensor Random(const std::vector<std::size_t>& shape, float low = 0.0f,
                       float high = 1.0f, const bool& requires_grad = false);
  static Tensor Zeros(const std::vector<std::size_t>& shape,
                      const bool& requires_grad = false);

  static Tensor Ones(const std::vector<std::size_t>& shape,
                     const bool& requires_grad = false);

  static Tensor Full(const std::vector<std::size_t>& shape, float value,
                     const bool& requires_grad = false);

  // is defined?
  bool defined() const { return impl_ != nullptr; }
  explicit operator bool() const { return defined(); }

  // getters
  float* data() { return impl_->data(); }
  const float* data() const {
    assert(defined());
    return impl_->data();
  }
  std::size_t size() const {
    assert(defined());
    return impl_->size();
  }
  const std::vector<std::size_t>& shape() const {
    assert(defined());
    return impl_->shape_;
  }
  const std::vector<std::size_t>& strides() const {
    assert(defined());
    return impl_->strides_;
  }
  const bool requires_grad() const {
    assert(defined());
    return impl_->requires_grad_;
  }
  const Tensor grad() const {
    assert(defined());
    return Tensor(impl_->grad_);
  }
  const std::shared_ptr<Node>& child_node() const {
    assert(defined());
    return impl_->child_node_;
  }

  // graph checks
  bool isLeaf() const { return impl_->child_node_ == nullptr; }

  // indexing
  template <typename... Args>
  float& operator()(Args... args) {
    assert(defined());
    static_assert((std::is_integral_v<Args> && ...));

    assert(sizeof...(args) == impl_->shape_.size());
    std::size_t indices[] = {static_cast<std::size_t>(args)...};

    std::size_t flat_index = impl_->offset_;
    for (std::size_t i = 0; i < sizeof...(args); ++i) {
      assert(indices[i] < static_cast<std::size_t>(impl_->shape_[i]));
      flat_index += indices[i] * impl_->strides_[i];
    }
    return impl_->storage_->data_[flat_index];
  }

  template <typename... Args>
  const float& operator()(Args... args) const {
    assert(defined());
    static_assert((std::is_integral_v<Args> && ...));

    assert(sizeof...(args) == impl_->shape_.size());
    std::size_t indices[] = {static_cast<std::size_t>(args)...};

    std::size_t flat_index = impl_->offset_;
    for (std::size_t i = 0; i < sizeof...(args); ++i) {
      assert(indices[i] < static_cast<std::size_t>(impl_->shape_[i]));
      flat_index += indices[i] * impl_->strides_[i];
    }
    return impl_->storage_->data_[flat_index];
  }

  // elementwise arithmetic
  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator/(const Tensor& other) const;

  // scalar arithmetic
  Tensor operator+(const float scalar) const;
  Tensor operator-(const float scalar) const;
  Tensor operator*(const float scalar) const;
  Tensor operator/(const float scalar) const;

  // reductions
  Tensor sum() const;
  Tensor mean() const;
  Tensor max() const;
  Tensor min() const;

  // unary operations
  Tensor relu() const;
  Tensor sigmoid() const;
  Tensor exp() const;
  Tensor log() const;
  Tensor sqrt() const;
  Tensor abs() const;

  // matrix multiplication
  Tensor matmul(const Tensor& other) const;

  // transpose
  Tensor transpose() const;

  // slice
  Tensor slice(std::vector<std::pair<std::size_t, std::size_t>> slices) const;

  // copy
  Tensor copy() const;

  // info
  std::string show(std::size_t max_k = 5, std::size_t precision = 2) const;
  std::string info() const;
  std::string child_info() const;

 private:
  // basic constructor
  explicit Tensor(const std::shared_ptr<TensorImpl> impl) : impl_(impl) {}

  // overloaded constructor for ease of life
  explicit Tensor(const std::vector<std::size_t>& shape, float value,
                  const bool& requires_grad = false,
                  const std::shared_ptr<Node> child_node = nullptr);
  // pointer to implementation
  std::shared_ptr<TensorImpl> impl_;

  // set as a parent of a node if exists
  void set_parent(const std::shared_ptr<Node>& node) const;

  // apply unary operator
  template <typename unary>
  Tensor apply_unary(unary op, const std::shared_ptr<Node>& node) const {
    assert(defined());
    auto shape = impl_->shape_;
    auto strides = impl_->strides_;
    bool requires_grad = node != nullptr;

    Tensor result(shape, 0.0f, requires_grad, node);
    result.set_parent(node);

    for (std::size_t i = 0; i < result.size(); ++i) {
      std::size_t iter_t = impl_->offset_;
      std::size_t remaining = i;
      for (std::size_t j = 0; j < shape.size(); ++j) {
        std::size_t cur = remaining / result.strides()[j];
        remaining -= cur * result.strides()[j];
        iter_t += cur * strides[j];
      }
      result.data()[i] = op((*this).data()[iter_t]);
    }
    return result;
  }
  // apply accumulation (usually returns 0d tensor)
  template <typename Reducer>
  auto apply_accumulate(Reducer reducer) const {
    assert(defined());
    auto shape = impl_->shape_;
    auto strides = impl_->strides_;

    std::vector<std::size_t> temp_strides(shape.size());
    std::size_t current_stride = 1;
    for (std::size_t ii = shape.size(); ii-- > 0;) {
      temp_strides[ii] = current_stride;
      current_stride *= shape[ii];
    }

    for (std::size_t i = 0; i < (*this).size(); ++i) {
      std::size_t iter_t = impl_->offset_;
      std::size_t remaining = i;
      for (std::size_t j = 0; j < shape.size(); ++j) {
        std::size_t cur = remaining / temp_strides[j];
        remaining -= cur * temp_strides[j];
        iter_t += cur * strides[j];
      }
      reducer.update((*this).data()[iter_t]);
    }
    return reducer.finalize((*this).size());
  }

  // apply binary operator (broadcast)
  std::vector<std::size_t> broadcast_shape(
      const std::vector<std::size_t>& other_shape) const;

  static const std::vector<std::size_t> broadcast_strides(
      const std::vector<std::size_t>& result_shape,
      const std::vector<std::size_t>& shape,
      const std::vector<std::size_t>& strides);

  template <typename binary>
  Tensor apply_binary(binary op, const Tensor& other,
                      const std::shared_ptr<Node>& node) const {
    assert(defined());
    assert(other.defined());
    auto result_shape = (*this).broadcast_shape(other.shape());
    bool requires_grad = node != nullptr;

    Tensor result(result_shape, 0.0f, requires_grad, node);
    result.set_parent(node);

    std::vector<std::size_t> strides_t = Tensor::broadcast_strides(
        result_shape, (*this).shape(), (*this).strides());
    std::vector<std::size_t> strides_o =
        Tensor::broadcast_strides(result_shape, other.shape(), other.strides());

    for (std::size_t i = 0; i < result.size(); ++i) {
      std::size_t iter_t = impl_->offset_;
      std::size_t iter_o = other.impl_->offset_;
      std::size_t remaining = i;
      for (std::size_t j = 0; j < result.shape().size(); ++j) {
        std::size_t cur = remaining / result.strides()[j];
        remaining -= cur * result.strides()[j];
        iter_t += cur * strides_t[j];
        iter_o += cur * strides_o[j];
      }
      result.data()[i] = op((*this).data()[iter_t], other.data()[iter_o]);
    }
    return result;
  }

  void format_dim(std::stringstream& c, std::size_t cur_dim,
                  std::vector<std::size_t>& cur_idx, std::size_t k) const;
};