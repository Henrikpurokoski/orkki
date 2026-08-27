#pragma once
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "math_components.hpp"
#include "node.hpp"

// for constructing tensors
struct TensorConstructorNode {
  bool leaf_;
  float value_;
  std::vector<TensorConstructorNode> children_;

  TensorConstructorNode(const float value) : value_(value), leaf_(true) {}
  TensorConstructorNode(
      const std::initializer_list<TensorConstructorNode>& children)
      : children_(children), leaf_(false) {}
};

// a separate storage struct: saves just the tensor data
struct Storage {
  // dator
  std::vector<float> data_;

  // constructoror
  explicit Storage(std::vector<float>& data) : data_(data) {}
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
  bool requires_grad_;

  // getters
  float* data() {
    if (storage_ == nullptr) {
      throw std::runtime_error("Cannot use data(), tensor has no storage");
    }
    return storage_->data_.data();
  }
  const float* data() const {
    if (storage_ == nullptr) {
      throw std::runtime_error("Cannot use data(), tensor has no storage");
    }
    return storage_->data_.data();
  }
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
  // constructors
  Tensor(std::initializer_list<TensorConstructorNode> nodes);

  static Tensor Zeros(const std::vector<std::size_t>& shape,
                      const bool& requires_grad = false);

  static Tensor Ones(const std::vector<std::size_t>& shape,
                     const bool& requires_grad = false);

  static Tensor Full(const std::vector<std::size_t>& shape, float value,
                     const bool& requires_grad = false);

  static Tensor Random(const std::vector<std::size_t>& shape, float low = 0.0f,
                       float high = 1.0f, const bool& requires_grad = false);

  static Tensor String(std::string tensor_string,
                       const bool& requires_grad = false);

  // is defined?
  bool defined() const { return impl_ != nullptr; }
  explicit operator bool() const { return defined(); }

  // getters
  float* data() { return impl_->data(); }
  const float* data() const {
    if (!defined()) {
      throw std::runtime_error(
          "Cannot use data(), tensor has no implementation");
    }
    return impl_->data();
  }
  std::size_t size() const {
    if (!defined()) {
      throw std::runtime_error(
          "Cannot use size(), tensor has no implementation");
    }
    return impl_->size();
  }
  const std::vector<std::size_t>& shape() const {
    if (!defined()) {
      throw std::runtime_error(
          "Cannot use shape(), tensor has no implementation");
    }
    return impl_->shape_;
  }
  const std::vector<std::size_t>& strides() const {
    if (!defined()) {
      throw std::runtime_error(
          "Cannot use strides(), tensor has no implementation");
    }
    return impl_->strides_;
  }
  const bool requires_grad() const {
    if (!defined()) {
      throw std::runtime_error(
          "Cannot use requires_grad(), tensor has no implementation");
    }
    return impl_->requires_grad_;
  }
  const Tensor grad() const {
    if (!defined()) {
      throw std::runtime_error(
          "Cannot use grad(), tensor has no implementation");
    }
    return Tensor(impl_->grad_);
  }
  const std::shared_ptr<Node>& child_node() const {
    if (!defined()) {
      throw std::runtime_error(
          "Cannot use child_node(), tensor has no implementation");
    }
    return impl_->child_node_;
  }

  // graph checks
  bool leaf() const {
    if (!defined()) {
      throw std::runtime_error(
          "Cannot use leaf(), tensor has no implementation");
    }
    return impl_->child_node_ == nullptr;
  }

  // set gradient
  void set_grad(bool grad_bool) const;

  // indexing
  template <typename... Args>
  float& operator()(Args... args) {
    if (!defined()) {
      throw std::runtime_error(
          "Cannot use operator(), tensor has no implementation");
    }
    static_assert((std::is_integral_v<Args> && ...));

    if (sizeof...(args) != impl_->shape_.size()) {
      throw std::runtime_error("Wrong number of arguments in operator()");
    }
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
    if (!defined()) {
      throw std::runtime_error(
          "Cannot use operator(), tensor has no implementation");
    }
    static_assert((std::is_integral_v<Args> && ...));

    if (sizeof...(args) != impl_->shape_.size()) {
      throw std::runtime_error("Wrong number of arguments in operator()");
    }
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
  std::string tree() const;

 private:
  // constructor
  explicit Tensor(const std::shared_ptr<TensorImpl> impl) : impl_(impl) {}

  // overloaded constructor for ease of life
  explicit Tensor(const std::vector<std::size_t>& shape,
                  std::vector<float>& data, const bool& requires_grad = false,
                  const std::shared_ptr<Node> child_node = nullptr);

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
    if (!defined()) {
      throw std::runtime_error(
          "Cannot use apply_unary(), tensor has no implementation");
    }
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
      result.data()[i] = op(data()[iter_t]);
    }
    return result;
  }
  // apply accumulation (usually returns 0d tensor)
  template <typename Reducer>
  auto apply_accumulate(Reducer reducer) const {
    if (!defined()) {
      throw std::runtime_error(
          "Cannot use apply_accumulate(), tensor has no implementation");
    }
    auto shape = impl_->shape_;
    auto strides = impl_->strides_;

    std::vector<std::size_t> temp_strides(shape.size());
    std::size_t current_stride = 1;
    for (std::size_t ii = shape.size(); ii-- > 0;) {
      temp_strides[ii] = current_stride;
      current_stride *= shape[ii];
    }

    for (std::size_t i = 0; i < size(); ++i) {
      std::size_t iter_t = impl_->offset_;
      std::size_t remaining = i;
      for (std::size_t j = 0; j < shape.size(); ++j) {
        std::size_t cur = remaining / temp_strides[j];
        remaining -= cur * temp_strides[j];
        iter_t += cur * strides[j];
      }
      reducer.update(data()[iter_t]);
    }
    return reducer.finalize(size());
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
    if (!defined()) {
      throw std::runtime_error(
          "Cannot use apply_binary(), tensor has no implementation");
    }
    if (!other.defined()) {
      throw std::runtime_error(
          "Cannot use apply_binary(), tensor has no implementation");
    }
    auto result_shape = broadcast_shape(other.shape());
    bool requires_grad = node != nullptr;

    Tensor result(result_shape, 0.0f, requires_grad, node);
    result.set_parent(node);

    std::vector<std::size_t> strides_t =
        Tensor::broadcast_strides(result_shape, shape(), strides());
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
      result.data()[i] = op(data()[iter_t], other.data()[iter_o]);
    }
    return result;
  }

  void format_dim(std::stringstream& c, std::size_t cur_dim,
                  std::vector<std::size_t>& cur_idx, std::size_t k) const;

  static std::size_t parse_string(std::string& s, std::size_t& cursor,
                                  std::size_t depth,
                                  std::vector<std::size_t>& shape,
                                  std::vector<float>& data);

  static std::size_t parse_constructor_tree(TensorConstructorNode& cur,
                                            std::size_t depth,
                                            std::vector<std::size_t>& shape,
                                            std::vector<float>& data);
};
