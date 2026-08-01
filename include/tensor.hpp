#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

// tensor class header
class Tensor {
 public:
  // two constructors: initiate with zeros and with random
  static Tensor Random(const std::vector<std::size_t>& shape, float low,
                       float high);
  static Tensor Zeros(const std::vector<std::size_t>& shape);

  // getters
  float* data() { return data_.data(); }
  const float* data() const { return data_.data(); }
  std::size_t size() const { return data_.size(); }
  const std::vector<std::size_t>& shape() const { return shape_; }
  const std::vector<std::size_t>& strides() const { return strides_; }

  // indexing
  template <typename... Args>
  float& operator()(Args... args) {
    static_assert((std::is_integral_v<Args> && ...));

    assert(sizeof...(args) == shape_.size());
    std::size_t indices[] = {static_cast<std::size_t>(args)...};

    std::size_t flat_index = 0;
    for (std::size_t i = 0; i < sizeof...(args); ++i) {
      assert(indices[i] < static_cast<std::size_t>(shape_[i]));
      flat_index += indices[i] * strides_[i];
    }
    return data_[flat_index];
  }

  template <typename... Args>
  const float& operator()(Args... args) const {
    static_assert((std::is_integral_v<Args> && ...));

    assert(sizeof...(args) == shape_.size());
    std::size_t indices[] = {static_cast<std::size_t>(args)...};

    std::size_t flat_index = 0;
    for (std::size_t i = 0; i < sizeof...(args); ++i) {
      assert(indices[i] < static_cast<std::size_t>(shape_[i]));
      flat_index += indices[i] * strides_[i];
    }
    return data_[flat_index];
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

  // elementwise compound arithmetic
  Tensor& operator+=(const Tensor& other);
  Tensor& operator-=(const Tensor& other);
  Tensor& operator*=(const Tensor& other);
  Tensor& operator/=(const Tensor& other);

  // scalar compound arithmetic
  Tensor& operator+=(const float scalar);
  Tensor& operator-=(const float scalar);
  Tensor& operator*=(const float scalar);
  Tensor& operator/=(const float scalar);

  // reductions
  float sum() const;
  float mean() const;
  float max() const;
  float min() const;

  // matrix multiplication
  Tensor matmul(const Tensor& other) const;

  // transpose
  Tensor transpose() const;

  // activations
  Tensor relu() const;
  Tensor sigmoid() const;
  Tensor exp() const;
  Tensor log() const;
  Tensor sqrt() const;
  Tensor abs() const;

 private:
  // basic constructor
  Tensor(const std::vector<std::size_t>& shape);
  // data inside tensor
  std::vector<float> data_;
  // shape of tensor
  std::vector<std::size_t> shape_;
  // how many slots have to be skipped for each dimension
  std::vector<std::size_t> strides_;

  // apply unary operator
  template <typename unary>
  Tensor apply(unary op) const {
    Tensor result(shape_);

    for (std::size_t i = 0; i < data_.size(); ++i) {
      result.data_[i] = op(data_[i]);
    }
    return result;
  }

  // broadcast
  const std::tuple<std::vector<std::size_t>, std::vector<std::size_t>,
                   std::vector<std::size_t>>
  broadcast_shape(const std::vector<std::size_t>& other_shape) const;
  static const std::vector<std::size_t> broadcast_strides(
      const std::vector<std::size_t> padded_shape);

  template <typename binary>
  Tensor broadcast_apply(binary op, const Tensor& other) const {
    auto [result_shape, t_shape, o_shape] =
        (*this).broadcast_shape(other.shape_);

    Tensor result(result_shape);

    std::vector<std::size_t> strides_t = Tensor::broadcast_strides(t_shape);
    std::vector<std::size_t> strides_o = Tensor::broadcast_strides(o_shape);

    for (std::size_t i = 0; i < result.data_.size(); ++i) {
      std::size_t iter_t = 0;
      std::size_t iter_o = 0;
      std::size_t remaining = i;
      for (std::size_t j = 0; j < result.shape_.size(); ++j) {
        std::size_t cur = remaining / result.strides_[j];
        remaining -= cur * result.strides_[j];
        iter_t += cur * strides_t[j];
        iter_o += cur * strides_o[j];
      }
      result.data_[i] = op(data_[iter_t], other.data_[iter_o]);
    }
    return result;
  }

  template <typename binary>
  Tensor& broadcast_apply(binary op, const Tensor& other) {
    auto [result_shape, t_shape, o_shape] =
        (*this).broadcast_shape(other.shape_);
    assert(result_shape == shape_);

    std::vector<std::size_t> strides_t = Tensor::broadcast_strides(t_shape);
    std::vector<std::size_t> strides_o = Tensor::broadcast_strides(o_shape);

    for (std::size_t i = 0; i < data_.size(); ++i) {
      std::size_t iter_t = 0;
      std::size_t iter_o = 0;
      std::size_t remaining = i;
      for (std::size_t j = 0; j < shape_.size(); ++j) {
        std::size_t cur = remaining / strides_[j];
        remaining -= cur * strides_[j];
        iter_t += cur * strides_t[j];
        iter_o += cur * strides_o[j];
      }
      data_[i] = op(data_[iter_t], other.data_[iter_o]);
    }
    return *this;
  }
};