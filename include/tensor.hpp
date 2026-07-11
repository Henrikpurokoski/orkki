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

  // get data from index
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

 private:
  // basic constructor
  Tensor(const std::vector<std::size_t>& shape);
  // data inside tensor
  std::vector<float> data_;
  // shape of tensor
  std::vector<std::size_t> shape_;
  // how many slots have to be skipped for each dimension
  std::vector<std::size_t> strides_;
};