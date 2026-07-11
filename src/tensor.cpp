#include "../include/tensor.hpp"

Tensor::Tensor(const std::vector<std::size_t>& shape) : shape_(shape) {
  const std::size_t size =
      std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                      std::multiplies<std::size_t>());
  const std::size_t dims = static_cast<std::size_t>(shape.size());

  data_.resize(size);
  strides_.resize(dims);

  std::size_t current_stride = 1;
  for (std::size_t i = dims; i-- > 0;) {
    strides_[i] = current_stride;
    current_stride *= shape[i];
  }
}

Tensor Tensor::Zeros(const std::vector<std::size_t>& shape) {
  return Tensor(shape);
};

Tensor Tensor::Random(const std::vector<std::size_t>& shape, float low,
                      float high) {
  static std::random_device rng_device;
  static std::mt19937 engine{rng_device()};
  std::uniform_real_distribution<float> dist{low, high};

  auto gen = [&]() { return dist(engine); };
  Tensor vec(shape);
  auto data_first = vec.data();
  auto data_len = vec.size();
  std::generate(data_first, data_first + data_len, gen);
  return vec;
};

Tensor Tensor::operator+(const Tensor& other) const {
  assert(shape_ == other.shape_);

  Tensor result(shape_);

  std::transform(data_.begin(), data_.end(), other.data_.begin(),
                 result.data_.begin(), std::plus<float>());

  return result;
};

Tensor Tensor::operator-(const Tensor& other) const {
  assert(shape_ == other.shape_);

  Tensor result(shape_);

  std::transform(data_.begin(), data_.end(), other.data_.begin(),
                 result.data_.begin(), std::minus<float>());

  return result;
};

Tensor Tensor::operator*(const Tensor& other) const {
  assert(shape_ == other.shape_);

  Tensor result(shape_);

  std::transform(data_.begin(), data_.end(), other.data_.begin(),
                 result.data_.begin(), std::multiplies<float>());

  return result;
};

Tensor Tensor::operator/(const Tensor& other) const {
  assert(shape_ == other.shape_);

  Tensor result(shape_);

  std::transform(data_.begin(), data_.end(), other.data_.begin(),
                 result.data_.begin(), std::divides<float>());

  return result;
};

Tensor Tensor::operator+(const float scalar) const {
  Tensor result(shape_);

  std::size_t size = data_.size();

  for (std::size_t i = 0; i < size; ++i) {
    result.data_[i] = data_[i] + scalar;
  }

  return result;
}

Tensor Tensor::operator-(const float scalar) const {
  Tensor result(shape_);

  std::size_t size = data_.size();

  for (std::size_t i = 0; i < size; ++i) {
    result.data_[i] = data_[i] - scalar;
  }

  return result;
}

Tensor Tensor::operator*(const float scalar) const {
  Tensor result(shape_);

  std::size_t size = data_.size();

  for (std::size_t i = 0; i < size; ++i) {
    result.data_[i] = data_[i] * scalar;
  }

  return result;
}

Tensor Tensor::operator/(const float scalar) const {
  Tensor result(shape_);

  std::size_t size = data_.size();

  for (std::size_t i = 0; i < size; ++i) {
    result.data_[i] = data_[i] / scalar;
  }

  return result;
}
