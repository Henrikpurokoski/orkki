#include "../include/tensor.hpp"

#include <iostream>

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

// elementwise arithmetic
Tensor Tensor::operator+(const Tensor& other) const {
  return broadcast_apply([](float x, float y) { return x + y; }, other);
};

Tensor Tensor::operator-(const Tensor& other) const {
  return broadcast_apply([](float x, float y) { return x - y; }, other);
};

Tensor Tensor::operator*(const Tensor& other) const {
  return broadcast_apply([](float x, float y) { return x * y; }, other);
};

Tensor Tensor::operator/(const Tensor& other) const {
  return broadcast_apply([](float x, float y) { return x / y; }, other);
};

// scalar arithmetic
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

// elementwise compound arithmetic
Tensor& Tensor::operator+=(const Tensor& other) {
  return broadcast_apply([](float x, float y) { return x + y; }, other);
}

Tensor& Tensor::operator-=(const Tensor& other) {
  return broadcast_apply([](float x, float y) { return x - y; }, other);
}

Tensor& Tensor::operator*=(const Tensor& other) {
  return broadcast_apply([](float x, float y) { return x * y; }, other);
}

Tensor& Tensor::operator/=(const Tensor& other) {
  return broadcast_apply([](float x, float y) { return x / y; }, other);
}

// scalar compound arithmetic
Tensor& Tensor::operator+=(const float scalar) {
  std::size_t size = data_.size();

  for (std::size_t i = 0; i < size; ++i) {
    data_[i] += scalar;
  }

  return *this;
}

Tensor& Tensor::operator-=(const float scalar) {
  std::size_t size = data_.size();

  for (std::size_t i = 0; i < size; ++i) {
    data_[i] -= scalar;
  }

  return *this;
}

Tensor& Tensor::operator*=(const float scalar) {
  std::size_t size = data_.size();

  for (std::size_t i = 0; i < size; ++i) {
    data_[i] *= scalar;
  }

  return *this;
}

Tensor& Tensor::operator/=(const float scalar) {
  std::size_t size = data_.size();

  for (std::size_t i = 0; i < size; ++i) {
    data_[i] /= scalar;
  }

  return *this;
}

// reductions
float Tensor::sum() const {
  return std::accumulate(data_.begin(), data_.end(), 0.0f);
}

float Tensor::mean() const { return sum() / static_cast<float>(data_.size()); }

float Tensor::max() const {
  return *std::max_element(data_.begin(), data_.end());
}

float Tensor::min() const {
  return *std::min_element(data_.begin(), data_.end());
}

// matrix multiplication
Tensor Tensor::matmul(const Tensor& other) const {
  assert(shape_.size() == 2);
  assert(other.shape().size() == 2);

  std::size_t t_rows = shape_[0];
  std::size_t t_columns = shape_[1];

  std::size_t o_rows = other.shape()[0];
  std::size_t o_columns = other.shape()[1];

  assert(t_columns == o_rows);

  std::vector<std::size_t> res_shape = {t_rows, o_columns};

  Tensor result(res_shape);

  for (std::size_t i = 0; i < o_columns; ++i) {
    for (std::size_t j = 0; j < t_rows; ++j) {
      float cur = 0.0f;
      for (std::size_t k = 0; k < t_columns; ++k) {
        cur += (*this)(j, k) * other(k, i);
      }
      result(j, i) = cur;
    }
  }
  return result;
}

Tensor Tensor::transpose() const {
  assert(shape_.size() == 2);

  std::size_t rows = shape_[0];
  std::size_t cols = shape_[1];

  std::vector<std::size_t> res_shape = {cols, rows};

  Tensor result(res_shape);

  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      result(j, i) = (*this)(i, j);
    }
  }
  return result;
}

Tensor Tensor::relu() const {
  return apply([](float x) { return std::max(0.0f, x); });
}

Tensor Tensor::sigmoid() const {
  return apply([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
}

Tensor Tensor::exp() const {
  return apply([](float x) { return std::exp(x); });
}

Tensor Tensor::log() const {
  return apply([](float x) { return std::log(x); });
}

Tensor Tensor::sqrt() const {
  return apply([](float x) { return std::sqrt(x); });
}

Tensor Tensor::abs() const {
  return apply([](float x) { return std::abs(x); });
}

const std::tuple<std::vector<std::size_t>, std::vector<std::size_t>,
                 std::vector<std::size_t>>
Tensor::broadcast_shape(const std::vector<std::size_t>& other_shape) const {
  const auto& longer =
      (shape_.size() >= other_shape.size()) ? shape_ : other_shape;
  std::vector<std::size_t> result(longer.size(), 1);
  std::vector<std::size_t> padded_t = result;
  std::vector<std::size_t> padded_o = result;

  auto t_pos = std::max(0, (static_cast<int>(other_shape.size()) -
                            static_cast<int>(shape_.size())));
  auto o_pos = std::max(0, (static_cast<int>(shape_.size()) -
                            static_cast<int>(other_shape.size())));

  for (std::size_t i = 0; i < longer.size(); ++i) {
    if (i >= t_pos) {
      padded_t[i] = shape_[(i - t_pos)];
    }
    if (i >= o_pos) {
      padded_o[i] = other_shape[(i - o_pos)];
    }

    assert(padded_t[i] == padded_o[i] ||
           (padded_t[i] == 1 || padded_o[i] == 1));
    result[i] = std::max(padded_t[i], padded_o[i]);
  }

  return std::make_tuple(result, padded_t, padded_o);
}

const std::vector<std::size_t> Tensor::broadcast_strides(
    const std::vector<std::size_t> padded_shape) {
  std::vector<std::size_t> result = padded_shape;

  std::size_t current_stride = 1;
  for (std::size_t i = padded_shape.size(); i-- > 0;) {
    result[i] = (padded_shape[i] == 1) ? 0 : current_stride;
    current_stride *= padded_shape[i];
  }
  return result;
}
