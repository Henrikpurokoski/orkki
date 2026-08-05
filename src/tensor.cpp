#include "../include/tensor.hpp"

TensorImpl::TensorImpl(const std::vector<std::size_t>& shape) : shape_(shape) {
  const std::size_t size =
      std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                      std::multiplies<std::size_t>());
  const std::size_t dims = static_cast<std::size_t>(shape.size());

  storage_ = std::make_shared<Storage>(size);
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
  return apply_binary([](float x, float y) { return x + y; }, other);
};

Tensor Tensor::operator-(const Tensor& other) const {
  return apply_binary([](float x, float y) { return x - y; }, other);
};

Tensor Tensor::operator*(const Tensor& other) const {
  return apply_binary([](float x, float y) { return x * y; }, other);
};

Tensor Tensor::operator/(const Tensor& other) const {
  return apply_binary([](float x, float y) { return x / y; }, other);
};

// scalar arithmetic
Tensor Tensor::operator+(const float scalar) const {
  return apply_unary([scalar](float x) { return x + scalar; });
}

Tensor Tensor::operator-(const float scalar) const {
  return apply_unary([scalar](float x) { return x - scalar; });
}

Tensor Tensor::operator*(const float scalar) const {
  return apply_unary([scalar](float x) { return x * scalar; });
}

Tensor Tensor::operator/(const float scalar) const {
  return apply_unary([scalar](float x) { return x / scalar; });
}

// reductions
float Tensor::sum() const { return apply_accumulate(tensor_ops::SumReducer{}); }

float Tensor::mean() const {
  return apply_accumulate(tensor_ops::MeanReducer{});
}

float Tensor::max() const { return apply_accumulate(tensor_ops::MaxReducer{}); }

float Tensor::min() const { return apply_accumulate(tensor_ops::MinReducer{}); }

Tensor Tensor::relu() const {
  return apply_unary([](float x) { return std::max(0.0f, x); });
}

Tensor Tensor::sigmoid() const {
  return apply_unary([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
}

Tensor Tensor::exp() const {
  return apply_unary([](float x) { return std::exp(x); });
}

Tensor Tensor::log() const {
  return apply_unary([](float x) { return std::log(x); });
}

Tensor Tensor::sqrt() const {
  return apply_unary([](float x) { return std::sqrt(x); });
}

Tensor Tensor::abs() const {
  return apply_unary([](float x) { return std::abs(x); });
}

// matrix multiplication
Tensor Tensor::matmul(const Tensor& other) const {
  auto shape = impl_->shape_;
  assert(shape.size() == 2);
  assert(other.shape().size() == 2);

  std::size_t t_rows = shape[0];
  std::size_t t_columns = shape[1];

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
  auto shape = impl_->shape_;
  assert(shape.size() == 2);

  std::size_t rows = shape[0];
  std::size_t cols = shape[1];

  std::vector<std::size_t> res_shape = {cols, rows};

  Tensor result(res_shape);

  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      result(j, i) = (*this)(i, j);
    }
  }
  return result;
}

std::vector<std::size_t> Tensor::broadcast_shape(
    const std::vector<std::size_t>& other_shape) const {
  auto shape = impl_->shape_;
  const auto& longer =
      (shape.size() >= other_shape.size()) ? shape : other_shape;
  std::vector<std::size_t> result(longer.size(), 1);
  std::vector<std::size_t> padded_t = result;
  std::vector<std::size_t> padded_o = result;

  auto t_pos = std::max(0, (static_cast<int>(other_shape.size()) -
                            static_cast<int>(shape.size())));
  auto o_pos = std::max(0, (static_cast<int>(shape.size()) -
                            static_cast<int>(other_shape.size())));

  for (std::size_t i = 0; i < longer.size(); ++i) {
    if (i >= t_pos) {
      padded_t[i] = shape[(i - t_pos)];
    }
    if (i >= o_pos) {
      padded_o[i] = other_shape[(i - o_pos)];
    }

    assert(padded_t[i] == padded_o[i] ||
           (padded_t[i] == 1 || padded_o[i] == 1));
    result[i] = std::max(padded_t[i], padded_o[i]);
  }

  return result;
}

const std::vector<std::size_t> Tensor::broadcast_strides(
    const std::vector<std::size_t>& result_shape,
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides) {
  std::vector<std::size_t> result(result_shape.size());
  std::size_t padded_dims = (result_shape.size() - shape.size());

  for (std::size_t i = 0; i < result.size(); ++i) {
    if (i >= padded_dims) {
      std::size_t og_i = i - padded_dims;

      result[i] = (shape[og_i] == 1) ? 0 : strides[og_i];
    } else
      result[i] = 0;
  }
  return result;
}

std::string Tensor::show() const {
  auto shape = impl_->shape_;
  std::stringstream o;
  if (shape.size() == 0) {
    o << "Cannot show empty tensor!\n";
  } else if (shape.size() > 2 ||
             *std::max_element(shape.begin(), shape.end()) > 30) {
    o << "Tensor too big to show!\n";
  } else if (shape.size() == 2) {
    std::size_t rows = shape[0];
    std::size_t cols = shape[1];
    for (std::size_t i = 0; i < rows; ++i) {
      o << "[";
      bool first = true;
      for (std::size_t j = 0; j < cols; ++j) {
        if (!first) {
          o << ", ";
        }
        float data = std::round((*this)(i, j) * 100.0f) / 100.0f;
        o << data;
        first = false;
      }
      o << "]\n";
    }
  } else if (shape.size() == 1) {
    o << "[";
    bool first = true;
    for (std::size_t i = 0; i < shape[0]; ++i) {
      if (!first) {
        o << ", ";
      }
      float data = std::round((*this)(i) * 100.0f) / 100.0f;
      o << data;
      first = false;
    }
    o << "]\n";
  }
  std::string str = o.str();
  return str;
}
