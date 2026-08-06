#include "../include/tensor.hpp"
// constructor
Tensor::Tensor(const std::vector<std::size_t>& shape, float value) {
  const std::size_t size =
      std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                      std::multiplies<std::size_t>());
  const std::size_t dims = static_cast<std::size_t>(shape.size());

  auto storage = std::make_shared<Storage>(size, value);
  auto strides = std::vector<std::size_t>(dims, 0);

  std::size_t current_stride = 1;
  for (std::size_t i = dims; i-- > 0;) {
    strides[i] = current_stride;
    current_stride *= shape[i];
  }
  impl_ = std::make_shared<TensorImpl>(storage, shape, strides, 0.0f);
}

// public
Tensor Tensor::Zeros(const std::vector<std::size_t>& shape) {
  return Tensor(shape, 0.0f);
};

Tensor Tensor::Ones(const std::vector<std::size_t>& shape) {
  return Tensor(shape, 1.0f);
};

Tensor Tensor::Full(const std::vector<std::size_t>& shape, float value) {
  return Tensor(shape, value);
};

Tensor Tensor::Random(const std::vector<std::size_t>& shape, float low,
                      float high) {
  static std::random_device rng_device;
  static std::mt19937 engine{rng_device()};
  std::uniform_real_distribution<float> dist{low, high};

  auto gen = [&]() { return dist(engine); };
  Tensor vec(shape, 0.0f);
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
Tensor Tensor::sum() const {
  return Tensor({}, apply_accumulate(tensor_ops::SumReducer{}));
}

Tensor Tensor::mean() const {
  return Tensor({}, apply_accumulate(tensor_ops::MeanReducer{}));
}

Tensor Tensor::max() const {
  return Tensor({}, apply_accumulate(tensor_ops::MaxReducer{}));
}

Tensor Tensor::min() const {
  return Tensor({}, apply_accumulate(tensor_ops::MinReducer{}));
}

// unary ops
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

  Tensor result(res_shape, 0.0f);

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

// transpose
Tensor Tensor::transpose() const {
  auto storage = impl_->storage_;
  auto shape = impl_->shape_;
  auto strides = impl_->strides_;

  std::reverse(shape.begin(), shape.end());
  std::reverse(strides.begin(), strides.end());
  auto impl = std::make_shared<TensorImpl>(storage, shape, strides, 0);
  return Tensor(impl);
}

// slice
Tensor Tensor::slice(
    std::vector<std::pair<std::size_t, std::size_t>> slices) const {
  auto storage = impl_->storage_;
  auto shape = impl_->shape_;
  auto strides = impl_->strides_;
  auto offset = impl_->offset_;

  assert(slices.size() == shape.size());

  for (std::size_t i = 0; i < slices.size(); ++i) {
    auto first = slices[i].first;
    auto second = slices[i].second;

    assert(first <= second);
    assert(second <= shape[i]);
    auto new_shape = second - first;
    shape[i] = new_shape;
    offset += first * strides[i];
  }
  auto impl = std::make_shared<TensorImpl>(storage, shape, strides, offset);
  return Tensor(impl);
}

std::string Tensor::show(std::size_t max_k, std::size_t precision) const {
  std::stringstream c;
  c << std::fixed << std::setprecision(precision);
  auto shape = impl_->shape_;
  if (shape.size() == 0) {
    c << "[" << (*this)() << "]";
  } else {
    std::vector<std::size_t> cur_idx(shape.size(), 0);
    format_dim(c, 0, cur_idx, max_k);
  }
  return c.str();
}

// private
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

void Tensor::format_dim(std::stringstream& c, std::size_t cur_dim,
                        std::vector<std::size_t>& cur_idx,
                        std::size_t k) const {
  auto shape = impl_->shape_;
  auto strides = impl_->strides_;
  auto offset = impl_->offset_;

  std::size_t total_dims = shape.size();
  std::size_t rev_dim = (total_dims - cur_dim - 1);
  std::string sep = std::string(rev_dim, '\n');
  auto cur_size = shape[cur_dim];

  if (rev_dim == 0) {
    if (cur_size > k * 2) {
      c << "[";
      for (std::size_t i = 0; i < k; ++i) {
        cur_idx[cur_dim] = i;
        auto flat_index = offset;
        for (std::size_t j = 0; j < total_dims; ++j) {
          flat_index += cur_idx[j] * strides[j];
        }
        c << std::round((*this).data()[flat_index] * 100.0f) / 100.0f;
        c << ", ";
      }
      c << "..., ";
      for (std::size_t i = cur_size - k; i < cur_size; ++i) {
        cur_idx[cur_dim] = i;
        auto flat_index = offset;
        for (std::size_t j = 0; j < total_dims; ++j) {
          flat_index += cur_idx[j] * strides[j];
        }
        c << std::round((*this).data()[flat_index] * 100.0f) / 100.0f;
        if (i < cur_size - 1) {
          c << ", ";
        }
      }
      c << "]";
    } else {
      c << "[";
      for (std::size_t i = 0; i < cur_size; ++i) {
        cur_idx[cur_dim] = i;
        auto flat_index = offset;
        for (std::size_t j = 0; j < total_dims; ++j) {
          flat_index += cur_idx[j] * strides[j];
        }
        c << std::round((*this).data()[flat_index] * 100.0f) / 100.0f;
        if (i < cur_size - 1) {
          c << ", ";
        }
      }
      c << "]";
    }

  } else {
    if (cur_size > k * 2) {
      c << "[";
      for (std::size_t i = 0; i < k; ++i) {
        cur_idx[cur_dim] = i;
        format_dim(c, cur_dim + 1, cur_idx, k);
        c << "," << sep;
      }
      c << "...," << sep;
      for (std::size_t i = cur_size - k; i < cur_size; ++i) {
        cur_idx[cur_dim] = i;
        format_dim(c, cur_dim + 1, cur_idx, k);
        if (i < cur_size - 1) {
          c << "," << sep;
        }
      }
      c << "]";
    } else {
      c << "[";
      for (std::size_t i = 0; i < cur_size; ++i) {
        cur_idx[cur_dim] = i;
        format_dim(c, cur_dim + 1, cur_idx, k);
        if (i < cur_size - 1) {
          c << "," << sep;
        }
      }
      c << "]";
    }
  }
}
