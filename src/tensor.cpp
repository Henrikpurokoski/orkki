#include "../include/tensor.hpp"

// private constructor
Tensor::Tensor(const std::vector<std::size_t>& shape, std::vector<float>& data,
               const bool& requires_grad,
               const std::shared_ptr<Node> child_node) {
  const std::size_t size =
      std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                      std::multiplies<std::size_t>());
  const std::size_t dims = shape.size();

  assert(data.size() == size);
  auto storage = std::make_shared<Storage>(data);
  auto strides = std::vector<std::size_t>(dims, 0);

  std::size_t current_stride = 1;
  for (std::size_t i = dims; i-- > 0;) {
    strides[i] = current_stride;
    current_stride *= shape[i];
  }
  impl_ = std::make_shared<TensorImpl>(storage, shape, strides, 0.0f,
                                       requires_grad, child_node);
}

Tensor::Tensor(const std::vector<std::size_t>& shape, float value,
               const bool& requires_grad,
               const std::shared_ptr<Node> child_node) {
  const std::size_t size =
      std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                      std::multiplies<std::size_t>());
  const std::size_t dims = shape.size();

  auto data = std::vector<float>(size, value);
  auto storage = std::make_shared<Storage>(data);
  auto strides = std::vector<std::size_t>(dims, 0);

  std::size_t current_stride = 1;
  for (std::size_t i = dims; i-- > 0;) {
    strides[i] = current_stride;
    current_stride *= shape[i];
  }
  impl_ = std::make_shared<TensorImpl>(storage, shape, strides, 0.0f,
                                       requires_grad, child_node);
}

// public
Tensor::Tensor(std::initializer_list<TensorConstructorNode> nodes) {
  std::vector<std::size_t> shape;
  std::vector<float> data;
  auto root = TensorConstructorNode(nodes);
  parse_constructor_tree(root, 0, shape, data);

  const std::size_t dims = shape.size();
  auto storage = std::make_shared<Storage>(data);
  auto strides = std::vector<std::size_t>(dims, 0);

  std::size_t current_stride = 1;
  for (std::size_t i = dims; i-- > 0;) {
    strides[i] = current_stride;
    current_stride *= shape[i];
  }
  impl_ = std::make_shared<TensorImpl>(storage, shape, strides, 0.0f, false,
                                       nullptr);
}

Tensor Tensor::Zeros(const std::vector<std::size_t>& shape,
                     const bool& requires_grad) {
  return Tensor(shape, 0.0f, requires_grad);
};

Tensor Tensor::Ones(const std::vector<std::size_t>& shape,
                    const bool& requires_grad) {
  return Tensor(shape, 1.0f, requires_grad);
};

Tensor Tensor::Full(const std::vector<std::size_t>& shape, float value,
                    const bool& requires_grad) {
  return Tensor(shape, value, requires_grad);
};

Tensor Tensor::Random(const std::vector<std::size_t>& shape, float low,
                      float high, const bool& requires_grad) {
  static std::random_device rng_device;
  static std::mt19937 engine{rng_device()};
  std::uniform_real_distribution<float> dist{low, high};

  auto gen = [&]() { return dist(engine); };
  Tensor vec(shape, 0.0f, requires_grad);
  auto data_first = vec.data();
  auto data_len = vec.size();
  std::generate(data_first, data_first + data_len, gen);
  return vec;
};

Tensor Tensor::String(std::string tensor_string, const bool& requires_grad) {
  std::size_t cur = 0;
  std::vector<std::size_t> shape;
  std::vector<float> data;
  parse_string(tensor_string, cur, 0, shape, data);
  while (tensor_string[cur] == ' ' || tensor_string[cur] == '\n') {
    ++cur;
  }
  if (cur < tensor_string.length()) {
    std::stringstream error_msg;
    error_msg << "Unexpected character: '" << tensor_string[cur]
              << "' at pos: " << cur << ".";
    throw std::runtime_error(error_msg.str());
  }
  return Tensor(shape, data, requires_grad);
}

// elementwise arithmetic
Tensor Tensor::operator+(const Tensor& other) const {
  std::shared_ptr<Node> node = nullptr;
  if ((*this).requires_grad() || other.requires_grad()) {
    std::vector<std::shared_ptr<TensorImpl>> operands{(*this).impl_,
                                                      other.impl_};
    node = std::make_shared<AddNode>(operands);
  }

  return apply_binary([](float x, float y) { return x + y; }, other, node);
};

Tensor Tensor::operator-(const Tensor& other) const {
  std::shared_ptr<Node> node = nullptr;
  if ((*this).requires_grad() || other.requires_grad()) {
    std::vector<std::shared_ptr<TensorImpl>> operands{(*this).impl_,
                                                      other.impl_};
    node = std::make_shared<SubNode>(operands);
  }
  return apply_binary([](float x, float y) { return x - y; }, other, node);
};

Tensor Tensor::operator*(const Tensor& other) const {
  std::shared_ptr<Node> node = nullptr;
  if ((*this).requires_grad() || other.requires_grad()) {
    std::vector<std::shared_ptr<TensorImpl>> operands{(*this).impl_,
                                                      other.impl_};
    node = std::make_shared<MulNode>(operands);
  }
  return apply_binary([](float x, float y) { return x * y; }, other, node);
};

Tensor Tensor::operator/(const Tensor& other) const {
  std::shared_ptr<Node> node = nullptr;
  if ((*this).requires_grad() || other.requires_grad()) {
    std::vector<std::shared_ptr<TensorImpl>> operands{(*this).impl_,
                                                      other.impl_};
    node = std::make_shared<DivNode>(operands);
  }
  return apply_binary([](float x, float y) { return x / y; }, other, node);
};

// scalar arithmetic
Tensor Tensor::operator+(const float scalar) const {
  Tensor other = Tensor({}, scalar, false);
  return (*this) + other;
}

Tensor Tensor::operator-(const float scalar) const {
  Tensor other = Tensor({}, scalar, false);
  return (*this) - other;
}

Tensor Tensor::operator*(const float scalar) const {
  Tensor other = Tensor({}, scalar, false);
  return (*this) * other;
}

Tensor Tensor::operator/(const float scalar) const {
  Tensor other = Tensor({}, scalar, false);
  return (*this) / other;
}

// reductions
Tensor Tensor::sum() const {
  assert(defined());
  std::shared_ptr<Node> node = nullptr;
  const bool requires_grad = impl_->requires_grad_;
  if (requires_grad) {
    std::vector<std::shared_ptr<TensorImpl>> operand{impl_};
    node = std::make_shared<SumNode>(operand);
  }
  Tensor result({}, apply_accumulate(tensor_ops::SumReducer{}), requires_grad,
                node);
  result.set_parent(node);
  return result;
}

Tensor Tensor::mean() const {
  assert(defined());
  std::shared_ptr<Node> node = nullptr;
  const bool requires_grad = impl_->requires_grad_;
  if (requires_grad) {
    std::vector<std::shared_ptr<TensorImpl>> operand{impl_};
    node = std::make_shared<MeanNode>(operand);
  }
  Tensor result({}, apply_accumulate(tensor_ops::MeanReducer{}), requires_grad,
                node);
  result.set_parent(node);
  return result;
}

Tensor Tensor::max() const {
  assert(defined());
  std::shared_ptr<Node> node = nullptr;
  const bool requires_grad = impl_->requires_grad_;
  if (requires_grad) {
    std::vector<std::shared_ptr<TensorImpl>> operand{impl_};
    node = std::make_shared<MaxNode>(operand);
  }
  Tensor result({}, apply_accumulate(tensor_ops::MaxReducer{}), requires_grad,
                node);
  result.set_parent(node);
  return result;
}

Tensor Tensor::min() const {
  assert(defined());
  std::shared_ptr<Node> node = nullptr;
  const bool requires_grad = impl_->requires_grad_;
  if (requires_grad) {
    std::vector<std::shared_ptr<TensorImpl>> operand{impl_};
    node = std::make_shared<MinNode>(operand);
  }
  Tensor result({}, apply_accumulate(tensor_ops::MinReducer{}), requires_grad,
                node);
  result.set_parent(node);
  return result;
}

// unary ops
Tensor Tensor::relu() const {
  assert(defined());
  std::shared_ptr<Node> node = nullptr;
  const bool requires_grad = impl_->requires_grad_;
  if (requires_grad) {
    std::vector<std::shared_ptr<TensorImpl>> operand{impl_};
    node = std::make_shared<ReluNode>(operand);
  }
  return apply_unary([](float x) { return std::max(0.0f, x); }, node);
}

Tensor Tensor::sigmoid() const {
  assert(defined());
  std::shared_ptr<Node> node = nullptr;
  const bool requires_grad = impl_->requires_grad_;
  if (requires_grad) {
    std::vector<std::shared_ptr<TensorImpl>> operand{impl_};
    node = std::make_shared<SigmaNode>(operand);
  }
  return apply_unary([](float x) { return 1.0f / (1.0f + std::exp(-x)); },
                     node);
}

Tensor Tensor::exp() const {
  assert(defined());
  std::shared_ptr<Node> node = nullptr;
  const bool requires_grad = impl_->requires_grad_;
  if (requires_grad) {
    std::vector<std::shared_ptr<TensorImpl>> operand{impl_};
    node = std::make_shared<ExpNode>(operand);
  }
  return apply_unary([](float x) { return std::exp(x); }, node);
}

Tensor Tensor::log() const {
  assert(defined());
  std::shared_ptr<Node> node = nullptr;
  const bool requires_grad = impl_->requires_grad_;
  if (requires_grad) {
    std::vector<std::shared_ptr<TensorImpl>> operand{impl_};
    node = std::make_shared<LogNode>(operand);
  }
  return apply_unary([](float x) { return std::log(x); }, node);
}

Tensor Tensor::sqrt() const {
  assert(defined());
  std::shared_ptr<Node> node = nullptr;
  const bool requires_grad = impl_->requires_grad_;
  if (requires_grad) {
    std::vector<std::shared_ptr<TensorImpl>> operand{impl_};
    node = std::make_shared<SqrtNode>(operand);
  }
  return apply_unary([](float x) { return std::sqrt(x); }, node);
}

Tensor Tensor::abs() const {
  assert(defined());
  std::shared_ptr<Node> node = nullptr;
  const bool requires_grad = impl_->requires_grad_;
  if (requires_grad) {
    std::vector<std::shared_ptr<TensorImpl>> operand{impl_};
    node = std::make_shared<AbsNode>(operand);
  }
  return apply_unary([](float x) { return std::abs(x); }, node);
}

// matrix multiplication
Tensor Tensor::matmul(const Tensor& other) const {
  assert(defined());
  auto shape = impl_->shape_;
  assert(shape.size() == 2);
  assert(other.shape().size() == 2);

  std::shared_ptr<Node> node = nullptr;
  bool requires_grad = false;
  if ((*this).requires_grad() || other.requires_grad()) {
    std::vector<std::shared_ptr<TensorImpl>> operands{(*this).impl_,
                                                      other.impl_};
    node = std::make_shared<MatmulNode>(operands);
    requires_grad = true;
  }
  std::size_t t_rows = shape[0];
  std::size_t t_columns = shape[1];

  std::size_t o_rows = other.shape()[0];
  std::size_t o_columns = other.shape()[1];

  assert(t_columns == o_rows);

  std::vector<std::size_t> res_shape = {t_rows, o_columns};

  Tensor result(res_shape, 0.0f, requires_grad, node);
  result.set_parent(node);

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
  assert(defined());
  auto storage = impl_->storage_;
  auto shape = impl_->shape_;
  auto strides = impl_->strides_;
  auto offset = impl_->offset_;
  auto requires_grad = impl_->requires_grad_;
  std::shared_ptr<Node> node = nullptr;

  if (requires_grad) {
    std::vector<std::shared_ptr<TensorImpl>> operand{impl_};
    node = std::make_shared<TransNode>(operand);
  }

  std::reverse(shape.begin(), shape.end());
  std::reverse(strides.begin(), strides.end());
  auto impl = std::make_shared<TensorImpl>(storage, shape, strides, offset,
                                           requires_grad, node);
  Tensor result(impl);
  result.set_parent(node);
  return result;
}

// slice
Tensor Tensor::slice(
    std::vector<std::pair<std::size_t, std::size_t>> slices) const {
  assert(defined());
  auto storage = impl_->storage_;
  auto shape = impl_->shape_;
  auto strides = impl_->strides_;
  auto offset = impl_->offset_;
  auto requires_grad = impl_->requires_grad_;
  std::shared_ptr<Node> node = nullptr;

  if (requires_grad) {
    std::vector<std::shared_ptr<TensorImpl>> operand{impl_};
    node = std::make_shared<SliceNode>(operand);
  }

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
  auto impl = std::make_shared<TensorImpl>(storage, shape, strides, offset,
                                           requires_grad, node);
  Tensor result(impl);
  result.set_parent(node);
  return result;
}

Tensor Tensor::copy() const {
  assert(defined());
  std::shared_ptr<Node> node = nullptr;
  const bool requires_grad = impl_->requires_grad_;
  if (requires_grad) {
    std::vector<std::shared_ptr<TensorImpl>> operand{impl_};
    node = std::make_shared<CopyNode>(operand);
  }
  return apply_unary([](float x) { return x; }, node);
}

std::string Tensor::show(std::size_t max_k, std::size_t precision) const {
  assert(defined());
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

std::string Tensor::info() const {
  assert(defined());
  auto shape = impl_->shape_;
  auto strides = impl_->strides_;
  auto offset = impl_->offset_;
  auto storage = impl_->storage_;
  auto requires_grad = impl_->requires_grad_;
  std::stringstream c;
  c << "Tensor with implementation at " << impl_ << " with shape: (";
  for (std::size_t i = 0; i < shape.size(); i++) {
    c << shape[i];
    if (i < shape.size() - 1) {
      c << ", ";
    }
  }
  c << "),\nstrides: (";
  for (std::size_t i = 0; i < strides.size(); i++) {
    c << strides[i];
    if (i < strides.size() - 1) {
      c << ", ";
    }
  }
  c << "),\noffset: " << offset;
  c << ",\n" << ((requires_grad) ? "using gradients" : "not using gradients");
  c << ",\n" << (((*this).leaf()) ? "not having children" : "having children");
  c << ",\npointing to storage at: " << storage << ".";
  return c.str();
}

std::string Tensor::child_info() const {
  return (((*this).leaf() ? "This tensor does not have a child node."
                          : impl_->child_node_->info()));
}

std::string Tensor::tree() const {
  assert(defined());
  std::stringstream c;
  c << show();
  if (!leaf()) {
    auto child = child_node();
    c << "\n\u2191\n";
    c << child->name();
    c << "{\n";
    auto grandchildren = child->children();
    for (std::size_t i = 0; i < grandchildren.size(); ++i) {
      auto grandchild = Tensor(grandchildren[i]);
      c << grandchild.tree();
      if (i < grandchildren.size() - 1) {
        c << "\n,\n";
      }
    }
    c << "\n}";
  }
  return c.str();
}

// private
void Tensor::set_parent(const std::shared_ptr<Node>& node) const {
  if (node != nullptr) {
    node->set_parent(impl_);
  }
}

std::vector<std::size_t> Tensor::broadcast_shape(
    const std::vector<std::size_t>& other_shape) const {
  assert(defined());
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
  assert(defined());
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

void Tensor::parse_string(std::string& s, std::size_t& cur, std::size_t depth,
                          std::vector<std::size_t>& shape,
                          std::vector<float>& data) {
  while (s[cur] == ' ' || s[cur] == '\n') {
    ++cur;
  }
  if (cur >= s.length()) {
    std::stringstream error_msg;
    error_msg << "Unexpected end of string at pos: " << cur << ".";
    throw std::runtime_error(error_msg.str());
  }
  if (s[cur] == '[') {
    ++cur;
    std::size_t count = 0;
    while (true) {
      while (s[cur] == ' ' || s[cur] == '\n') {
        ++cur;
      }
      if (cur >= s.length()) {
        std::stringstream error_msg;
        error_msg << "Unexpected end of string at pos: " << cur << ".";
        throw std::runtime_error(error_msg.str());
      }
      if (s[cur] == ']') {
        ++cur;
        break;
      }
      parse_string(s, cur, depth + 1, shape, data);
      ++count;
      while (s[cur] == ' ' || s[cur] == '\n') {
        ++cur;
      }
      if (cur >= s.length()) {
        std::stringstream error_msg;
        error_msg << "Unexpected end of string at pos: " << cur << ".";
        throw std::runtime_error(error_msg.str());
      }
      auto done = false;
      switch (s[cur]) {
        case ']':
          ++cur;
          done = true;
          break;
        case ',':
          ++cur;
          break;
        default:
          std::stringstream error_msg;
          error_msg << "Unexpected character: '" << s[cur]
                    << "' at pos: " << cur << ".";
          throw std::runtime_error(error_msg.str());
          break;
      }
      if (done) {
        break;
      }
    }
    if (shape.size() <= depth) {
      shape.resize(depth + 1, 0);
    }
    if (shape[depth] == 0) {
      shape[depth] = count;
    } else if (shape[depth] != count) {
      std::stringstream error_msg;
      error_msg << "Shape mismatch: " << count
                << " did not match the previously initialized shape of "
                << shape[depth] << " at depth " << depth << " and pos " << cur
                << ".";
      throw std::runtime_error(error_msg.str());
    }
  } else if (s[cur] == '-' ||
             std::isdigit(static_cast<unsigned char>(s[cur]))) {
    std::size_t chars_read;
    std::string sub = s.substr(cur);
    float cur_data = std::stof(sub, &chars_read);
    cur += chars_read;
    data.push_back(cur_data);
  } else {
    std::stringstream error_msg;
    error_msg << "Unexpected character: '" << s[cur] << "' at pos: " << cur
              << ".";
    throw std::runtime_error(error_msg.str());
  }
  return;
}

void Tensor::parse_constructor_tree(TensorConstructorNode& cur,
                                    std::size_t depth,
                                    std::vector<std::size_t>& shape,
                                    std::vector<float>& data) {
  if (!cur.leaf_) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < cur.children_.size(); ++i) {
      parse_constructor_tree(cur.children_[i], depth + 1, shape, data);
      ++count;
    }
    if (shape.size() <= depth) {
      shape.resize(depth + 1, -1);
    }
    if (shape[depth] == -1) {
      shape[depth] = count;
    } else if (shape[depth] != count) {
      std::stringstream error_msg;
      error_msg << "Shape mismatch: " << count
                << " did not match the previously initialized shape of "
                << shape[depth] << " at depth " << depth << ".";
      throw std::runtime_error(error_msg.str());
    }
  } else {
    data.push_back(cur.value_);
  }
  return;
}
