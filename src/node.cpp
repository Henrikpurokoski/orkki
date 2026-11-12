#include "../include/node.hpp"

#include "../include/tensor.hpp"

std::string Node::info_helper(const std::string& node_type) const {
  std::stringstream c;
  c << "This a " << node_type << ".\nParent:\n";
  if (auto parent = parent_.lock()) {
    c << parent << ".\n";
  } else {
    c << "an object that no longer exists.\n";
  }
  c << "Children:\n";

  for (std::size_t i = 0; i < children_.size(); ++i) {
    c << children_[i];
    if (i < children_.size() - 1) {
      c << "\n";
    }
  }
  c << ".";
  return c.str();
}

const std::vector<std::shared_ptr<const Node>> Node::topo_sort() const {
  std::vector<std::shared_ptr<const Node>> sorted;
  std::unordered_set<const Node*> visited;

  depth_first(sorted, visited);

  return sorted;
}

void Node::depth_first(std::vector<std::shared_ptr<const Node>>& sorted,
                       std::unordered_set<const Node*>& visited) const {
  if (!visited.count(this)) {
    visited.insert(this);
    for (std::size_t i = 0; i < children_.size(); ++i) {
      if (children_[i]->child_node_ != nullptr) {
        children_[i]->child_node_->depth_first(sorted, visited);
      }
    }
    sorted.push_back(shared_from_this());
  }
  return;
}

void Node::backward() const { return; }

void AddNode::backward() const {
  if (children_.size() != 2) {
    throw std::runtime_error("wrong number of children in Add!");
  }
  std::shared_ptr<TensorImpl> init_grad_impl = nullptr;
  if (auto strong = parent_.lock()) {
    init_grad_impl = strong->grad_;
  } else {
    throw std::runtime_error("what no parent :(");
  }
  auto init_grad = Tensor::From_impl(init_grad_impl);
  for (std::size_t i = 0; i < children_.size(); ++i) {
    auto grad_shape = children_[i]->shape_;
    Tensor grad = init_grad.reduce(grad_shape);
    children_[i]->backward(grad.impl());
  }
}

void SubNode::backward() const {
  if (children_.size() != 2) {
    throw std::runtime_error("wrong number of children in Sub!");
  }
  std::shared_ptr<TensorImpl> init_grad_impl = nullptr;
  if (auto strong = parent_.lock()) {
    init_grad_impl = strong->grad_;
  } else {
    throw std::runtime_error("what no parent :(");
  }
  auto init_grad = Tensor::From_impl(init_grad_impl);
  for (std::size_t i = 0; i < children_.size(); ++i) {
    auto grad_shape = children_[i]->shape_;
    Tensor grad = (i == 0) ? init_grad.reduce(grad_shape)
                           : (init_grad * -1).reduce(grad_shape);
    children_[i]->backward(grad.impl());
  }
}

void MulNode::backward() const {
  if (children_.size() != 2) {
    throw std::runtime_error("wrong number of children in Mul!");
  }
  std::shared_ptr<TensorImpl> init_grad_impl = nullptr;
  if (auto strong = parent_.lock()) {
    init_grad_impl = strong->grad_;
  } else {
    throw std::runtime_error("what no parent :(");
  }
  auto init_grad = Tensor::From_impl(init_grad_impl);
  auto other_idx = children_.size() - 1;
  for (std::size_t i = 0; i < children_.size(); ++i) {
    auto other = Tensor::From_impl(children_[other_idx]);
    other.infect(false);
    auto grad_shape = children_[i]->shape_;

    Tensor grad = (init_grad * other).reduce(grad_shape);
    children_[i]->backward(grad.impl());
    other.infect(true);
    other_idx -= 1;
  }
}

void DivNode::backward() const {
  if (children_.size() != 2) {
    throw std::runtime_error("wrong number of children in Div!");
  }
  std::shared_ptr<TensorImpl> init_grad_impl = nullptr;
  if (auto strong = parent_.lock()) {
    init_grad_impl = strong->grad_;
  } else {
    throw std::runtime_error("what no parent :(");
  }
  auto init_grad = Tensor::From_impl(init_grad_impl);
  auto other_idx = children_.size() - 1;
  for (std::size_t i = 0; i < children_.size(); ++i) {
    auto other = Tensor::From_impl(children_[other_idx]);
    auto thiss = Tensor::From_impl(children_[i]);
    other.infect(false);
    thiss.infect(false);
    auto grad_shape = children_[i]->shape_;

    Tensor grad =
        (i == 0)
            ? (init_grad / other).reduce(grad_shape)
            : (init_grad * (other / (thiss * thiss)) * -1).reduce(grad_shape);
    children_[i]->backward(grad.impl());
    other.infect(true);
    thiss.infect(true);
    other_idx -= 1;
  }
}

// void SumNode::backward() const {}

// void MeanNode::backward() const {}

// void MaxNode::backward() const {}

// void MinNode::backward() const {}

// void ReluNode::backward() const {}

// void SigmaNode::backward() const {}

// void ExpNode::backward() const {}

// void LogNode::backward() const {}

// void SqrtNode::backward() const {}

// void AbsNode::backward() const {}

void CopyNode::backward() const {
  if (children_.size() != 1) {
    throw std::runtime_error("wrong number of children in Copy!");
  }
  std::shared_ptr<TensorImpl> init_grad = nullptr;
  if (auto strong = parent_.lock()) {
    init_grad = strong->grad_;
  } else {
    throw std::runtime_error("what no parent :(");
  }
  children_[0]->backward(init_grad);
}

void ViewNode::backward() const { return; }

void TransNode::backward() const { return; }

void SliceNode::backward() const { return; }

// void MatmulNode::backward() const {}