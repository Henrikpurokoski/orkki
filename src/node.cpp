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

void Node::backward() const {
  for (std::size_t i = 0; i < children_.size(); ++i) {
    Tensor init_grad = Tensor::Zeros(children_[i]->shape_);
    children_[i]->backward(init_grad.impl_);
  }
  return;
}