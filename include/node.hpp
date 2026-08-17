#pragma once

#include <cassert>
#include <memory>
#include <sstream>
#include <vector>

class TensorImpl;

class Node {
 public:
  explicit Node(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : children_(children) {}

  virtual ~Node() = default;

  virtual std::string info() const = 0;

  const std::vector<std::shared_ptr<TensorImpl>>& children() {
    return children_;
  }

  const std::shared_ptr<TensorImpl> parent() { return parent_.lock(); }

  void set_parent(const std::weak_ptr<TensorImpl>& parent) { parent_ = parent; }

 protected:
  std::vector<std::shared_ptr<TensorImpl>> children_;
  std::weak_ptr<TensorImpl> parent_;
};

class AddNode : public Node {
 public:
  AddNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override {
    std::stringstream c;
    c << "This an AddNode.\nParent:\n";
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
};

class SubNode : public Node {
 public:
  SubNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override {
    std::stringstream c;
    c << "This a SubNode.\nParent:\n";
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
};

class MulNode : public Node {
 public:
  MulNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override {
    std::stringstream c;
    c << "This a MulNode.\nParent:\n";
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
};

class DivNode : public Node {
 public:
  DivNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override {
    std::stringstream c;
    c << "This a DivNode.\nParent:\n";
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
};