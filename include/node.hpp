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

  std::string info_helper(const std::string& node_type) const;

  virtual std::string info() const = 0;
  virtual std::string name() const = 0;

  const std::vector<std::shared_ptr<TensorImpl>>& children() {
    return children_;
  }

  const std::shared_ptr<TensorImpl> parent() { return parent_.lock(); }

  void set_parent(const std::weak_ptr<TensorImpl>& parent) { parent_ = parent; }

 protected:
  std::vector<std::shared_ptr<TensorImpl>> children_;
  std::weak_ptr<TensorImpl> parent_;
};

// basic arithmetic
class AddNode : public Node {
 public:
  AddNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "AddNode"; }
  std::string info() const override { return info_helper(name()); }
};

class SubNode : public Node {
 public:
  SubNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "SubNode"; }
  std::string info() const override { return info_helper(name()); }
};

class MulNode : public Node {
 public:
  MulNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "MulNode"; }
  std::string info() const override { return info_helper(name()); }
};

class DivNode : public Node {
 public:
  DivNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "DivNode"; }
  std::string info() const override { return info_helper(name()); }
};

// reductions
class SumNode : public Node {
 public:
  SumNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "SumNode"; }
  std::string info() const override { return info_helper(name()); }
};

class MeanNode : public Node {
 public:
  MeanNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "MeanNode"; }
  std::string info() const override { return info_helper(name()); }
};

class MaxNode : public Node {
 public:
  MaxNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "MaxNode"; }
  std::string info() const override { return info_helper(name()); }
};

class MinNode : public Node {
 public:
  MinNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "MinNode"; }
  std::string info() const override { return info_helper(name()); }
};

// unary stuff

class ReluNode : public Node {
 public:
  ReluNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "ReluNode"; }
  std::string info() const override { return info_helper(name()); }
};

class SigmaNode : public Node {
 public:
  SigmaNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "SigmaNode"; }
  std::string info() const override { return info_helper(name()); }
};

class ExpNode : public Node {
 public:
  ExpNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "ExpNode"; }
  std::string info() const override { return info_helper(name()); }
};

class LogNode : public Node {
 public:
  LogNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "LogNode"; }
  std::string info() const override { return info_helper(name()); }
};

class SqrtNode : public Node {
 public:
  SqrtNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "SqrtNode"; }
  std::string info() const override { return info_helper(name()); }
};

class AbsNode : public Node {
 public:
  AbsNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "AbsNode"; }
  std::string info() const override { return info_helper(name()); }
};

class CopyNode : public Node {
 public:
  CopyNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "CopyNode"; }
  std::string info() const override { return info_helper(name()); }
};

// transformations

class TransNode : public Node {
 public:
  TransNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "TransNode"; }
  std::string info() const override { return info_helper(name()); }
};

class SliceNode : public Node {
 public:
  SliceNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "SliceNode"; }
  std::string info() const override { return info_helper(name()); }
};

// matmul :D

class MatmulNode : public Node {
 public:
  MatmulNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string name() const override { return "MatmulNode"; }
  std::string info() const override { return info_helper(name()); }
};
