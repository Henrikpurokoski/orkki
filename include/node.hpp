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

  std::string info() const override { return info_helper("AddNode"); }
};

class SubNode : public Node {
 public:
  SubNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("SubNode"); }
};

class MulNode : public Node {
 public:
  MulNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("MulNode"); }
};

class DivNode : public Node {
 public:
  DivNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("DivNode"); }
};

// reductions
class SumNode : public Node {
 public:
  SumNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("SumNode"); }
};

class MeanNode : public Node {
 public:
  MeanNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("MeanNode"); }
};

class MaxNode : public Node {
 public:
  MaxNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("MaxNode"); }
};

class MinNode : public Node {
 public:
  MinNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("MinNode"); }
};

// unary stuff

class ReluNode : public Node {
 public:
  ReluNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("ReluNode"); }
};

class SigmaNode : public Node {
 public:
  SigmaNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("SigmaNode"); }
};

class ExpNode : public Node {
 public:
  ExpNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("ExpNode"); }
};

class LogNode : public Node {
 public:
  LogNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("LogNode"); }
};

class SqrtNode : public Node {
 public:
  SqrtNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("SqrtNode"); }
};

class AbsNode : public Node {
 public:
  AbsNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("AbsNode"); }
};

class CopyNode : public Node {
 public:
  CopyNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("CopyNode"); }
};

// transformations

class TransNode : public Node {
 public:
  TransNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("TransNode"); }
};

class SliceNode : public Node {
 public:
  SliceNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("SliceNode"); }
};

// matmul :D

class MatmulNode : public Node {
 public:
  MatmulNode(const std::vector<std::shared_ptr<TensorImpl>>& children)
      : Node(children) {}

  std::string info() const override { return info_helper("MatmulNode"); }
};
