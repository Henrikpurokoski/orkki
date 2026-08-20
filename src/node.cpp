#include "../include/node.hpp"

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