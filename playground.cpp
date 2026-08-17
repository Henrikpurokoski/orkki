#include <iostream>

#include "include/tensor.hpp"

int main() {
  Tensor a = Tensor::Random({3, 3});
  std::cout << "\nTensor A:\n"
            << a.info() << "\nLooks like:\n"
            << a.show() << std::endl;

  Tensor b = Tensor::Random({3, 1}, 0.0f, 1.0f, true);
  std::cout << "\nTensor B:\n"
            << b.info() << "\nLooks like:\n"
            << b.show() << std::endl;
  Tensor c = a * b;
  std::cout << "\nTensor C = A + B:\n"
            << c.info() << "\nLooks like:\n"
            << c.show() << "\n\nIs the parent of a node:\n"
            << c.child_node()->info() << std::endl;
}