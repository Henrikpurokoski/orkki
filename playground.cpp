#include <iostream>

#include "include/tensor.hpp"

int main() {
  Tensor a = Tensor::Random({3, 3}, 0.0f, 1.0f, false);
  std::cout << "\nTensor A:\n"
            << a.info() << "\nLooks like:\n"
            << a.show() << std::endl;

  Tensor b = Tensor::Random({3, 3}, 0.0f, 1.0f, true);
  std::cout << "\nTensor B:\n"
            << b.info() << "\nLooks like:\n"
            << b.show() << std::endl;
  std::cout << "this works" << std::endl;
  Tensor c = a.matmul(b);
  std::cout << "this works2" << std::endl;
  std::cout << "\nTensor C = a matmul b:\n"
            << c.info() << "\nLooks like:\n"
            << c.show() << "\n\nIs the parent of a node:\n"
            << c.child_info() << std::endl;
}