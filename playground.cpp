#include <iostream>

#include "include/tensor.hpp"

int main() {
  Tensor a = Tensor::Random({3, 3});
  std::cout << "\nTensor A:\n"
            << a.info() << "\nLooks like:\n"
            << a.show() << std::endl;
  Tensor b = a.slice({{1, 3}, {1, 3}});
  std::cout << "\n\nTensor B (slice of A):\n"
            << b.info() << "\nLooks like:\n"
            << b.show() << std::endl;
  Tensor c = b.transpose();
  std::cout << "\n\nTensor C (transpose of B):\n"
            << c.info() << "\nLooks like:\n"
            << c.show() << std::endl;
  Tensor d = c.copy();
  std::cout << "\n\nTensor D (copy of C):\n"
            << d.info() << "\nLooks like:\n"
            << d.show() << std::endl;
}