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
  Tensor c = a + b;
  std::cout << "\nTensor C = A + B:\n"
            << c.info() << "\nLooks like:\n"
            << c.show() << std::endl;
  Tensor d = c.sum();
  std::cout << "\nTensor D = sum of C:\n"
            << d.info() << "\nLooks like:\n"
            << d.show() << std::endl;
  Tensor e = c.transpose();
  std::cout << "\nTensor E = transpose of C:\n"
            << e.info() << "\nLooks like:\n"
            << e.show() << std::endl;
  Tensor f = c.slice({{1, 3}, {1, 3}});
  std::cout << "\nTensor F = slice of C:\n"
            << f.info() << "\nLooks like:\n"
            << f.show() << std::endl;
  std::cout << "is this defined? " << f.defined()
            << " What about the gradients? (should not be) "
            << f.grad().defined() << std::endl;
}