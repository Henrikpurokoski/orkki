#include <iostream>

#include "include/tensor.hpp"

int main() {
  Tensor a = Tensor::Ones({2, 3, 4}, false);
  std::cout << "\nTensor A:\n" << a.show() << std::endl;

  Tensor b = Tensor::String("[[[1]], [[2], 3]]");
  std::cout << "\nTensor B:\n" << b.show() << std::endl;
}