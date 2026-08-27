#include <iostream>

#include "include/tensor.hpp"

int main() {
  Tensor a = Tensor::Ones({3, 3, 3}, false);

  a({0, 1, 0}) = 78;
  std::cout << "\nTensor A:\n" << a.show() << std::endl;
}