#include <iostream>

#include "include/tensor.hpp"

int main() {
  Tensor a = Tensor::Eye({3, 3}, false);

  std::cout << "\nTensor A:\n" << a.show() << std::endl;
}