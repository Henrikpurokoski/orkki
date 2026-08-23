#include <iostream>

#include "include/tensor.hpp"

int main() {
  Tensor a = Tensor::Ones({2, 3, 4}, false);
  std::cout << "\nTensor A:\n" << a.show() << std::endl;

  Tensor b = {{7.6, 3.5}, {2.5, 6.5}};
  std::cout << "\nTensor B:\n" << b.show() << std::endl;
}