#include <iostream>

#include "include/tensor.hpp"

int main() {
  Tensor a = {{0, 0}, {1, 0}};
  std::cout << a.show() << "\n" << std::endl;
  std::cout << a.transpose().slice({{0, 2}, {1, 2}}).show() << std::endl;

  a += a.transpose().slice({{0, 2}, {1, 2}});

  std::cout << "\nTensor A:\n" << a.show() << std::endl;
}