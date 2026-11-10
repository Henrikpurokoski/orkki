#include <iostream>

#include "include/tensor.hpp"

int main() {
  Tensor a = {{{2, 4}, {1, 3}}, {{5, 6}, {9, 8}}};
  std::cout << a.show() << std::endl;
  std::cout << a.info() << std::endl;

  Tensor b = a.reduce({2, 1, 2});

  std::cout << b.show() << std::endl;
  std::cout << b.info() << std::endl;
}