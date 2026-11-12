#include <iostream>

#include "include/tensor.hpp"

int main() {
  Tensor a = {{1, 2}};
  a.set_grad(true);
  Tensor b = {{2, 3}, {4, 5}};
  b.set_grad(true);
  Tensor c = b / a;

  c.backward();
  std::cout << c.tree(true) << std::endl;
}