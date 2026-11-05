#include <iostream>

#include "include/tensor.hpp"

int main() {
  Tensor a = {{0, 0}, {1, 0}};
  a.set_grad(true);
  std::cout << a.grad().info() << std::endl;
  std::cout << a.grad().show() << std::endl;
  Tensor b = a.slice({{0, 1}, {0, 2}});
  std::cout << b.grad().info() << std::endl;
  std::cout << b.grad().show() << std::endl;
}