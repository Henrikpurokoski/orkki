#include <iostream>

#include "include/tensor.hpp"

int main() {
  Tensor a = {{0, 0}, {1, 0}};
  Tensor b = {{2, 3}, {6, 5}};
  a.set_grad(true);
  b.set_grad(true);
  Tensor c = a + b;
  Tensor d = c * a;
  Tensor e = c - d;
  e.backward();
}