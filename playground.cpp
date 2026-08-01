#include <iostream>

#include "include/tensor.hpp"

int main() {
  Tensor a = Tensor::Zeros({4, 2});
  std::cout << "A Nestori:" << std::endl;
  std::cout << a.show() << std::endl;

  Tensor b = Tensor::Zeros({
      2,
  });

  b(0) = 1;
  b(1) = 12;

  std::cout << "B Nestori:" << std::endl;
  std::cout << b.show() << std::endl;

  Tensor c = a + b;

  std::cout << "C = A + B:" << std::endl;
  std::cout << c.show() << std::endl;

  Tensor d = Tensor::Random({4, 2}, -2.0f, 2.0f);
  std::cout << "D Nestori:" << std::endl;
  std::cout << d.show() << std::endl;

  Tensor e = d.transpose();
  std::cout << "Ja sen transpossu E:" << std::endl;
  std::cout << e.show() << std::endl;

  Tensor f = c.matmul(e);
  std::cout << "seuraavaks F on yhtäkuin C kertaa E:" << std::endl;
  std::cout << f.show() << std::endl;

  Tensor g = d.matmul(e);
  std::cout << "ja viälä G on yhtäkuin D kertaa E:" << std::endl;
  std::cout << g.show() << std::endl;
}