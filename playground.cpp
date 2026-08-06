#include <iostream>

#include "include/tensor.hpp"

int main() {
  Tensor a = Tensor::Random({5000, 5000, 3});
  std::cout << "Moi:\n\n" << a.show() << std::endl;
  std::cout << "\n\nHei:\n\n" << a.transpose().show() << std::endl;
}