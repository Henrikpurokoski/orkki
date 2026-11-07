#include <iostream>

#include "include/tensor.hpp"

int main() {
  Tensor a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  a.set_grad(true);
  Tensor b = a.transpose();
  Tensor c = b.slice({{0, 2}, {1, 3}});
  Tensor d = c.view();
  Tensor e = d.copy();

  std::cout << "before backprob:" << std::endl;

  std::cout << "\n\nthis is A:\n\n"
            << a.info() << "\n"
            << a.show() << "\n\n this is A's gradinet now \n\n"
            << a.grad().info() << "\n"
            << a.grad().show() << "\n\nthis is B:\n\n"
            << b.info() << "\n"
            << b.show() << "\n\n this is B's gradinet now \n\n"
            << b.grad().info() << "\n"
            << b.grad().show() << "\n\nthis is C:\n\n"
            << c.info() << "\n"
            << c.show() << "\n\n this is C's gradinet now \n\n"
            << c.grad().info() << "\n"
            << c.grad().show() << "\n\nthis is D:\n\n"
            << d.info() << "\n"
            << d.show() << "\n\n this is D's gradinet now \n\n"
            << d.grad().info() << "\n"
            << d.grad().show() << "\n\nthis is E:\n\n"
            << e.info() << "\n"
            << e.show() << "\n\n this is E's gradinet now \n\n"
            << e.grad().info() << "\n"
            << e.grad().show() << std::endl;

  std::cout << "tree:\n\n" << e.tree() << std::endl;

  e.backward();

  std::cout << "after backprob:" << std::endl;

  std::cout << "\n\nthis is A:\n\n"
            << a.info() << "\n"
            << a.show() << "\n\n this is A's gradinet now \n\n"
            << a.grad().info() << "\n"
            << a.grad().show() << "\n\nthis is B:\n\n"
            << b.info() << "\n"
            << b.show() << "\n\n this is B's gradinet now \n\n"
            << b.grad().info() << "\n"
            << b.grad().show() << "\n\nthis is C:\n\n"
            << c.info() << "\n"
            << c.show() << "\n\n this is C's gradinet now \n\n"
            << c.grad().info() << "\n"
            << c.grad().show() << "\n\nthis is D:\n\n"
            << d.info() << "\n"
            << d.show() << "\n\n this is D's gradinet now \n\n"
            << d.grad().info() << "\n"
            << d.grad().show() << "\n\nthis is E:\n\n"
            << e.info() << "\n"
            << e.show() << "\n\n this is E's gradinet now \n\n"
            << e.grad().info() << "\n"
            << e.grad().show() << std::endl;
}