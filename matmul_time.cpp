#include <chrono>
#include <iostream>

#include "include/tensor.hpp"

using timer = std::chrono::steady_clock;

constexpr int N = 10;

int main() {
  Tensor a = Tensor::Random({512, 512}, -2.0f, 2.0f);
  Tensor b = Tensor::Random({512, 512}, -2.0f, 2.0f);

  double checksum = 0.0;

  auto start = timer::now();

  for (std::size_t i = 0; i < N; ++i) {
    Tensor res = a.matmul(b);
    checksum += res(0, 0);
    std::cout << i + 1 << "/" << N << '\n';
  }

  auto end = timer::now();

  auto total = std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "Average matmul: " << total / N << " ms\n";
  std::cout << "Checksum: " << checksum << '\n';
}