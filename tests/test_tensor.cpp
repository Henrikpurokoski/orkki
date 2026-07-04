#include <doctest/doctest.h>

#include "../include/tensor.hpp"

TEST_CASE("zeros constructor") {
  Tensor t = Tensor::Zeros({2, 4});
  CHECK(t.size() == 8);
  CHECK(t.shape() == std::vector<std::size_t>{2, 4});
  CHECK(t.strides() == std::vector<std::size_t>{4, 1});
  for (std::size_t i = 0; i < t.size(); i++) {
    CHECK(t.data()[i] == 0.0f);
  }
}
TEST_CASE("random constructor") {
  Tensor t = Tensor::Random({2, 4}, -1.0f, 2.0f);
  CHECK(t.size() == 8);
  CHECK(t.shape() == std::vector<std::size_t>{2, 4});
  CHECK(t.strides() == std::vector<std::size_t>{4, 1});
  for (std::size_t i = 0; i < t.size(); i++) {
    CHECK(t.data()[i] > -1.0f);
    CHECK(t.data()[i] < 2.0f);
  }
}

TEST_CASE("indexing") {
  Tensor t = Tensor::Zeros({2, 2});

  float value = 0.0f;
  for (std::size_t i = 0; i < 2; i++) {
    for (std::size_t j = 0; j < 2; j++) {
      t(i, j) = value++;
    }
  }
  CHECK(t(0, 0) == 0);
  CHECK(t(0, 1) == 1);
  CHECK(t(1, 0) == 2);
  CHECK(t(1, 1) == 3);
}