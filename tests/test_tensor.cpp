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

TEST_CASE("elementwise plus") {
  Tensor t_a = Tensor::Zeros({2, 2});
  Tensor t_b = Tensor::Zeros({2, 2});

  float value_a = 1.0f;
  float value_b = 0.0f;
  for (std::size_t i = 0; i < 2; i++) {
    for (std::size_t j = 0; j < 2; j++) {
      t_a(i, j) = value_a;
      t_b(i, j) = value_b++;
    }
  }

  Tensor t_c = t_a + t_b;

  CHECK(t_c(0, 0) == 1);
  CHECK(t_c(0, 1) == 2);
  CHECK(t_c(1, 0) == 3);
  CHECK(t_c(1, 1) == 4);
}

TEST_CASE("elementwise minus") {
  Tensor t_a = Tensor::Zeros({2, 2});
  Tensor t_b = Tensor::Zeros({2, 2});

  float value_a = 2.0f;
  float value_b = 0.0f;
  for (std::size_t i = 0; i < 2; i++) {
    for (std::size_t j = 0; j < 2; j++) {
      t_a(i, j) = value_a;
      t_b(i, j) = value_b++;
    }
  }

  Tensor t_c = t_a - t_b;

  CHECK(t_c(0, 0) == 2);
  CHECK(t_c(0, 1) == 1);
  CHECK(t_c(1, 0) == 0);
  CHECK(t_c(1, 1) == -1);
}

TEST_CASE("elementwise multiplies") {
  Tensor t_a = Tensor::Zeros({2, 2});
  Tensor t_b = Tensor::Zeros({2, 2});

  float value_a = 2.0f;
  float value_b = 0.0f;
  for (std::size_t i = 0; i < 2; i++) {
    for (std::size_t j = 0; j < 2; j++) {
      t_a(i, j) = value_a;
      t_b(i, j) = value_b++;
    }
  }

  Tensor t_c = t_a * t_b;

  CHECK(t_c(0, 0) == 0);
  CHECK(t_c(0, 1) == 2);
  CHECK(t_c(1, 0) == 4);
  CHECK(t_c(1, 1) == 6);
}

TEST_CASE("elementwise divides") {
  Tensor t_a = Tensor::Zeros({2, 2});
  Tensor t_b = Tensor::Zeros({2, 2});

  float value_a = 12.0f;
  float value_b = 4.0f;
  for (std::size_t i = 0; i < 2; i++) {
    for (std::size_t j = 0; j < 2; j++) {
      t_a(i, j) = value_a;
      t_b(i, j) = value_b--;
    }
  }

  Tensor t_c = t_a / t_b;

  CHECK(t_c(0, 0) == 3);
  CHECK(t_c(0, 1) == 4);
  CHECK(t_c(1, 0) == 6);
  CHECK(t_c(1, 1) == 12);
}

TEST_CASE("scalar plus") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 1;
  t(0, 1) = 2;

  Tensor r = t + 2;

  CHECK(r(0, 0) == 3);
  CHECK(r(0, 1) == 4);
}

TEST_CASE("scalar minus") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 1;
  t(0, 1) = 2;

  Tensor r = t - 2;

  CHECK(r(0, 0) == -1);
  CHECK(r(0, 1) == 0);
}

TEST_CASE("scalar multiplies") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 1;
  t(0, 1) = 2;

  Tensor r = t * 2;

  CHECK(r(0, 0) == 2);
  CHECK(r(0, 1) == 4);
}

TEST_CASE("scalar divides") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 1;
  t(0, 1) = 2;

  Tensor r = t / 2;

  CHECK(r(0, 0) == 0.5f);
  CHECK(r(0, 1) == 1);
}

TEST_CASE("elementwise compound plus") {
  Tensor t_a = Tensor::Zeros({1, 2});
  Tensor t_b = Tensor::Zeros({1, 2});

  t_a(0, 0) = 1;
  t_a(0, 1) = 2;

  t_b(0, 0) = 2;
  t_b(0, 1) = 3;

  t_a += t_b;

  CHECK(t_a(0, 0) == 3);
  CHECK(t_a(0, 1) == 5);
}

TEST_CASE("elementwise compound minus") {
  Tensor t_a = Tensor::Zeros({1, 2});
  Tensor t_b = Tensor::Zeros({1, 2});

  t_a(0, 0) = 5;
  t_a(0, 1) = 5;

  t_b(0, 0) = 2;
  t_b(0, 1) = 3;

  t_a -= t_b;

  CHECK(t_a(0, 0) == 3);
  CHECK(t_a(0, 1) == 2);
}

TEST_CASE("elementwise compound multiplies") {
  Tensor t_a = Tensor::Zeros({1, 2});
  Tensor t_b = Tensor::Zeros({1, 2});

  t_a(0, 0) = 1;
  t_a(0, 1) = 2;

  t_b(0, 0) = 2;
  t_b(0, 1) = 3;

  t_a *= t_b;

  CHECK(t_a(0, 0) == 2);
  CHECK(t_a(0, 1) == 6);
}

TEST_CASE("elementwise compound divides") {
  Tensor t_a = Tensor::Zeros({1, 2});
  Tensor t_b = Tensor::Zeros({1, 2});

  t_a(0, 0) = 2;
  t_a(0, 1) = 4;

  t_b(0, 0) = 4;
  t_b(0, 1) = 2;

  t_a /= t_b;

  CHECK(t_a(0, 0) == 0.5f);
  CHECK(t_a(0, 1) == 2);
}

TEST_CASE("scalar compound plus") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 1;
  t(0, 1) = 2;

  t += 2;

  CHECK(t(0, 0) == 3);
  CHECK(t(0, 1) == 4);
}

TEST_CASE("scalar compound minus") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 1;
  t(0, 1) = 2;

  t -= 2;

  CHECK(t(0, 0) == -1);
  CHECK(t(0, 1) == 0);
}

TEST_CASE("scalar compound multiplies") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 1;
  t(0, 1) = 2;

  t *= 2;

  CHECK(t(0, 0) == 2);
  CHECK(t(0, 1) == 4);
}

TEST_CASE("scalar compound divides") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 1;
  t(0, 1) = 2;

  t /= 2;

  CHECK(t(0, 0) == 0.5f);
  CHECK(t(0, 1) == 1);
}

TEST_CASE("sum") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 1;
  t(0, 1) = 2;

  float r = t.sum();

  CHECK(r == 3);
}

TEST_CASE("mean") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 1;
  t(0, 1) = 2;

  float r = t.mean();

  CHECK(r == 1.5);
}

TEST_CASE("max") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 1;
  t(0, 1) = 2;

  float r = t.max();

  CHECK(r == 2);
}

TEST_CASE("min") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 1;
  t(0, 1) = 2;

  float r = t.min();

  CHECK(r == 1);
}

TEST_CASE("matmul1") {
  Tensor t_a = Tensor::Zeros({2, 2});
  Tensor t_b = Tensor::Zeros({2, 2});

  t_a(0, 0) = 1;
  t_a(0, 1) = 2;
  t_a(1, 0) = 3;
  t_a(1, 1) = 4;

  t_b(0, 0) = 4;
  t_b(0, 1) = 3;
  t_b(1, 0) = 2;
  t_b(1, 1) = 1;

  Tensor res = t_a.matmul(t_b);

  CHECK(res.shape() == std::vector<std::size_t>{2, 2});

  CHECK(res(0, 0) == 8);
  CHECK(res(0, 1) == 5);
  CHECK(res(1, 0) == 20);
  CHECK(res(1, 1) == 13);
}

TEST_CASE("transpose") {
  Tensor t = Tensor::Zeros({2, 3});

  t(0, 0) = 1;
  t(0, 1) = 2;
  t(0, 2) = 3;
  t(1, 0) = 4;
  t(1, 1) = 5;
  t(1, 2) = 6;

  Tensor res = t.transpose();

  CHECK(res.shape() == std::vector<std::size_t>{3, 2});

  CHECK(res(0, 0) == 1);
  CHECK(res(1, 0) == 2);
  CHECK(res(2, 0) == 3);
  CHECK(res(0, 1) == 4);
  CHECK(res(1, 1) == 5);
  CHECK(res(2, 1) == 6);
}

TEST_CASE("relu") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 1;
  t(0, 1) = -1;

  Tensor res = t.relu();

  CHECK(res(0, 0) == 1);
  CHECK(res(0, 1) == 0);
}

TEST_CASE("sigmoid") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 0;
  t(0, 1) = 1;

  Tensor res = t.sigmoid();

  CHECK(res(0, 0) == 0.5);
  CHECK(res(0, 1) < 1);
}

TEST_CASE("exp") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 0;
  t(0, 1) = 1;

  Tensor res = t.exp();

  CHECK(res(0, 0) == 1);
  CHECK(res(0, 1) > 1);
}

TEST_CASE("log") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 1;
  t(0, 1) = 10;

  Tensor res = t.log();

  CHECK(res(0, 0) == 0);
  CHECK(res(0, 1) < 10);
}

TEST_CASE("sqrt") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 4;
  t(0, 1) = 16;

  Tensor res = t.sqrt();

  CHECK(res(0, 0) == 2);
  CHECK(res(0, 1) == 4);
}

TEST_CASE("abs") {
  Tensor t = Tensor::Zeros({1, 2});

  t(0, 0) = 4;
  t(0, 1) = -4;

  Tensor res = t.abs();

  CHECK(res(0, 0) == 4);
  CHECK(res(0, 1) == 4);
}

TEST_CASE("broadcast1") {
  Tensor t_a = Tensor::Zeros({2, 2});
  Tensor t_b = Tensor::Zeros({2, 1});

  t_a(0, 0) = 1;
  t_a(0, 1) = 2;
  t_a(1, 0) = 3;
  t_a(1, 1) = 4;

  t_b(0, 0) = 5;
  t_b(1, 0) = 6;

  Tensor res = t_a + t_b;

  CHECK(res.shape() == std::vector<std::size_t>{2, 2});

  CHECK(res(0, 0) == 6);
  CHECK(res(0, 1) == 7);
  CHECK(res(1, 0) == 9);
  CHECK(res(1, 1) == 10);
}

TEST_CASE("broadcast2") {
  Tensor t_a = Tensor::Zeros({2, 2, 2});
  Tensor t_b = Tensor::Zeros({1, 2});

  t_b(0, 0) = 1;
  t_b(0, 1) = 2;

  Tensor res = t_a + t_b;

  CHECK(res.shape() == std::vector<std::size_t>{2, 2, 2});

  CHECK(res(0, 0, 0) == 1);
  CHECK(res(0, 0, 1) == 2);
}