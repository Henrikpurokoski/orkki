#include <random>
#include <vector>

// tensor class header
class Tensor {
 public:
  // two constructors: initiate with zeros and with random
  static Tensor Random(const std::vector<int>& shape);
  static Tensor Zeros(const std::vector<int>& shape);

  // get shape and data
  std::vector<int> get_shape();
  std::vector<float> get_data();

  // get index
  int operator()(int row, int column);

 private:
  Tensor(const std::vector<int>& shape);
  std::vector<float> data;
  std::vector<int> shape;
};