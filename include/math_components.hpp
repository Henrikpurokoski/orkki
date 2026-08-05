#pragma once

namespace tensor_ops {
struct SumReducer {
  float sum = 0.0f;
  void update(float val) { sum += val; }
  float finalize(float size) { return sum; }
};

struct MeanReducer {
  float sum = 0.0f;
  void update(float val) { sum += val; }
  float finalize(float size) { return (size == 0) ? 0.0f : sum / size; }
};

struct MinReducer {
  float init = false;
  float min = 0;
  void update(float val) {
    if (init == false) {
      min = val;
      init = true;
    }
    min = (val < min) ? val : min;
  }
  float finalize(float size) { return min; }
};

struct MaxReducer {
  float init = false;
  float max = 0;
  void update(float val) {
    if (init == false) {
      max = val;
      init = true;
    }
    max = (val > max) ? val : max;
  }
  float finalize(float size) { return max; }
};
};  // namespace tensor_ops