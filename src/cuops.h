#ifndef __JHU_CUOPS_H__
#define __JHU_CUOPS_H__

#include <math_constants.h>

// Numerically stable way of computing f(a) = log(1 + exp(a))
inline float log1pexp(float a) {
  return log1pf(exp(a));
}

inline float log_add(float a, float b) {
  if (a == -CUDART_INF_F) return b;
  if (b == -CUDART_INF_F) return a;
  return a>b? a+log1pexp(b-a):  b+log1pexp(a-b);
}

inline void log_plus_equals(float *l1, float l2) {
  *l1 = log_add(*l1,l2);
}

#endif
