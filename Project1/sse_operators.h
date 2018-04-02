#pragma once

#include <tmmintrin.h>

#include "sse_shuffle.h"

#ifdef _MSC_VER
#define FORCEINLINE __forceinline
#else
#define FORCEINLINE inline __attribute__((always_inline))
#endif

FORCEINLINE static __m128 operator-(const __m128& data, const __m128& other)
{
    return _mm_sub_ps(data, other);
}
FORCEINLINE static __m128 operator+(const __m128& data, const __m128& other)
{
    return _mm_add_ps(data, other);
}
FORCEINLINE static __m128 operator/(const __m128& data, const __m128& other)
{
    return _mm_div_ps(data, other);
}
FORCEINLINE static __m128 operator*(const __m128& data, const __m128& other)
{
    return _mm_mul_ps(data, other);
}