#pragma once

#include <tmmintrin.h>
#include <immintrin.h>

#include "core.h"

namespace simd
{
    template<>
    struct engine<SUPERSPEED>
    {
        template<typename T>
        struct native_simd {};

        template<>
        struct native_simd<float>
        {
        public:
            typedef __m256 underlying_type;
            typedef native_simd<float> representation_type;
            typedef native_simd<float> this_type;

            FORCEINLINE static void load(representation_type& target, const underlying_type* other)
            {
                target._data = _mm256_castsi256_ps(_mm256_load_si256((const __m256i*)other));
            }

            FORCEINLINE static void store(const representation_type& src, underlying_type* target)
            {
                _mm256_storeu_ps((float*)target, src._data);
            }

            FORCEINLINE static underlying_type vectorize(float x)
            {
                return _mm256_set_ps(x, x, x, x, x, x, x, x);
            }

            FORCEINLINE native_simd(underlying_type data) : _data(data) {}
            FORCEINLINE native_simd(const underlying_type* data) : _data(_mm256_castsi256_ps(_mm256_load_si256((const __m256i*)data))) {}
            FORCEINLINE native_simd() : _data(_mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0)) {}
            FORCEINLINE native_simd(const native_simd& data) { _data = data._data; }

            FORCEINLINE native_simd operator+(const native_simd& y) const
            {
                return native_simd(_mm256_add_ps(_data, y._data));
            }
            FORCEINLINE native_simd operator-(const native_simd& y) const
            {
                return native_simd(_mm256_sub_ps(_data, y._data));
            }
            FORCEINLINE native_simd operator/(const native_simd& y) const
            {
                return native_simd(_mm256_div_ps(_data, y._data));
            }
            FORCEINLINE native_simd operator*(const native_simd& y) const
            {
                return native_simd(_mm256_mul_ps(_data, y._data));
            }
            FORCEINLINE void store(underlying_type* ptr) const
            {
                _mm256_storeu_ps((float*)ptr, _data);
            }
            FORCEINLINE operator underlying_type() const { return _data; }

        private:
            underlying_type _data;
        };

        template<class T, unsigned int START, unsigned int GAP>
        using gather_utils = simd::engine<DEFAULT>::gather_utils<T, START, GAP>;

        template<class T, unsigned int START, unsigned int GAP>
        using scatter_utils = simd::engine<DEFAULT>::scatter_utils<T, START, GAP>;
    };
}