#pragma once

#include <tmmintrin.h>
#include <immintrin.h>

#include "core.h"
#include "avx_shuffle.h"

#if defined (ANDROID) || (defined (__linux__) && !defined (__x86_64__))
bool has_avx() { return false; }
#else
    #ifdef _WIN32
    #define cpuid(info, x)    __cpuidex(info, x, 0)
    #else
    #include <cpuid.h>
    void cpuid(int info[4], int info_type) {
        __cpuid_count(info_type, 0, info[0], info[1], info[2], info[3]);
    }
    #endif

    bool has_avx()
    {
        int info[4];
        cpuid(info, 0);
        cpuid(info, 0x80000000);
        return (info[2] & ((int)1 << 28)) != 0;
    }
#endif

namespace simd
{
    template<>
    struct fallback_engine<SUPERSPEED> { static const engine_type FT = DEFAULT; };

    template<>
    struct engine<SUPERSPEED>
    {
        static bool can_run()
        {
            return has_avx();
        }

        template<typename T, typename Dummy = int>
        struct native_simd {};

        template<typename Dummy>
        struct native_simd<float, Dummy>
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
        struct gather_utils {};

        template<unsigned int START, unsigned int GAP>
        struct gather_utils<float, START, GAP>
        {
            template<class GT, class QT, unsigned int J>
            static void do_gather(const QT& res, GT& result)
            {
                const auto shuf = avx::gather_shuffle<GAP, START, J>::shuffle();
                const auto mask = avx::gather_shuffle<GAP, START, J>::mask();

                auto s1 = res.fetch(J);

                auto res1 = _mm256_permutevar8x32_ps(s1, shuf);
                auto res1i = _mm256_castps_si256(res1);

                res1i = _mm256_and_si256(res1i, mask);
                res1 = _mm256_castsi256_ps(res1i);

                auto so_far = result.fetch(0);
                result.assign(0, _mm256_or_ps(res1, so_far));
            }

            template<class GT, class QT, unsigned int J>
            struct gather_loop
            {
                static void gather(const QT& res, GT& result)
                {
                    do_gather<GT, QT, J - 1>(res, result);
                    gather_loop<GT, QT, J - 1>::gather(res, result);
                }
            };
            template<class GT, class QT>
            struct gather_loop<GT, QT, 0>
            {
                static void gather(const QT& res, GT& result) {}
            };

            template<class GT, class QT>
            static void gather(const QT& res, GT& result)
            {
                // Go over every block of QT
                // Do gather on it
                // Merge everything into block GT[I]
                gather_loop<GT, QT, QT::blocks>::gather(res, result);
            }
        };

        template<class T, unsigned int START, unsigned int GAP>
        struct scatter_utils {};

        template<unsigned int START, unsigned int GAP>
        struct scatter_utils<float, START, GAP>
        {
            template<class OT, class ST, unsigned int LINE>
            static void do_scatter(OT& output_block, const ST& curr_var)
            {
                const auto gap = GAP;
                const auto start = START;
                const auto line = LINE;

                const auto shuf = avx::scatter_shuffle<GAP, START, LINE>::shuffle();
                const auto mask = avx::scatter_shuffle<GAP, START, LINE>::mask();

                // Take the value from curr_var[0],
                // (being var with offset START and GAP)
                // and scatter it over output_block[LINE]

                auto s1 = curr_var.fetch(0);
                auto res1 = _mm256_permutevar8x32_ps(s1, shuf);

                auto res1i = _mm256_castps_si256(res1);
                res1i = _mm256_and_si256(res1i, mask);
                res1 = _mm256_castsi256_ps(res1i);

                auto so_far = output_block.fetch(LINE);
                output_block.assign(LINE, _mm256_or_ps(res1, so_far));
            }

            template<class OT, class ST, unsigned int J>
            struct scatter_loop
            {
                static void scatter(OT& output_block, const ST& curr_var)
                {
                    do_scatter<OT, ST, J - 1>(output_block, curr_var);
                    scatter_loop<OT, ST, J - 1>::scatter(output_block, curr_var);
                }
            };
            template<class OT, class ST>
            struct scatter_loop<OT, ST, 0>
            {
                static void scatter(OT& output_block, const ST& curr_var) {}
            };

            template<class OT, class ST>
            static void scatter(OT& output_block, const ST& curr_var)
            {
                scatter_loop<OT, ST, OT::blocks>::scatter(output_block, curr_var);
            }
        };
    };
}