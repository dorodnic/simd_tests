#pragma once

#include "core.h"

#include "sse_shuffle.h"
#include "sse_operators.h"

namespace simd
{
    template<>
    struct engine<DEFAULT>
    {
        template<typename T, typename Dummy = int>
        struct native_simd {};

        template<typename Dummy>
        struct native_simd<float, Dummy>
        {
        public:
            typedef __m128 underlying_type;
            typedef native_simd<float> representation_type;
            typedef native_simd<float> this_type;

            FORCEINLINE static void load(representation_type& target, const underlying_type* other)
            {
                target._data = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)other));
            }

            FORCEINLINE static void store(const representation_type& src, underlying_type* target)
            {
                _mm_storeu_ps((float*)target, src._data);
            }

            FORCEINLINE static underlying_type vectorize(float x)
            {
                return _mm_set_ps1(x);
            }

            FORCEINLINE native_simd(underlying_type data) : _data(data) {}
            FORCEINLINE native_simd(const underlying_type* data) : _data(_mm_castsi128_ps(_mm_loadu_si128((const __m128i*)data))) {}
            FORCEINLINE native_simd() : _data(_mm_set_ps1(0)) {}
            FORCEINLINE native_simd(const native_simd& data) { _data = data._data; }

            FORCEINLINE native_simd operator+(const native_simd& y) const
            {
                return native_simd(_mm_add_ps(_data, y._data));
            }
            FORCEINLINE native_simd operator-(const native_simd& y) const
            {
                return native_simd(_mm_sub_ps(_data, y._data));
            }
            FORCEINLINE native_simd operator/(const native_simd& y) const
            {
                return native_simd(_mm_div_ps(_data, y._data));
            }
            FORCEINLINE native_simd operator*(const native_simd& y) const
            {
                return native_simd(_mm_mul_ps(_data, y._data));
            }
            FORCEINLINE void store(underlying_type* ptr) const
            {
                _mm_storeu_ps((float*)ptr, _data);
            }
            FORCEINLINE operator underlying_type() const { return _data; }

        private:
            underlying_type _data;
        };

        static __m128i load_mask(unsigned int x)
        {
            return  _mm_set_epi32(
                (x & 0xff000000) ? 0xffffffff : 0,
                (x & 0x00ff0000) ? 0xffffffff : 0,
                (x & 0x0000ff00) ? 0xffffffff : 0,
                (x & 0x000000ff) ? 0xffffffff : 0);
        }

        template<class T, unsigned int START, unsigned int GAP>
        struct gather_utils {};

        template<unsigned int START, unsigned int GAP>
        struct gather_utils<float, START, GAP>
        {
            template<class GT, class QT, unsigned int J>
            static void do_gather(const QT& res, GT& result)
            {
                const auto shuf = sse::gather_shuffle<GAP, START, J>::shuffle();
                const auto mask = sse::gather_shuffle<GAP, START, J>::mask();

                auto s1 = res.fetch(J);

                auto res1 = _mm_shuffle_ps(s1, s1, shuf);
                auto res1i = _mm_castps_si128(res1);

                res1i = _mm_and_si128(res1i, load_mask(mask));
                res1 = _mm_castsi128_ps(res1i);

                auto so_far = result.fetch(0);
                result.assign(0, _mm_or_ps(res1, so_far));
            }

            template<class GT, class QT, unsigned int J>
            struct gather_loop
            {
                static void gather(const QT& res, GT& result)
                {
                    do_gather<GT, QT, J>(res, result);
                    gather_loop<GT, QT, J - 1>::gather(res, result);
                }
            };
            template<class GT, class QT>
            struct gather_loop<GT, QT, 0>
            {
                static void gather(const QT& res, GT& result)
                {
                    do_gather<GT, QT, 0>(res, result);
                }
            };

            template<class GT, class QT>
            static void gather(const QT& res, GT& result)
            {
                // Go over every block of QT
                // Do gather on it
                // Merge everything into block GT[I]
                gather_loop<GT, QT, QT::blocks - 1>::gather(res, result);
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

                const auto shuf = sse::scatter_shuffle<GAP, START, LINE>::shuffle();
                const auto mask = sse::scatter_shuffle<GAP, START, LINE>::mask();

                // Take the value from curr_var[0],
                // (being var with offset START and GAP)
                // and scatter it over output_block[LINE]

                auto s1 = curr_var.fetch(0);
                auto res1 = _mm_shuffle_ps(s1, s1, shuf);

                auto res1i = _mm_castps_si128(res1);
                res1i = _mm_and_si128(res1i, load_mask(mask));
                res1 = _mm_castsi128_ps(res1i);

                auto so_far = output_block.fetch(LINE);
                output_block.assign(LINE, _mm_or_ps(res1, so_far));
            }

            template<class OT, class ST, unsigned int J>
            struct scatter_loop
            {
                static void scatter(OT& output_block, const ST& curr_var)
                {
                    do_scatter<OT, ST, J>(output_block, curr_var);
                    scatter_loop<OT, ST, J - 1>::scatter(output_block, curr_var);
                }
            };
            template<class OT, class ST>
            struct scatter_loop<OT, ST, 0>
            {
                static void scatter(OT& output_block, const ST& curr_var)
                {
                    do_scatter<OT, ST, 0>(output_block, curr_var);
                }
            };

            template<class OT, class ST>
            static void scatter(OT& output_block, const ST& curr_var)
            {
                scatter_loop<OT, ST, OT::blocks - 1>::scatter(output_block, curr_var);
            }
        };
    };

}
