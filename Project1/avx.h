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
            enum { block_width = sizeof(native_simd<float>::underlying_type) / sizeof(float) };
            enum { bits_in_byte = 8 };

            template<unsigned int line>
            static constexpr unsigned int mask()
            {
                return  1 << index<line>() * bits_in_byte | ((create_mask<num_of_items<line>()>()) << (index<line>() + 1) * bits_in_byte);
            }

            template<unsigned int num_of_items>
            static constexpr unsigned int create_mask()
            {
                return 1 | create_mask<num_of_items - 1>() << bits_in_byte;
            }
            template<>
            static constexpr unsigned int create_mask<1>()
            {
                return  0;
            }

            template<unsigned int line>
            static constexpr unsigned int start()
            {
                return (start<line - 1>() + num_of_items<line - 1>() * GAP) % block_width;
            }
            template<>
            static constexpr unsigned int start<1>()
            {
                return START;
            }

            template<unsigned int line>
            static constexpr unsigned int num_of_items()
            {
                return (7 - start<line>()) / GAP + 1;
            }
            template<>
            static constexpr unsigned int num_of_items<0>()
            {
                return 0;
            }

            template<unsigned int line>
            static constexpr unsigned int index()
            {
                return index<line - 1>() + num_of_items<line - 1>();
            }
            template<>
            static constexpr unsigned int index<1>()
            {
                return 0;
            }

            template<unsigned int line>
            static constexpr unsigned int calc1()
            {
                return (_MM_SHUFFLE(start<line>() + 3 * GAP,
                    start<line>() + 2 * GAP,
                    start<line>() + GAP,
                    start<line>())
                    << index<line>() * 2);
            }

            template<class GT, class QT, unsigned int J>
            static void do_gather(const QT& res, GT& result)
            {
                auto s1 = res.fetch(J);

                const auto shuffle = calc<J + 1>();

                auto res1 = _mm_shuffle_ps(s1, s1, shuffle);
                auto res1i = _mm_castps_si128(res1);

                const auto maskJ = mask<J + 1>();

                res1i = _mm_and_si128(res1i, load_mask(maskJ));
                res1 = _mm_castsi128_ps(res1i);

                auto so_far = result.fetch(0);
                result.assign(0, _mm_or_ps(res1, so_far));
            }

            template<class GT, class QT, unsigned int J>
            struct gather_loop
            {
                static constexpr void gather(const QT& res, GT& result)
                {
                    do_gather<GT, QT, J>(res, result);
                    gather_loop<GT, QT, J - 1>::gather(res, result);
                }
            };
            template<class GT, class QT>
            struct gather_loop<GT, QT, 0>
            {
                static constexpr void gather(const QT& res, GT& result)
                {
                    do_gather<GT, QT, 0>(res, result);
                }
            };

            template<class GT, class QT>
            static constexpr void gather(const QT& res, GT& result)
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
                static constexpr void scatter(OT& output_block, const ST& curr_var)
                {
                    do_scatter<OT, ST, J>(output_block, curr_var);
                    scatter_loop<OT, ST, J - 1>::scatter(output_block, curr_var);
                }
            };
            template<class OT, class ST>
            struct scatter_loop<OT, ST, 0>
            {
                static constexpr void scatter(OT& output_block, const ST& curr_var)
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