#pragma once
#include <vector>
#include <assert.h>
#include <type_traits>
#include <tmmintrin.h>

#include "sse_shuffle.h"
#include "sse_operators.h"

typedef unsigned char byte;

namespace simd
{
    enum engine_type
    {
        DEFAULT,
        NAIVE,
    };

    template<engine_type ST>
    struct engine { };

    template<>
    struct engine<NAIVE>
    {
        template<typename T>
        struct native_simd
        {
        public:
            typedef T underlying_type;
            typedef T representation_type;

            template<class S> // Load vector variable from memory
            FORCEINLINE static void load(T& target, const S* source)
            {
                target = *source;
            }

            template<class S> // Dump vector variable to memory
            FORCEINLINE static void store(const T& source, S* target)
            {
                *target = source;
            }

            template<class S> // Convert random constant to a "vector" type
            FORCEINLINE static underlying_type vectorize(S other)
            {
                return other;
            }
        };

        template<class T, unsigned int START, unsigned int GAP>
        struct gather_utils {};

        template<unsigned int START, unsigned int GAP>
        struct gather_utils<float, START, GAP>
        {
            template<class GT, class QT>
            static constexpr void gather(const QT& res, GT& result)
            {
                result.assign(0, res.fetch(START));
            }
        };

        template<class T, unsigned int START, unsigned int GAP>
        struct scatter_utils {};

        template<unsigned int START, unsigned int GAP>
        struct scatter_utils<float, START, GAP>
        {
            template<class OT, class ST>
            static void scatter(OT& output_block, const ST& curr_var)
            {
                output_block.assign(START, curr_var.fetch(0));
            }
        };
    };

    template<>
    struct engine<DEFAULT>
    {
        template<typename T>
        struct native_simd {};

        template<>
        struct native_simd<float>
        {
        public:
            typedef __m128 underlying_type;
            typedef __m128 representation_type;
            typedef native_simd<float> this_type;

            FORCEINLINE static void load(representation_type& target, const underlying_type* other)
            {
                target = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)other));
            }

            FORCEINLINE static void store(const representation_type& src, underlying_type* target)
            {
                _mm_storeu_ps((float*)target, src);
            }

            FORCEINLINE static underlying_type vectorize(float x)
            {
                return _mm_set_ps1(x);
            }
        private:
            this_type() = delete;
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
                return (3 - start<line>()) / GAP + 1;
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
            static constexpr unsigned int calc()
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

    template<typename E, typename T, int K>
    class vector
    {
    public:
        typedef typename E::template native_simd<T> vectorized_wrapper;
        typedef typename E::template native_simd<T>::representation_type simd_t;
        typedef typename E::template native_simd<T>::underlying_type underlying_t;
        typedef vector<E, T, K> this_class;
        enum { blocks = K };

        FORCEINLINE vector() : _data() {}
        FORCEINLINE vector(simd_t vals[K]) : _data(vals) {}
        FORCEINLINE vector(const typename underlying_t* src)
        {
            for (int i = 0; i < K; i++)
                vectorized_wrapper::load(_data[i], src + i);
        }
        FORCEINLINE void store(typename underlying_t* ptr) const
        {
            for (int i = 0; i < K; i++)
                vectorized_wrapper::store(_data[i], ptr + i);
        }

        FORCEINLINE void assign(int idx, const simd_t& val)
        {
            vectorized_wrapper::load(_data[idx], &val);
        }
        FORCEINLINE const simd_t& fetch(int idx) const { return _data[idx]; }

        template<class F>
        FORCEINLINE vector(vector& from, F f)
        {
            for (int i = 0; i < K; i++)
                _data[i] = f(from._data[i]);
        }

        template<class F>
        FORCEINLINE vector(vector& from, const vector& to, F f)
        {
            for (int i = 0; i < K; i++)
                _data[i] = f(from._data[i], to._data[i]);
        }

        FORCEINLINE this_class operator-(float y)
        {
            auto vec_y = vectorized_wrapper::vectorize(y);
            return{ *this, [&](simd_t& item) { return item - vec_y; } };
        }
        FORCEINLINE this_class operator+(float y)
        {
            auto vec_y = vectorized_wrapper::vectorize(y);
            return{ *this, [&](simd_t& item) { return item + vec_y; } };
        }
        FORCEINLINE this_class operator+(const this_class& y)
        {
            return{ *this, y, [&](simd_t& a,  const simd_t& b) { return a + b; } };
        }
        FORCEINLINE this_class operator-(const this_class& y)
        {
            return{ *this, y, [&](simd_t& a,  const simd_t& b) { return a - b; } };
        }
        FORCEINLINE this_class operator*(float y)
        {
            auto vec_y = vectorized_wrapper::vectorize(y);
            return{ *this, [&](simd_t& item) { return item * vec_y; } };
        }
        FORCEINLINE this_class operator/(const this_class& y)
        {
            return{ *this, y, [&](simd_t& a, const simd_t& b) { return a / b; } };
        }
        FORCEINLINE this_class operator/(float y)
        {
            auto vec_y = vectorized_wrapper::vectorize(y);
            return{ *this, [&](simd_t& item) { return item / vec_y; } };
        }

    private:
        simd_t _data[K];
    };

    template<int A, int B>
    struct GCD {
        enum { value = GCD<B, A % B>::value };
    };
    template<int A>
    struct GCD<A, 0> {
        enum { value = A };
    };
    template<int A, int B>
    struct LCM {
        enum { value = (A * B) / GCD<A, B>::value };
    };

    template<typename T1, class D1, typename T2, class D2, engine_type ET = DEFAULT>
    class transformation
    {
    public:
        typedef typename engine<ET>::template native_simd<T1>::underlying_type input_underlying_type;
        typedef typename engine<ET>::template native_simd<T2>::underlying_type output_underlying_type;

        enum { elements_in = sizeof(D1) / sizeof(T1) };
        enum { elements_out = sizeof(D2) / sizeof(T2) };

        enum { blocks_gather  = sizeof(input_underlying_type) / sizeof(T1) };
        enum { blocks_in      = blocks_gather * elements_in };
        enum { blocks_out     = blocks_gather * elements_out * sizeof(T1) / sizeof(T2) };

        enum { width_gather = (blocks_gather * sizeof(T1)) / sizeof(input_underlying_type) };
        enum { width_in = (blocks_in * sizeof(T1)) / sizeof(input_underlying_type) };
        enum { width_out = (blocks_out * sizeof(T2)) / sizeof(output_underlying_type) };

        typedef vector<engine<ET>, T1, width_in> input_type;
        typedef vector<engine<ET>, T1, width_gather> gather_type;
        typedef vector<engine<ET>, T2, width_out / elements_out> scatter_type;
        typedef vector<engine<ET>, T2, width_out> output_type;

        transformation(T1 * input, T2 * output, int count) : _count(count)
        {
            assert((count * sizeof(D1)) % (width_in * sizeof(input_underlying_type)) == 0);
            assert((count * sizeof(D2)) % (width_out * sizeof(output_underlying_type)) == 0);

            _src = reinterpret_cast<const input_underlying_type*>(input);
            _dst = reinterpret_cast<output_underlying_type*>(output);
        }

        template<class S>
        void print(S& s)
        {
            s << "Input Type:\t" << width_in << " x "
              << typeid(input_underlying_type).name() << "\t"
              << sizeof(input_underlying_type) * width_in << " bytes\t"
              << sizeof(input_underlying_type) * width_in / sizeof(T1) << " x "
              << typeid(T1).name() << "\t"
              << sizeof(input_underlying_type) * width_in / sizeof(D1) << " x "
              << typeid(D1).name() << "\t"
              << "\n";
            s << "Gather Type:\t" << width_gather
              << " x " << typeid(input_underlying_type).name() << "\t"
              << sizeof(input_underlying_type) * width_gather << " bytes\t"
              << sizeof(input_underlying_type) * width_gather / sizeof(T1) << " x "
              << typeid(T1).name() << "\t"
              << "\n";
            s << "Output Type:\t" << width_out << " x "
              << typeid(output_underlying_type).name() << "\t"
              << sizeof(output_underlying_type) * width_out << " bytes\t"
              << sizeof(output_underlying_type) * width_out / sizeof(T2) << " x "
              << typeid(T2).name() << "\t"
              << sizeof(output_underlying_type) * width_out / sizeof(D2) << " x "
              << typeid(D2).name() << "\t"
              << "\n";
        }
        
        template<class T>
        void apply(T action)
        {
            action(*this);
        }

        class iterator
        {
        public:
            typedef transformation<T1, D1, T2, D2> this_class;

            FORCEINLINE iterator(transformation* owner, size_t index = 0) : _owner(owner), _index(index) {}
            FORCEINLINE iterator& operator++() { ++_index; return *this; }
            FORCEINLINE bool operator==(const iterator& other) const { return _index == other._index; }
            FORCEINLINE bool operator!=(const iterator& other) const { return !(*this == other); }

            FORCEINLINE iterator operator*() { return *this; }

            /// ========================= GATHER ===============================================

        private:
            template<int INDEX>
            static void perform_gather(const input_type& block, gather_type& result)
            {
                engine<ET>::template gather_utils<typename T1, INDEX, elements_in>
                    ::template gather<gather_type, input_type>(block, result);
            }

        public:
            template<unsigned int INDEX>
            gather_type gather(const input_type& block) const
            {
                static_assert(INDEX < elements_in, "Must be a valid feild index!");
                static_assert(input_type::blocks == elements_in, "No extra unrolling assumption!");

                gather_type result;
                perform_gather<INDEX>(block, result);
                return result;
            }

            /// ========================= SCATTER ===============================================
        private:
            template<int INDEX>
            static void perform_scatter(output_type& block, const scatter_type& result)
            {
                const auto idx = INDEX;
                const auto start = elements_out - INDEX - 1;
                engine<ET>::template scatter_utils<typename T2, elements_out - INDEX - 1, elements_out>
                    ::template scatter<output_type, scatter_type>(block, result);
            }

            template<int INDEX, class T, class... A>
            struct scatter_helper
            {
                static void scatter_internal(output_type& result, const T& t, const A&... args)
                {
                    perform_scatter<INDEX>(result, t);
                    scatter_helper<INDEX - 1, A...>::scatter_internal(result, args...);
                }
            };
            template<class T>
            struct scatter_helper<0, T>
            {
                static void scatter_internal(output_type& result, const T& t)
                {
                    perform_scatter<0>(result, t);
                }
            };

        public:
            template<class T, class... A>
            output_type scatter(const T& t, const A&... args) const
            {
                static_assert(sizeof...(args) == elements_out - 1, 
                    "Scatter must be called with exactly number of arguments in the output type!");
                output_type result;
                scatter_helper<elements_out - 1, T, A...>::scatter_internal(result, t, args...);
                return result;
            }

            /// ========================= LOAD & STORE ===============================================

            input_type load()
            {
                return &_owner->_src[_index * width_in];
            }

            void store(const output_type& val)
            {
                val.store(&_owner->_dst[_index * width_out]);
            }

        private:
            size_t _index = 0;
            transformation* _owner;
        };

        FORCEINLINE iterator begin() { return iterator(this); }
        FORCEINLINE iterator end()
        {
            return iterator(this, (_count * sizeof(T1)) / sizeof(input_underlying_type));
        }

    private:
        const input_underlying_type* _src;
        output_underlying_type* _dst;
        const int _count;
    };

}