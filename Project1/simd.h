#pragma once
#include <vector>
#include <assert.h>
#include <type_traits>
#include <tmmintrin.h>
#include <array>

#include "core.h"
#include "sse.h"
#include "naive.h"
#include "avx.h"

namespace simd
{
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
        FORCEINLINE vector(const underlying_t* src)
        {
            for (int i = 0; i < K; i++)
                vectorized_wrapper::load(_data[i], src + i);
        }
        FORCEINLINE void store(underlying_t* ptr) const
        {
            for (int i = 0; i < K; i++)
                vectorized_wrapper::store(_data[i], ptr + i);
        }

        FORCEINLINE void assign(int idx, const simd_t& val)
        {
            //vectorized_wrapper::load(_data[idx], &val);
            _data[idx] = val;
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
            simd_t vec_y = vectorized_wrapper::vectorize(y);
            return{ *this, [&](simd_t& item) { return item - vec_y; } };
        }
        FORCEINLINE this_class operator+(float y)
        {
            simd_t vec_y = vectorized_wrapper::vectorize(y);
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
            simd_t vec_y = vectorized_wrapper::vectorize(y);
            return{ *this, [&](simd_t& item) { return item * vec_y; } };
        }
        FORCEINLINE this_class operator/(const this_class& y)
        {
            return{ *this, y, [&](simd_t& a, const simd_t& b) { return a / b; } };
        }
        FORCEINLINE this_class operator/(float y)
        {
            simd_t vec_y = vectorized_wrapper::vectorize(y);
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
            template<unsigned int INDEX, typename Dummy = int>
            struct gather_loop
            {
                static void gather(const input_type& block, std::array<gather_type, elements_in>& results)
                {
                    engine<ET>::template gather_utils<T1, INDEX - 1, elements_in>
                        ::template gather<gather_type, input_type>(block, results[INDEX - 1]);

                    gather_loop<INDEX - 1>::gather(block, results);
                }
            };
            template<typename Dummy>
            struct gather_loop<0, Dummy>
            {
                FORCEINLINE static void gather(const input_type& block, std::array<gather_type, elements_in>& results) {}
            };

        public:
            std::array<gather_type, elements_in> gather(const input_type& block) const
            {
                static_assert(input_type::blocks == elements_in, "No extra unrolling assumption!");

                std::array<gather_type, elements_in> result;
                gather_loop<elements_in>::gather(block, result);
                return result;
            }

            /// ========================= SCATTER ===============================================
        private:
            template<int INDEX>
            static void perform_scatter(output_type& block, const scatter_type& result)
            {
                const auto idx = INDEX;
                const auto start = elements_out - INDEX - 1;
                engine<ET>::template scatter_utils<T2, elements_out - INDEX - 1, elements_out>
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
