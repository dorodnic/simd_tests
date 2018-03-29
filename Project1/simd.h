#pragma once
#include <vector>
#include <assert.h>

typedef unsigned char byte;

template<class T, class S, S T::* ptr>
struct offset_of_helper
{
    static constexpr int calc()
    {
        return offsetof(T, ptr);
    }
};

template<class T, class S, S T::* ptr>
static constexpr T foo();

template<class T, class S, S T::* ptr>
static constexpr S goo();

#define SET_OFFSET_OF(X, O) \
template<>\
struct offset_of_helper<decltype(foo<X>()), decltype(goo<X>()), X>\
{\
    static constexpr int calc()\
    {\
        return O;\
    }\
};

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

            inline native_simd(underlying_type data) : _data(data) {}
            inline native_simd(const underlying_type* data) : _data(*data) {}
            inline native_simd() : _data(T{}) {}
            inline native_simd(const native_simd& data) { _data = data._data; }

            inline native_simd operator-(int y) const
            {
                return native_simd(_data - y);
            }
            inline native_simd operator+(int y) const
            {
                return native_simd(_data + y);
            }
            inline native_simd operator+(const native_simd& y) const
            {
                return native_simd(_data + y._data);
            }
            inline native_simd operator-(const native_simd& y) const
            {
                return native_simd(_data - y._data);
            }
            inline native_simd operator >> (int y) const
            {
                return native_simd(_data >> y);
            }
            inline native_simd operator<<(int y) const
            {
                return native_simd(_data << y);
            }
            inline native_simd operator*(int y) const
            {
                return native_simd(_data * y);
            }
            inline native_simd min(int y) const
            {
                return native_simd(std::min(_data, y));
            }
            inline native_simd max(int y) const
            {
                return native_simd(std::max(_data, y));
            }
            inline void store(underlying_type* ptr) const
            {
                *ptr = _data;
            }

        private:
            underlying_type _data;
        };

    };

    template<>
    struct engine<DEFAULT>
    {
        template<typename T>
        struct native_simd {};


        template<>
        struct native_simd<uint16_t>
        {
        public:
            typedef __m128i underlying_type;

            inline native_simd(underlying_type data) : _data(data) {}
            inline native_simd(const underlying_type* data) : _data(_mm_loadu_si128(data)) {}
            inline native_simd() : _data(_mm_set1_epi8(0)) {}
            inline native_simd(const native_simd& data) { _data = data._data; }

            inline native_simd operator-(int y) const
            {
                return native_simd(_mm_subs_epi16(_data, _mm_set1_epi16(y)));
            }
            inline native_simd operator+(int y) const
            {
                return native_simd(_mm_adds_epi16(_data, _mm_set1_epi16(y)));
            }
            inline native_simd operator+(const native_simd& y) const
            {
                return native_simd(_mm_adds_epi16(_data, y._data));
            }
            inline native_simd operator-(const native_simd& y) const
            {
                return native_simd(_mm_subs_epi16(_data, y._data));
            }
            inline native_simd operator>>(int y) const
            {
                return native_simd(_mm_slli_epi16(_data, y));
            }
            inline native_simd operator<<(int y) const
            {
                return native_simd(_mm_srai_epi16(_data, y));
            }
            inline native_simd operator*(int y) const
            {
                return native_simd(_mm_mulhi_epi16(_data, _mm_set1_epi16(y << 4)));
            }
            inline native_simd min(int y) const
            {
                return native_simd(_mm_min_epi16(_mm_set1_epi16(y), _data));
            }
            inline native_simd max(int y) const
            {
                return native_simd(_mm_max_epi16(_mm_set1_epi8(y), _data));
            }
            inline void store(underlying_type* ptr) const
            {
                _mm_storeu_si128(ptr, _data);
            }

        private:
            underlying_type _data;
        };

        template<>
        struct native_simd<float>
        {
        public:
            typedef __m128 underlying_type;

            inline native_simd(underlying_type data) : _data(data) {}
            inline native_simd(const underlying_type* data) : _data(_mm_castsi128_ps(_mm_loadu_si128((const __m128i*)data))) {}
            inline native_simd() : _data(_mm_set_ps1(0)) {}
            inline native_simd(const native_simd& data) { _data = data._data; }

            inline native_simd operator-(float y) const
            {
                return native_simd(_mm_sub_ps(_data, _mm_set_ps1(y)));
            }
            inline native_simd operator+(float y) const
            {
                return native_simd(*this + _mm_set_ps1(y));
            }
            inline native_simd operator+(const native_simd& y) const
            {
                return native_simd(_mm_add_ps(_data, y._data));
            }
            inline native_simd operator-(const native_simd& y) const
            {
                return native_simd(_mm_sub_ps(_data, y._data));
            }
            inline native_simd operator/(const native_simd& y) const
            {
                return native_simd(_mm_div_ps(_data, y._data));
            }
            inline native_simd operator/(float y) const
            {
                return native_simd(_mm_div_ps(_data, _mm_set_ps1(y)));
            }
            inline native_simd operator*(float y) const
            {
                return native_simd(_mm_mul_ps(_data, _mm_set_ps1(y)));
            }
            inline void store(underlying_type* ptr) const
            {
                _mm_storeu_ps((float*)ptr, _data);
            }

        private:
            underlying_type _data;
        };
    };

    template<typename E, typename T, int K>
    class vector
    {
    public:
        typedef typename E::template native_simd<T> simd_t;
        typedef vector<E, T, K> this_class;
        enum { blocks = K };

        inline vector() : _data() {}
        inline vector(simd_t vals[K]) : _data(vals) {}
        inline vector(const typename simd_t::underlying_type* src)
        {
            for (int i = 0; i < K; i++)
                _data[i] = src + i;
        }
        inline void store(typename simd_t::underlying_type* ptr) const
        {
            for (int i = 0; i < K; i++)
                _data[i].store(ptr + i);
        }

        inline void assign(int idx, const simd_t& val)
        {
            _data[idx] = val;
        }

        template<int START, int COUNT>
        vector<E, T, COUNT> subset() const
        {
            vector<E, T, COUNT> result;
            for (int i = 0; i < COUNT; i++)
                result.assign(i, _data[START + i]);
            return result;
        }

        template<class F>
        inline vector(vector& from, F f)
        {
            for (int i = 0; i < K; i++)
                data[i] = F(from._data[i]);
        }

        inline this_class operator-(float y)
        {
            return{ *this, [&](simd_t& item) { return item - y; } };
        }
        inline this_class operator+(float y)
        {
            return{ *this, [&](simd_t& item) { return item + y; } };
        }
        inline this_class operator+(const this_class& y)
        {
            return{ *this, [&](simd_t& item) { return item + y; } };
        }
        inline this_class operator-(const this_class& y)
        {
            return{ *this, [&](simd_t& item) { return item - y; } };
        }
        inline this_class operator>>(float y)
        {
            return{ *this, [&](simd_t& item) { return item >> y; } };
        }
        inline this_class operator<<(float y)
        {
            return{ *this, [&](simd_t& item) { return item << y; } };
        }
        inline this_class operator*(float y)
        {
            return{ *this, [&](simd_t& item) { return item * y; } };
        }
        inline this_class operator/(const this_class& y)
        {
            return{ *this, [&](simd_t& item) { return item / y; } };
        }
        inline this_class operator/(int y)
        {
            return{ *this, [&](simd_t& item) { return item / y; } };
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

        enum { blocks_in  = LCM<sizeof(D1) / sizeof(T1), 
                                sizeof(input_underlying_type) / sizeof(T1)>::value };
        enum { blocks_out = LCM<sizeof(D2) / sizeof(T2), 
                                sizeof(output_underlying_type) / sizeof(T2)>::value };

        enum { width_bytes  = LCM<blocks_in * sizeof(T1), blocks_out * sizeof(T2)>::value };
        enum { width_in = width_bytes / sizeof(input_underlying_type) };
        enum { width_out = width_bytes / sizeof(output_underlying_type) };

        enum { elements_in = sizeof(D1) / sizeof(T1) };
        enum { elements_out = sizeof(D2) / sizeof(T2) };

        typedef vector<engine<ET>, T1, width_in> input_type;
        typedef vector<engine<ET>, T1, width_in / elements_in> gather_type;
        typedef vector<engine<ET>, T2, width_out> output_type;

        transformation(T1 * input, T2 * output, int count) : _count(count)
        {
            assert((count * sizeof(D1)) % (width_bytes) == 0);
            assert((count * sizeof(D2)) % (width_bytes) == 0);

            _src = reinterpret_cast<const input_underlying_type*>(input);
            _dst = reinterpret_cast<output_underlying_type*>(output);
        }

        class iterator
        {
        public:
            typedef transformation<T1, D1, T2, D2> this_class;

            inline iterator(transformation* owner, size_t index = 0) : _owner(owner), _index(index) {}
            inline iterator& operator++() { ++_index; return *this; }
            inline bool operator==(const iterator& other) const { return _index == other._index; }
            inline bool operator!=(const iterator& other) const { return !(*this == other); }

            inline iterator operator*() { return *this; }

            input_type load()
            {
                return &_owner->_src[_index * width_in];
            }

            template<int OFFSET, int I>
            static void perform_gather(const input_type& block, gather_type& result)
            {
                auto res = block.subset<I * gather_type::blocks, gather_type::blocks>();
                // Now we need to somehow gather from res into block[i]
            }

            template<int OFFSET, int I>
            struct gather_loop
            {
                static void gather(const input_type& block, gather_type& result)
                {
                    perform_gather<OFFSET, I>(block, result);
                    gather_loop<OFFSET, I - 1>::gather(block, result);
                }
            };
            template<int OFFSET>
            struct gather_loop<OFFSET, 0>
            {
                static void gather(const input_type& block, gather_type& result)
                {
                    perform_gather<OFFSET, 0>(block, result);
                }
            };

            template<T1 D1::*ptr>
            gather_type gather(const input_type& block)
            {
                gather_type result;

                gather_loop<offset(ptr), gather_type::blocks>::gather(block, result);
                
                return result;
            }

            void store(const output_type& val)
            {
                val.store(&_owner->_dst[_index * width_out]);
            }

        private:
            size_t _index = 0;
            transformation* _owner;
        };

        inline iterator begin() { return iterator(this); }
        inline iterator end()
        {
            return iterator(this, (_count * sizeof(T1)) / sizeof(input_underlying_type));
        }

    private:
        const input_underlying_type* _src;
        output_underlying_type* _dst;
        const int _count;
    };

}



