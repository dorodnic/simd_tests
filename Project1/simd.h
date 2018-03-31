#pragma once
#include <vector>
#include <assert.h>
#include <type_traits>

typedef unsigned char byte;

namespace simd
{
    template<class T, class S, T S::* ptr>
    static constexpr int index_of()
    {
        return index_of_helper<T, S, ptr>::calc();
    }

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
            operator underlying_type() const { return _data; }

        private:
            underlying_type _data;
        };

        template<class T, unsigned int START, unsigned int GAP>
        struct gather_utils {};

        template<unsigned int START, unsigned int GAP>
        struct gather_utils<float, START, GAP>
        {
            template<class GT, class QT, unsigned int I>
            static constexpr void gather(const QT& res, GT& result)
            {
                result.assign(I, res.fetch(START));
            }
        };

        template<class T, unsigned int START, unsigned int GAP>
        struct scatter_utils {};

        template<unsigned int START, unsigned int GAP>
        struct scatter_utils<float, START, GAP>
        {
            template<class OT, class ST, unsigned int I>
            static void scatter(OT& output_block, const ST& curr_var)
            {
                const auto i = I;
                const auto s = START;
                const auto g = GAP;
                output_block.assign(I * GAP + START, curr_var.fetch(I));
            }
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
            operator underlying_type() const { return _data; }

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
            operator underlying_type() const { return _data; }

        private:
            underlying_type _data;
        };

        template<class T, unsigned int START, unsigned int GAP>
        struct gather_utils {};

        template<unsigned int START, unsigned int GAP>
        struct gather_utils<float, START, GAP>
        {
            enum { block_width = sizeof(native_simd<float>::underlying_type) / sizeof(float) };
            enum { bits_in_byte = 8 };

            static __m128i convt(const unsigned int x)
            {
                return  _mm_set_epi32(
                    (x & 0xff000000) ? 0xffffffff : 0, 
                    (x & 0x00ff0000) ? 0xffffffff : 0, 
                    (x & 0x0000ff00) ? 0xffffffff : 0, 
                    (x & 0x000000ff) ? 0xffffffff : 0);
            }

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

                res1i = _mm_and_si128(res1i, convt(maskJ));
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
            template<class OT, class ST, unsigned int I>
            static void scatter(OT& output_block, const ST& curr_var)
            {
                const auto i = I;
                const auto s = START;
                const auto g = GAP;
                auto val = curr_var.fetch(I);

                output_block.assign(I * GAP + START, val);
            }
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
        inline const simd_t& fetch(int idx) const { return _data[idx]; }

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

        class iterator
        {
        public:
            typedef transformation<T1, D1, T2, D2> this_class;

            inline iterator(transformation* owner, size_t index = 0) : _owner(owner), _index(index) {}
            inline iterator& operator++() { ++_index; return *this; }
            inline bool operator==(const iterator& other) const { return _index == other._index; }
            inline bool operator!=(const iterator& other) const { return !(*this == other); }

            inline iterator operator*() { return *this; }

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
                    ::template scatter<output_type, scatter_type, 0>(block, result);
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



