#include <chrono>
#include <iostream>
#include <vector>

#include <tmmintrin.h> // For SSE3 intrinsic used in unpack_yuy2_sse
#include "simd.h"

struct float2 { float x; float y; };
struct float3 { float x; float y; float z; };
struct float5 { float x; float y; float z; float w; float u; };

template<class T>
void measure(T func)
{
    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < 1000; j++)
    {
        func();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << diff << " micro" << std::endl;
}

void main()
{
    std::vector<float> input(12 * 16 * 10);
    std::vector<float> output(12 * 16 * 10);

    for (int i = 0; i < input.size(); i++)
    {
        if (i % 3 == 0) input[i] = i / 3;
        if (i % 3 == 1) input[i] = 100 + i / 3;
        if (i % 3 == 2) input[i] = 1000 - i / 3;
    }

    using namespace simd;
    transformation<float, float3, float, float5, NAIVE> simd_ptr(input.data(), output.data(), 12 * 5);

    measure([&]()
    {
        for (auto i : simd_ptr)
        {
            auto block = i.load();
            auto x = i.gather<0>(block);
            auto y = i.gather<1>(block);
            auto z = i.gather<2>(block);

            auto u = y;
            auto v = z;

            decltype(u) data[5] = { u, v, x, y, z };
            auto out_block = i.scatter(data);
            i.store(out_block);
        }
    });

    //auto x_gather = simd_data_gather<float, 0, 3>();
    //auto y_gather = simd_data_gather<float, 1, 3>();
    //auto z_gather = simd_data_gather<float, 2, 3>();

    //for (auto i : simd_ptr)
    //{
    //    auto x = i.gather(x_gather);
    //    auto y = i.gather(y_gather);
    //    auto z = i.gather(z_gather);

    //    simd_item<float> data[2] = { x, y };
    //    i.scatter(data);
    //}
}