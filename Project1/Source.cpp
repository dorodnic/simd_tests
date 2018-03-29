#include <chrono>
#include <iostream>

#include <tmmintrin.h> // For SSE3 intrinsic used in unpack_yuy2_sse
#include "simd.h"

struct float2 { float x; float y; };
struct float3 { float x; float y; float z; };
struct float5 { float x; float y; float z; float w; float u; };

SET_OFFSET_OF(&float2::x, 0);

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
    float input[]{ 1.f, 11.f, 99.f, 2.f, 12.f, 98.f, 3.f, 13.f, 97.f, 4.f, 14.f, 98.f };
    float output[12];
    memset(output, 0, 12 * sizeof(float));

    using namespace simd;
    transformation<float, float3, float, float3, DEFAULT> simd_ptr(input, output, 4);

    measure([&]()
    {
        for (auto i : simd_ptr)
        {
            auto block = i.load();
            auto res = i.gather<0>(block);

            i.store(block);
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