#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>

#include "simd.h"

struct float2 { float x; float y; };
struct float3 { float x; float y; float z; };
struct float4 { float x; float y; float z; float w; };
struct float5 { float x; float y; float z; float w; float u; };

template<class T>
void measure(T func)
{
    const auto cycles = 400;
    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < cycles; j++)
    {
        func();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration<double, std::micro>(end - start).count() / cycles;
    std::cout << diff << " micro" << std::endl;
}

template<class T>
struct test_app
{
    void operator()(T& ptr)
    {
        for (auto i : ptr)
        {
            auto block = i.load();
            auto x = i.gather<0>(block);
            auto y = i.gather<1>(block);
            auto z = i.gather<2>(block);

            auto out_block = i.scatter(x, y);
            i.store(out_block);
        }
    }
};

static std::vector<char> read_bytes(char const* filename)
{
    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    std::ifstream::pos_type pos = ifs.tellg();

    std::vector<char> result(pos);

    ifs.seekg(0, std::ios::beg);
    ifs.read(&result[0], pos);

    return result;
}

void main()
{
    std::vector<char> input = read_bytes("test.bin");
    std::vector<char> output(input.size(), 0);

    const auto input_size = input.size() / sizeof(float3);

    using namespace simd;
    transformation<float, float3, float, float2, NAIVE> simd_ptr((float*)input.data(), (float*)output.data(), input_size);
    transformation<float, float3, float, float2, DEFAULT> simd_ptr2((float*)input.data(), (float*)output.data(), input_size);

    //simd_ptr.print(std::cout);
    measure([&]()
    {
        for (int i = 0; i < input_size; i++)
        {
            auto&& xyz = ((float3*)input.data())[i];
            auto&& xy = ((float2*)output.data())[i];
            xy.x = xyz.x / xyz.z;
            xy.y = xyz.y / xyz.z;
        }
    });
    measure([&]()
    {
        simd_ptr.apply(test_app<decltype(simd_ptr)>());
    });
    measure([&]()
    {
        simd_ptr2.apply(test_app<decltype(simd_ptr2)>());
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

    int x;
    std::cin >> x;
}