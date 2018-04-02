#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>

#include "simd.h"

struct float2 { float x; float y; };
struct float3 { float x; float y; float z; };
struct float4 { float x; float y; float z; float w; };
struct float5 { float x; float y; float z; float w; float u; };

typedef struct rs2_intrinsics
{
    float         width;
    float         height;
    float         ppx;
    float         ppy;
    float         fx;
    float         fy;
} rs2_intrinsics;
typedef struct rs2_extrinsics
{
    float rotation[9];    /**< Column-major 3x3 rotation matrix */
    float translation[3]; /**< Three-element translation vector, in meters */
} rs2_extrinsics;


template<class T>
void measure(T func)
{
    const auto cycles = 5;
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
        rs2_intrinsics intr{ 640, 480, 100, 200, 50, 70 };
        rs2_extrinsics extr{ { 1.1, 0.9, 0.2, 0.3, 0.9, 0.7, 0, 0.2, 0.3 },{ 0.1, 0.5, 0.6 } };

        for (auto i : ptr)
        {
            auto block = i.load();
            auto x = i.gather<0>(block);
            auto y = i.gather<1>(block);
            auto z = i.gather<2>(block);

            auto to_point_x = x* extr.rotation[0] + y* extr.rotation[3] + z * extr.rotation[6] + extr.translation[0];
            auto to_point_y = x* extr.rotation[1] + y* extr.rotation[4] + z * extr.rotation[7] + extr.translation[1];
            auto to_point_z = x* extr.rotation[2] + y* extr.rotation[5] + z * extr.rotation[8] + extr.translation[2];

            auto u1 = to_point_x / to_point_z, v1 = to_point_y / to_point_z;


            auto px = u1 * intr.fx + intr.ppx;
            auto py = v1 * intr.fy + intr.ppy;

            auto u = px / (intr.width);
            auto v = py / (intr.height);

            auto out_block = i.scatter(u, v);
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

    auto input_ptr = (float3*)input.data();
    auto output_ptr = (float2*)output.data();

    const auto input_size = input.size() / sizeof(float3);

    using namespace simd;
    transformation<float, float3, float, float2, NAIVE>   simd_ptr((float*)input.data(), (float*)output.data(), input_size);
    transformation<float, float3, float, float2, DEFAULT> simd_ptr2((float*)input.data(), (float*)output.data(), input_size);

    simd_ptr.print(std::cout);
    measure([&]()
    {
        rs2_intrinsics intr{ 640, 480, 100, 200, 50, 70 };
        rs2_extrinsics extr{ { 1.1, 0.9, 0.2, 0.3, 0.9, 0.7, 0, 0.2, 0.3 },{ 0.1, 0.5, 0.6 } };

        for (int i = 0; i < input_size; i++)
        {
            auto&& xyz = input_ptr[i];
            auto&& xy = output_ptr[i];

            auto x = xyz.x; auto y = xyz.y; auto z = xyz.z;

            auto to_point_x = x * extr.rotation[0] + y* extr.rotation[3] + z * extr.rotation[6] + extr.translation[0];
            auto to_point_y = x * extr.rotation[1] + y* extr.rotation[4] + z * extr.rotation[7] + extr.translation[1];
            auto to_point_z = x * extr.rotation[2] + y* extr.rotation[5] + z * extr.rotation[8] + extr.translation[2];

            auto u1 = to_point_x / to_point_z, v1 = to_point_y / to_point_z;


            auto px = u1 * intr.fx + intr.ppx;
            auto py = v1 * intr.fy + intr.ppy;

            auto u = px / (intr.width);
            auto v = py / (intr.height);

            xy.x = u; xy.y = v;
        }
    });

    for (int i = 0; i < 10; i++)
    {
        std::cout << output_ptr[i].x << ", " << output_ptr[i].y << "  ";
    }
    std::cout << std::endl;

    measure([&]()
    {
        simd_ptr.apply(test_app<decltype(simd_ptr)>());
    });

    for (int i = 0; i < 10; i++)
    {
        std::cout << output_ptr[i].x << ", " << output_ptr[i].y << "  ";
    }
    std::cout << std::endl;

    measure([&]()
    {
        simd_ptr2.apply(test_app<decltype(simd_ptr2)>());
    });

    for (int i = 0; i < 10; i++)
    {
        std::cout << output_ptr[i].x << ", " << output_ptr[i].y << "  ";
    }
    std::cout << std::endl;

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