#pragma once
#include <vector>
#include <assert.h>
#include <type_traits>

typedef unsigned char byte;

#ifdef _MSC_VER
#define FORCEINLINE __forceinline
#else
#define FORCEINLINE inline __attribute__((always_inline))
#endif

namespace simd
{
    enum engine_type
    {
        DEFAULT,
        NAIVE,
        SUPERSPEED,
    };

    template<engine_type ST>
    struct engine { };
}
