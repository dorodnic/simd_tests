#pragma once

#include "core.h"

namespace simd
{
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
}