/*
 * Tiny self-contained version of the PCG Random Number Generation for C++ put
 * together from pieces of the much larger C/C++ codebase with vectorization
 * using Enoki.
 *
 * Wenzel Jakob, February 2017
 *
 * The PCG random number generator was developed by Melissa O'Neill
 * <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 */

#pragma once

#include <mitsuba/core/simd.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/traits.h>
#include <mitsuba/core/fwd.h>
#include <enoki/random.h>

NAMESPACE_BEGIN(enoki)
/// Prints the canonical representation of a PCG32 object.
template <typename Value>
std::ostream& operator<<(std::ostream &os, const enoki::PCG32<Value> &p) {
    os << "PCG32[" << std::endl
       << "  state = 0x" << std::hex << p.state << "," << std::endl
       << "  inc = 0x" << std::hex << p.inc << std::endl
       << "]";
    return os;
}
NAMESPACE_END(enoki)

NAMESPACE_BEGIN(mitsuba)

template <typename UInt32> using PCG32 = std::conditional_t<is_dynamic_array_v<UInt32>,
                                                            enoki::PCG32<UInt32, 1>,
                                                            enoki::PCG32<UInt32>>;

/**
 * \brief Generate fast and reasonably good pseudorandom numbers using the
 * Tiny Encryption Algorithm (TEA) by David Wheeler and Roger Needham.
 *
 * For details, refer to "GPU Random Numbers via the Tiny Encryption Algorithm"
 * by Fahad Zafar, Marc Olano, and Aaron Curtis.
 *
 * \param v0
 *     First input value to be encrypted (could be the sample index)
 * \param v1
 *     Second input value to be encrypted (e.g. the requested random number dimension)
 * \param rounds
 *     How many rounds should be executed? The default for random number
 *     generation is 4.
 * \return
 *     A uniformly distributed 32-bit integer
 */

template <typename UInt32>
UInt32 sample_tea_32(UInt32 v0, UInt32 v1, int rounds = 4) {
    UInt32 sum = 0;

    ENOKI_NOUNROLL for (int i = 0; i < rounds; ++i) {
        sum += 0x9e3779b9;
        v0 += (sl<4>(v1) + 0xa341316c) ^ (v1 + sum) ^ (sr<5>(v1) + 0xc8013ea4);
        v1 += (sl<4>(v0) + 0xad90777d) ^ (v0 + sum) ^ (sr<5>(v0) + 0x7e95761e);
    }

    return v1;
}

/**
 * \brief Generate fast and reasonably good pseudorandom numbers using the
 * Tiny Encryption Algorithm (TEA) by David Wheeler and Roger Needham.
 *
 * For details, refer to "GPU Random Numbers via the Tiny Encryption Algorithm"
 * by Fahad Zafar, Marc Olano, and Aaron Curtis.
 *
 * \param v0
 *     First input value to be encrypted (could be the sample index)
 * \param v1
 *     Second input value to be encrypted (e.g. the requested random number dimension)
 * \param rounds
 *     How many rounds should be executed? The default for random number
 *     generation is 4.
 * \return
 *     A uniformly distributed 64-bit integer
 */

template <typename UInt32>
uint64_array_t<UInt32> sample_tea_64(UInt32 v0, UInt32 v1, int rounds = 4) {
    UInt32 sum = 0;

    ENOKI_NOUNROLL for (int i = 0; i < rounds; ++i) {
        sum += 0x9e3779b9;
        v0 += (sl<4>(v1) + 0xa341316c) ^ (v1 + sum) ^ (sr<5>(v1) + 0xc8013ea4);
        v1 += (sl<4>(v0) + 0xad90777d) ^ (v0 + sum) ^ (sr<5>(v0) + 0x7e95761e);
    }

    return uint64_array_t<UInt32>(v0) + sl<32>(uint64_array_t<UInt32>(v1));
}


/**
 * \brief Generate fast and reasonably good pseudorandom numbers using the
 * Tiny Encryption Algorithm (TEA) by David Wheeler and Roger Needham.
 *
 * This function uses \ref sample_tea to return single precision floating point
 * numbers on the interval <tt>[0, 1)</tt>
 *
 * \param v0
 *     First input value to be encrypted (could be the sample index)
 * \param v1
 *     Second input value to be encrypted (e.g. the requested random number dimension)
 * \param rounds
 *     How many rounds should be executed? The default for random number
 *     generation is 4.
 * \return
 *     A uniformly distributed floating point number on the interval <tt>[0, 1)</tt>
 */
template <typename UInt32>
float32_array_t<UInt32> sample_tea_float32(UInt32 v0, UInt32 v1, int rounds = 4) {
    return reinterpret_array<float32_array_t<UInt32>>(
        sr<9>(sample_tea_32(v0, v1, rounds)) | 0x3f800000u) - 1.f;
}

/**
 * \brief Generate fast and reasonably good pseudorandom numbers using the
 * Tiny Encryption Algorithm (TEA) by David Wheeler and Roger Needham.
 *
 * This function uses \ref sample_tea to return double precision floating point
 * numbers on the interval <tt>[0, 1)</tt>
 *
 * \param v0
 *     First input value to be encrypted (could be the sample index)
 * \param v1
 *     Second input value to be encrypted (e.g. the requested random number dimension)
 * \param rounds
 *     How many rounds should be executed? The default for random number
 *     generation is 4.
 * \return
 *     A uniformly distributed floating point number on the interval <tt>[0, 1)</tt>
 */

template <typename UInt32>
float64_array_t<UInt32> sample_tea_float64(UInt32 v0, UInt32 v1, int rounds = 4) {
    return reinterpret_array<float64_array_t<UInt32>>(
        sr<12>(sample_tea_64(v0, v1, rounds)) | 0x3ff0000000000000ull) - 1.0;
}


/// Alias to \ref sample_tea_float32 or \ref sample_tea_float64 based given type size
template <typename UInt>
auto sample_tea_float(UInt v0, UInt v1, int rounds = 4) {
    if constexpr(std::is_same_v<scalar_t<UInt>, uint32_t>)
        return sample_tea_float32(v0, v1, rounds);
    else
        return sample_tea_float64(v0, v1, rounds);
}

template <typename Float, typename PCG32>
MTS_INLINE Float next_float(PCG32 *rng, mask_t<Float> active = true) {
    if constexpr (is_double_v<scalar_t<Float>>)
        return rng->next_float64(active);
    else
        return rng->next_float32(active);
}

/// Generate latin hypercube samples
template <typename Point, typename PCG32>
auto latin_hypercube(PCG32 *rng, size_t sample_count, bool jitter=true) {
    using Float = std::conditional_t<is_static_array_v<Point>, value_t<Point>, Point>;
    using UInt32 = replace_scalar_t<Float, uint32_t>;
    using ScalarFloat = scalar_t<Float>;
    using FloatStorage = DynamicBuffer<Float>;

    constexpr size_t DIM = is_static_array_v<Point> ? array_size_v<Point> : 1;
    size_t wavefront_size = enoki::slices(rng->state);

    FloatStorage dest = empty<FloatStorage>(wavefront_size * sample_count * DIM);
    UInt32 lane_offsets = arange<UInt32>(wavefront_size) * sample_count;
    UInt32 lane_offsets_dest = lane_offsets * DIM;

    ScalarFloat delta = rcp(ScalarFloat(sample_count));
    if (jitter) {
        for (size_t i = 0; i < sample_count; ++i)
            for (size_t j = 0; j < DIM; ++j)
                scatter(dest, (i + rng->next_float32()) * delta, lane_offsets_dest + DIM * i + j);
    } else {
        for (size_t i = 0; i < sample_count; ++i)
            for (size_t j = 0; j < DIM; ++j)
                scatter(dest, (i + Float(0.5f)) * delta, lane_offsets_dest + DIM * i + j);
    }

    if constexpr (is_cuda_array_v<Float>)
        cuda_eval();

    // Swap the sample values (independently for every dimensions)
    for (size_t i = 0; i < sample_count; ++i) {
        for (size_t j = 0; j < DIM; ++j) {
            UInt32 current = lane_offsets_dest + DIM * i + j;
            UInt32 other = lane_offsets_dest + DIM * rng->next_uint32_bounded(sample_count) + j;

            Float tmp_current = gather<Float>(dest, current);
            Float tmp_other   = gather<Float>(dest, other);

            scatter(dest, tmp_current, other);
            scatter(dest, tmp_other, current);

            if constexpr (is_cuda_array_v<Float>)
                cuda_eval();
        }
    }

    std::vector<Point> result(sample_count);
    for (size_t i = 0; i < sample_count; ++i)
        result[i] = gather<Point>(dest, lane_offsets + i);

    return result;
}

/// Generate latin hypercube samples
template <typename Point, typename PCG32>
auto wavefront_latin_hypercube(PCG32 *rng, size_t wavefront_count, size_t samples_per_wavefront = 1, bool jitter=true) {
    using Float = std::conditional_t<is_static_array_v<Point>, value_t<Point>, Point>;
    using UInt32 = replace_scalar_t<Float, uint32_t>;
    using ScalarFloat = scalar_t<Float>;
    using FloatStorage  = DynamicBuffer<Float>;
    using UInt32Storage = DynamicBuffer<UInt32>;

    constexpr size_t dim = is_static_array_v<Point> ? array_size_v<Point> : 1;
    size_t wavefront_size = enoki::slices(rng->state);
    size_t wavefront_res = wavefront_size / samples_per_wavefront;
    size_t total_sample_count = wavefront_count * samples_per_wavefront;

    // Generate indices table
    UInt32Storage indices = arange<UInt32Storage>(wavefront_res * total_sample_count * dim);
    indices = (indices / UInt32Storage(dim)) % total_sample_count;

    // Jitter indices to get positions (if necessary)
    FloatStorage positions = indices;
    if (jitter) {
        UInt32 offsets = arange<UInt32>(wavefront_size) * wavefront_count * dim;
        for (size_t i = 0; i < wavefront_count * dim; ++i) {
            scatter_add(positions, rng->next_float32(), offsets);
            offsets += 1;
        }
    } else {
        positions += 0.5f;
    }

    // Scale values so they fall in [0, 1)
    positions *= rcp(ScalarFloat(total_sample_count));

    if constexpr (is_cuda_array_v<Float>)
        cuda_eval();

    // Generate random numbers for permutations
    UInt32Storage uint32_rands = empty<FloatStorage>(wavefront_size * wavefront_count * dim);
    UInt32 offsets = arange<UInt32>(wavefront_size) * wavefront_count * dim;
    for (size_t i = 0; i < wavefront_count * dim; ++i) {
        scatter(uint32_rands, rng->next_uint32_bounded(total_sample_count), offsets);
        offsets += 1;
    }

    // Swap the sample values (independently for every dimensions)
    UInt32 wavefront_offsets = arange<UInt32>(wavefront_res) * total_sample_count * dim;
    for (size_t i = 0; i < total_sample_count; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            UInt32 current = wavefront_offsets + dim * i + j;
            UInt32 other   = wavefront_offsets + dim * gather<UInt32>(uint32_rands, current) + j;

            Float tmp_current = gather<Float>(positions, current);
            Float tmp_other   = gather<Float>(positions, other);

            scatter(positions, tmp_current, other);
            scatter(positions, tmp_other, current);

            if constexpr (is_cuda_array_v<Float>)
                cuda_eval();
        }
    }

    // Reconstruct wavefront samples
    UInt32 wavefront_point_offsets = arange<UInt32>(wavefront_size) * wavefront_count;
    std::vector<Point> result(wavefront_count);
    for (size_t i = 0; i < wavefront_count; ++i)
        result[i] = gather<Point>(positions, wavefront_point_offsets + i);

    return result;
}

template <typename UInt32>
UInt32 sample_permutation(UInt32 value, uint32_t sample_count, UInt32 seed, int rounds = 2) {
    uint32_t  n = log2i(sample_count);
    Assert((1 << n) == sample_count, "sample_count should be a power of 2");

    for (uint32_t  level = 0; level < n; ++level) {
        UInt32 bit = UInt32(1 << level);
        // Take a random integer indentical for values that might be swapped at this level
        UInt32 rand = sample_tea_32(value | bit, seed, rounds);
        masked(value, eq(rand & bit, bit)) = value ^ bit;
    }

    return value;
}

template <typename UInt32>
UInt32 kensler_permute(UInt32 i, uint32_t l, UInt32 p, mask_t<UInt32> active = true) {
    UInt32 w = l - 1;
    w |= w >> 1;
    w |= w >> 2;
    w |= w >> 4;
    w |= w >> 8;
    w |= w >> 16;

    mask_t<UInt32> invalid = true;
    do {
        masked(i, invalid) ^= p;
        masked(i, invalid) *= 0xe170893d;
        masked(i, invalid) ^= p >> 16;
        masked(i, invalid) ^= (i & w) >> 4;
        masked(i, invalid) ^= p >> 8;
        masked(i, invalid) *= 0x0929eb3f;
        masked(i, invalid) ^= p >> 23;
        masked(i, invalid) ^= (i & w) >> 1;
        masked(i, invalid) *= 1 | p >> 27;
        masked(i, invalid) *= 0x6935fa69;
        masked(i, invalid) ^= (i & w) >> 11;
        masked(i, invalid) *= 0x74dcb303;
        masked(i, invalid) ^= (i & w) >> 2;
        masked(i, invalid) *= 0x9e501cc3;
        masked(i, invalid) ^= (i & w) >> 2;
        masked(i, invalid) *= 0xc860a3df;
        masked(i, invalid) &= w;
        masked(i, invalid) ^= i >> 5;
        invalid = (i >= l);
    } while (any(active && invalid));

    return (i + p) % l;
}

NAMESPACE_END(mitsuba)
