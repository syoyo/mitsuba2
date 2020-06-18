#include <mitsuba/core/profiler.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/sampler.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sampler-multijitter:

Multijitter sampler (:monosp:`multijitter`)
-------------------------------------------

.. pluginparameters::

 * - sample_count
   - |int|
   - Number of samples per pixel (Default: 4)
 * - seed
   - |int|
   - Seed offset (Default: 0)
 * - jitter
   - |bool|
   - Adds additional random jitter withing the substratum (Default: True)

Based on https://graphics.pixar.com/library/MultiJitteredSampling/paper.pdf
 */

#define USE_KENSLER_PERMUTE

template <typename Float, typename Spectrum>
class MultijitterSampler final : public RandomSampler<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(RandomSampler, m_sample_count, m_base_seed, m_rng,
                    check_rng, m_samples_per_wavefront, m_wavefront_count)
    MTS_IMPORT_TYPES()

    MultijitterSampler(const Properties &props = Properties()) : Base(props) {
        m_jitter = props.bool_("jitter", true);

        // Make sure sample_count is power of two and square (e.g. 4, 16, 64, 256, 1024, ...)
        m_resolution = 2;
        while (sqr(m_resolution) < m_sample_count)
            m_resolution = math::round_to_power_of_two(++m_resolution);

        if (m_sample_count != sqr(m_resolution))
            Log(Warn, "Sample count should be square and power of two, rounding to %i", sqr(m_resolution));

        m_sample_count = sqr(m_resolution);
        m_inv_sample_count = rcp(ScalarFloat(m_sample_count));
        m_inv_resolution   = rcp(ScalarFloat(m_resolution));

        // Default
        m_samples_per_wavefront = 1;
        m_wavefront_count = m_sample_count;

        m_dimension_index = 0;
        m_wavefront_index = -1;
    }

    ref<Sampler<Float, Spectrum>> clone() override {
        MultijitterSampler *sampler = new MultijitterSampler();
        sampler->m_jitter                = m_jitter;
        sampler->m_sample_count          = m_sample_count;
        sampler->m_inv_sample_count      = m_inv_sample_count;
        sampler->m_resolution            = m_resolution;
        sampler->m_inv_resolution        = m_inv_resolution;
        sampler->m_samples_per_wavefront = m_samples_per_wavefront;
        sampler->m_wavefront_count       = m_wavefront_count;
        sampler->m_base_seed             = m_base_seed;
        sampler->m_dimension_index       = 0u;
        sampler->m_wavefront_index       = -1;
        return sampler;
    }

    void seed(UInt64 seed_value) override {
        ScopedPhase scope_phase(ProfilerPhase::SamplerSeed);
        Base::seed(seed_value);

        m_dimension_index = 0u;
        m_wavefront_index = -1;

        if constexpr (is_dynamic_array_v<Float>) {
            UInt32 indices = arange<UInt32>(seed_value.size());

            // Get the seed value of the first sample for every pixel
            UInt32 sequence_seeds = gather<UInt32>(seed_value, m_samples_per_wavefront * (indices / m_samples_per_wavefront));
            m_permutations_seed = sample_tea_32<UInt32>(UInt32(m_base_seed), sequence_seeds);

            m_wavefront_sample_offsets = indices % m_samples_per_wavefront;
        } else {
            m_permutations_seed = m_base_seed + seed_value;
            m_wavefront_sample_offsets = 0;
        }
    }

    void prepare_wavefront() override {
        m_dimension_index = 0u;
        m_wavefront_index++;
        Assert(m_wavefront_index < m_wavefront_count);
    }

    Float next_1d(Mask active = true) override {
        Assert(m_wavefront_index > -1);
        check_rng(active);

        UInt32 x = m_wavefront_index + m_wavefront_sample_offsets;

#ifdef USE_KENSLER_PERMUTE
        Float p = kensler_permute(x, m_sample_count, (m_permutations_seed + m_dimension_index++) * 0x45fbe943, active);
#else
        Float p = sample_permutation(x, m_sample_count, (m_permutations_seed + m_dimension_index++) * 0x45fbe943);
#endif

        if (m_jitter)
            p += next_float<Float>(m_rng.get(), active);
        else
            p += 0.5f;

        return p * m_inv_sample_count;
    }

    Point2f next_2d(Mask active = true) override {
        Assert(m_wavefront_index > -1);
        check_rng(active);

        UInt32 sample_indices = m_wavefront_index + m_wavefront_sample_offsets;

#ifdef USE_KENSLER_PERMUTE
        UInt32 s = kensler_permute(sample_indices, m_sample_count, (m_permutations_seed + m_dimension_index) * 0x51633e2d, active);
#else
        UInt32 s = sample_permutation(sample_indices, m_sample_count, (m_permutations_seed + m_dimension_index) * 0x51633e2d);
#endif

        UInt32 x = s % m_resolution,
               y = s / m_resolution;

#ifdef USE_KENSLER_PERMUTE
        Float sx = kensler_permute(x, m_resolution, (m_permutations_seed + m_dimension_index) * 0xa511e9b3, active);
        Float sy = kensler_permute(y, m_resolution, (m_permutations_seed + m_dimension_index) * 0x63d83595, active);
#else
        Float sx = sample_permutation(x, m_resolution, (m_permutations_seed + m_dimension_index) * 0xa511e9b3);
        Float sy = sample_permutation(y, m_resolution, (m_permutations_seed + m_dimension_index) * 0x63d83595);
#endif
        m_dimension_index++;

        Float jx, jy;
        if (m_jitter) {
            jx = next_float<Float>(m_rng.get(), active);
            jy = next_float<Float>(m_rng.get(), active);
        } else {
            jx = 0.5f;
            jy = 0.5f;
        }

        return Point2f((x + (sy + jx) * m_inv_resolution) * m_inv_resolution,
                       (y + (sx + jy) * m_inv_resolution) * m_inv_resolution);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "MultijitterSampler[" << std::endl
            << "  sample_count = " << m_sample_count << std::endl
            << "  jitter = " << m_jitter << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    bool m_jitter;

    /// Stratification grid resolution
    ScalarUInt32 m_resolution;
    ScalarFloat m_inv_resolution;
    ScalarFloat m_inv_sample_count;

    /// Sampler state
    ScalarUInt32 m_dimension_index;
    ScalarInt32  m_wavefront_index;

    UInt32 m_permutations_seed;
    UInt32 m_wavefront_sample_offsets;
};

MTS_IMPLEMENT_CLASS_VARIANT(MultijitterSampler, Sampler)
MTS_EXPORT_PLUGIN(MultijitterSampler, "Multijitter Sampler");
NAMESPACE_END(mitsuba)
