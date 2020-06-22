#include <mitsuba/core/profiler.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/qmc.h>
#include <mitsuba/render/sampler.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sampler-ldsampler:

Low discrepancy sampler (:monosp:`ldsampler`)
-------------------------------------------

This plugin implements a simple hybrid sampler that combines aspects of a Quasi-Monte Carlo
sequence with a pseudorandom number generator based on a technique proposed by Kollig and
Keller \cite{Kollig2002Efficient}. It is a good and fast general-purpose sample generator and
therefore chosen as the default option in Mitsuba. Some of the QMC samplers in the following pages
can generate even better distributed samples, but this comes at a higher cost in terms of
performance.

Based on https://github.com/mitsuba-renderer/mitsuba/blob/master/src/samplers/ldsampler.cpp

 */

template <typename Float, typename Spectrum>
class LowDiscrepancySampler  final : public RandomSampler<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(RandomSampler, m_sample_count, m_base_seed,
                    m_samples_per_wavefront, m_wavefront_count, wavefront_size)
    MTS_IMPORT_TYPES()

    LowDiscrepancySampler (const Properties &props = Properties()) : Base(props) {
        // Make sure sample_count is power of two and square (e.g. 4, 16, 64, 256, 1024, ...)
        ScalarUInt32 res = 2;
        while (sqr(res) < m_sample_count)
            res = math::round_to_power_of_two(++res);

        if (m_sample_count != sqr(res))
            Log(Warn, "Sample count should be square and power of two, rounding to %i", sqr(res));

        m_sample_count = sqr(res);

        // Default
        m_samples_per_wavefront = 1;
        m_wavefront_count = m_sample_count;

        m_dimension_index = 0;
        m_wavefront_index = -1;
    }

    ref<Sampler<Float, Spectrum>> clone() override {
        LowDiscrepancySampler  *sampler = new LowDiscrepancySampler ();
        sampler->m_sample_count          = m_sample_count;
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
            UInt32 sequence_idx = m_samples_per_wavefront * (indices / m_samples_per_wavefront);
            UInt32 sequence_seeds = gather<UInt32>(seed_value, sequence_idx);
            m_permutations_seed = sample_tea_32<UInt32>(UInt32(m_base_seed), sequence_seeds);

            m_wavefront_sample_offsets = indices % UInt32(m_samples_per_wavefront);
        } else {
            m_permutations_seed = sample_tea_32<UInt32>(m_base_seed, seed_value);
            m_wavefront_sample_offsets = 0;
        }
    }

    void prepare_wavefront() override {
        m_dimension_index = 0u;
        m_wavefront_index++;
        Assert(m_wavefront_index < m_wavefront_count);
    }

    Float next_1d(Mask active = true) override {
        ENOKI_MARK_USED(active);
        Assert(m_wavefront_index > -1);

        UInt32 sample_indices = m_wavefront_index + m_wavefront_sample_offsets;
        UInt32 i = permute(sample_indices, m_sample_count, m_permutations_seed + m_dimension_index);
        UInt32 scramble = sample_tea_32<UInt32>(m_permutations_seed, m_dimension_index++);

        return radical_inverse_2(i, scramble);
    }

    Point2f next_2d(Mask active = true) override {
        ENOKI_MARK_USED(active);
        Assert(m_wavefront_index > -1);

        UInt32 sample_indices = m_wavefront_index + m_wavefront_sample_offsets;
        UInt32 i = permute(sample_indices, m_sample_count, m_permutations_seed + m_dimension_index);
        UInt32 scramble_x = sample_tea_32<UInt32>(m_permutations_seed, m_dimension_index * 0x68bc21eb);
        UInt32 scramble_y = sample_tea_32<UInt32>(m_permutations_seed, m_dimension_index * 0x51633e2d);
        m_dimension_index++;

        Float x = radical_inverse_2(i, scramble_x),
              y = sobol_2(i, scramble_y);

        return Point2f(x, y);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "LowDiscrepancySampler [" << std::endl
            << "  sample_count = " << m_sample_count << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    /// Sampler state
    ScalarUInt32 m_dimension_index;
    ScalarInt32  m_wavefront_index;

    UInt32 m_permutations_seed;
    UInt32 m_wavefront_sample_offsets;
};

MTS_IMPLEMENT_CLASS_VARIANT(LowDiscrepancySampler , Sampler)
MTS_EXPORT_PLUGIN(LowDiscrepancySampler , "Low Discrepancy Sampler");
NAMESPACE_END(mitsuba)
