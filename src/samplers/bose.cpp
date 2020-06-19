#include <mitsuba/core/profiler.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/sampler.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sampler-bose:

Bose Orthogonal Array sampler (:monosp:`bose`)
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

Based on https://cs.dartmouth.edu/~wjarosz/publications/jarosz19orthogonal.pdf
 */

#define USE_KENSLER_PERMUTE

template <typename Float, typename Spectrum>
class BoseSampler final : public RandomSampler<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(RandomSampler, m_sample_count, m_base_seed, m_rng,
                    check_rng, m_samples_per_wavefront, m_wavefront_count)
    MTS_IMPORT_TYPES()

    BoseSampler(const Properties &props = Properties()) : Base(props) {
        m_jitter = props.bool_("jitter", true);

        // Make sure m_resolution is a prime number
        auto is_prime = [](uint32_t x) {
            for (uint32_t i = 2; i <= x / 2; ++i)
                if (x % i == 0)
                    return false;
            return true;
        };

        m_resolution = 2;
        while (sqr(m_resolution) < m_sample_count || !is_prime(m_resolution))
            m_resolution++;

        if (m_sample_count != sqr(m_resolution))
            Log(Warn, "Sample count should be the square of a prime number, rounding to %i", sqr(m_resolution));

        m_sample_count = sqr(m_resolution);

        // Default
        m_samples_per_wavefront = 1;
        m_wavefront_count = m_sample_count;

        m_dimension_index = 0;
        m_wavefront_index = -1;
    }

    ref<Sampler<Float, Spectrum>> clone() override {
        BoseSampler *sampler = new BoseSampler();
        sampler->m_jitter                = m_jitter;
        sampler->m_sample_count          = m_sample_count;
        sampler->m_resolution            = m_resolution;
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
            m_permutations_seed = sample_tea_32<UInt32>(m_base_seed, seed_value);
            m_wavefront_sample_offsets = 0;
        }
    }

    void prepare_wavefront() override {
        m_dimension_index = 0u;
        m_wavefront_index++;
        Assert(m_wavefront_index < m_wavefront_count);
    }

    Float bose_oa(UInt32 i,   // sample index
                  uint32_t j, // dimension
                  uint32_t s, // number of levels/stratas
                  UInt32 p,   // pseudo-random permutation seed
                  Mask active = true) {

        // Permutes the sample index so that samples are obtained in random order
        i = kensler_permute(i % (s * s), s * s, p, active);

        UInt32 a_i0 = i / s;
        UInt32 a_i1 = i % s;

        UInt32 a_ij, a_ik;
        if (j == 0) {
            a_ij = a_i0;
            a_ik = a_i1;
        } else if (j == 1) {
            a_ij = a_i1;
            a_ik = a_i0;
        } else {
            UInt32 k = (j % 2) ? j - 1 : j + 1;
            a_ij     = (a_i0 + (j - 1) * a_i1) % s;
            a_ik     = (a_i0 + (k - 1) * a_i1) % s;
        }

        UInt32 stratum     = kensler_permute(a_ij, s, p * (j + 1) * 0x51633e2d, active);
        // UInt32 sub_stratum = kensler_permute(a_ik, s, (a_ik * s + a_ij + 1) * p * (j + 1) * 0x68bc21eb, active); // J
        // UInt32 sub_stratum = kensler_permute(a_ik, s, (a_ij + 1) * p * (j + 1) * 0x68bc21eb, active); // MJ
        UInt32 sub_stratum = kensler_permute(a_ik, s, p * (j + 1) * 0x68bc21eb, active); // CMJ
        Float jitter = m_jitter ? next_float<Float>(m_rng.get(), active) : 0.5f;
        return (stratum + (sub_stratum + jitter) / s) / s;
    }

    Float next_1d(Mask active = true) override {
        Assert(m_wavefront_index > -1);
        check_rng(active);

        return bose_oa(m_wavefront_index + m_wavefront_sample_offsets,
                       m_dimension_index++,
                       m_resolution,
                       m_permutations_seed, active);
    }

    Point2f next_2d(Mask active = true) override {
        Assert(m_wavefront_index > -1);
        check_rng(active);

        Float f1 = next_1d(active),
              f2 = next_1d(active);
        return Point2f(f1, f2);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "BoseSampler[" << std::endl
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

    /// Sampler state
    ScalarUInt32 m_dimension_index;
    ScalarInt32  m_wavefront_index;

    UInt32 m_permutations_seed;
    UInt32 m_wavefront_sample_offsets;
};

MTS_IMPLEMENT_CLASS_VARIANT(BoseSampler, Sampler)
MTS_EXPORT_PLUGIN(BoseSampler, "Bose OA Sampler");
NAMESPACE_END(mitsuba)
