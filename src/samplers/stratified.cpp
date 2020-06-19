#include <mitsuba/core/profiler.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/sampler.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sampler-stratified:

Stratified sampler (:monosp:`stratified`)
-------------------------------------------

.. pluginparameters::

 * - sample_count
   - |int|
   - Number of samples per pixel (Default: 4)
 * - dimension
   - |int|
   - Effective dimension, up to which stratified samples are provided. (Default: 4)
 * - seed
   - |int|
   - Seed offset (Default: 0)
 * - jitter
   - |bool|
   - Adds additional random jitter withing the stratum (Default: True)

The stratified sample generator divides the domain into a discrete number of strata and produces
a sample within each one of them. This generally leads to less sample clumping when compared to
the independent sampler, as well as better convergence. Due to internal storage costs, stratified
samples are only provided up to a certain dimension, after which independent sampling takes over.

 */

template <typename Float, typename Spectrum>
class StratifiedSampler final : public RandomSampler<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(RandomSampler, m_sample_count, m_base_seed, m_rng,
                    check_rng, m_samples_per_wavefront, m_wavefront_count, wavefront_size)
    MTS_IMPORT_TYPES()

    StratifiedSampler(const Properties &props = Properties()) : Base(props) {
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
        StratifiedSampler *sampler = new StratifiedSampler();
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
        Assert(m_wavefront_index > -1);
        check_rng(active);

        UInt32 sample_indices = m_wavefront_index + m_wavefront_sample_offsets;
        Float p = sample_permutation(sample_indices,
                                     m_sample_count,
                                     m_permutations_seed + m_dimension_index++);

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
        UInt32 p = sample_permutation(sample_indices,
                                      m_sample_count,
                                      m_permutations_seed + m_dimension_index++);

        Float x = p % m_resolution,
              y = p / m_resolution;

        if (m_jitter) {
            x += next_float<Float>(m_rng.get(), active);
            y += next_float<Float>(m_rng.get(), active);
        } else {
            x += 0.5f;
            y += 0.5f;
        }

        return Point2f(x, y) * m_inv_resolution;
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "StratifiedSampler[" << std::endl
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

MTS_IMPLEMENT_CLASS_VARIANT(StratifiedSampler, Sampler)
MTS_EXPORT_PLUGIN(StratifiedSampler, "Stratified Sampler");
NAMESPACE_END(mitsuba)
