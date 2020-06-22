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

template <typename Float, typename Spectrum>
class MultijitterSampler final : public RandomSampler<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(RandomSampler, m_sample_count, m_base_seed, m_rng,
                    check_rng, m_samples_per_wavefront, m_wavefront_count)
    MTS_IMPORT_TYPES()

    MultijitterSampler(const Properties &props = Properties()) : Base(props) {
        m_jitter = props.bool_("jitter", true);

        // Find stratification grid resolution with aspect ratio close to 1
        m_resolution[1] = ScalarUInt32(sqrt(ScalarFloat(m_sample_count)));
        m_resolution[0] = (m_sample_count + m_resolution[1] - 1) / m_resolution[1];

        if (m_sample_count != hprod(m_resolution))
            Log(Warn, "Sample count rounded up to %i", hprod(m_resolution));

        m_sample_count = hprod(m_resolution);
        m_inv_sample_count = rcp(ScalarFloat(m_sample_count));
        m_inv_resolution   = rcp(ScalarPoint2f(m_resolution));
        m_resolution_x_div = m_resolution[0];

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
        sampler->m_resolution_x_div      = m_resolution_x_div;
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
            UInt32 sequence_seeds = gather<UInt64>(seed_value, sequence_idx);
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
        Assert(m_wavefront_index > -1);
        check_rng(active);

        UInt32 sample_indices = m_wavefront_index + m_wavefront_sample_offsets;
        UInt32 perm_seed = m_permutations_seed + m_dimension_index++;

        Float p = kensler_permute(sample_indices, m_sample_count, perm_seed * 0x45fbe943, active);
        Float j = m_jitter ? next_float<Float>(m_rng.get(), active) : 0.5f;

        return (p + j) * m_inv_sample_count;
    }

    Point2f next_2d(Mask active = true) override {
        Assert(m_wavefront_index > -1);
        check_rng(active);

        UInt32 sample_indices = m_wavefront_index + m_wavefront_sample_offsets;
        UInt32 perm_seed = m_permutations_seed + m_dimension_index++;

        UInt32 s = kensler_permute(sample_indices, m_sample_count, perm_seed * 0x51633e2d, active);

        UInt32 y = m_resolution_x_div(s);    // s / m_resolution.x()
        UInt32 x = s - y * m_resolution.x(); // s % m_resolution.x()

        UInt32 sx = kensler_permute(x, m_resolution.x(), perm_seed * 0x68bc21eb, active);
        UInt32 sy = kensler_permute(y, m_resolution.y(), perm_seed * 0x02e5be93, active);

        Float jx = 0.5f, jy = 0.5f;
        if (m_jitter) {
            jx = next_float<Float>(m_rng.get(), active);
            jy = next_float<Float>(m_rng.get(), active);
        }

        return Point2f((x + (sy + jx) * m_inv_resolution.y()) * m_inv_resolution.x(),
                       (y + (sx + jy) * m_inv_resolution.x()) * m_inv_resolution.y());
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

    /// Stratification grid resolution and precomputed variables
    ScalarPoint2u m_resolution;
    ScalarPoint2f m_inv_resolution;
    ScalarFloat m_inv_sample_count;
    enoki::divisor<ScalarUInt32> m_resolution_x_div;

    /// Sampler state
    ScalarUInt32 m_dimension_index;
    ScalarInt32  m_wavefront_index;

    UInt32 m_permutations_seed;
    UInt32 m_wavefront_sample_offsets;
};

MTS_IMPLEMENT_CLASS_VARIANT(MultijitterSampler, Sampler)
MTS_EXPORT_PLUGIN(MultijitterSampler, "Multijitter Sampler");
NAMESPACE_END(mitsuba)
