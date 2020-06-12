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

The stratified sample generator divides the domain into a discrete number
of strata and produces a sample within each one of them. This generally leads to less
sample clumping when compared to the independent sampler, as well as better
convergence. Due to internal storage costs, stratified samples are only provided up to a
certain dimension, after which independent sampling takes over.

 */

template <typename Float, typename Spectrum>
class StratifiedSampler final : public RandomSampler<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(RandomSampler, m_sample_count, m_base_seed, m_rng)
    MTS_IMPORT_TYPES()

    using UInt32Storage = DynamicBuffer<UInt32>;

    StratifiedSampler(const Properties &props = Properties()) : Base(props) {
        m_jitter = props.bool_("jitter", true);
        m_max_dimension = props.int_("dimension", 4);

        m_sample_count = math::round_to_power_of_two(m_sample_count);
        m_resolution = ScalarUInt32(enoki::sqrt(m_sample_count));
        m_inv_resolution = rcp(ScalarFloat(m_resolution));

        m_current_dimension = 0;
        m_sample_index = 0;
    }

    ref<Sampler<Float, Spectrum>> clone() override {
        StratifiedSampler *sampler = new StratifiedSampler();
        sampler->m_jitter = m_jitter;
        sampler->m_max_dimension = m_max_dimension;
        sampler->m_sample_count  = m_sample_count;
        sampler->m_resolution  = m_resolution;
        sampler->m_inv_resolution  = m_inv_resolution;
        sampler->m_base_seed = m_base_seed;
        sampler->m_current_dimension = 0;
        sampler->m_sample_index = 0;
        return sampler;
    }

    void seed(UInt64 seed_value) override {
        Base::seed(seed_value);

        // TODO Should be done in constructor for scalar modes
        m_permutations.resize(m_max_dimension);
        for (size_t i = 0; i < m_max_dimension; i++) {
            std::vector<Float> samples = latin_hypercube<Float>(m_rng.get(), m_sample_count, false);
            m_permutations[i].resize(m_sample_count);
            for (size_t j = 0; j < m_sample_count; j++)
                m_permutations[i][j] = UInt32(samples[j] * m_sample_count);
        }
    }

    void next_sample() override {
        m_current_dimension = 0;
        ++m_sample_index;
        Assert(m_sample_index < m_sample_count);
    }

    Float next_1d(Mask active = true) override {
        // TODO move this to parent class
        if constexpr (is_dynamic_array_v<Float>)  {
            if (m_rng == nullptr)
                Throw("Sampler::seed() must be invoked before using this sampler!");
            if (active.size() != 1 && active.size() != m_rng->state.size())
                Throw("Invalid mask size (%d), expected %d", active.size(), m_rng->state.size());
        }

        // TODO
        return next_float<Float>(m_rng.get(), active);
    }

    Point2f next_2d(Mask active = true) override {
        if (m_current_dimension < m_max_dimension) {
            UInt32 p = m_permutations[m_current_dimension++][m_sample_index];
            Float x = p % m_resolution + (m_jitter ? next_float<Float>(m_rng.get(), active) : 0.5f);
            Float y = p / m_resolution + (m_jitter ? next_float<Float>(m_rng.get(), active) : 0.5f);
            return Point2f(x, y) * m_inv_resolution;
        } else {
            Float x = next_float<Float>(m_rng.get(), active),
                  y = next_float<Float>(m_rng.get(), active);
            return Point2f(x, y);
        }
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "StratifiedSampler[" << std::endl
            << "  sample_count = " << m_sample_count << std::endl
            << "  max_dimension = " << m_max_dimension << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    bool m_jitter;
    ScalarUInt32 m_max_dimension;
    ScalarUInt32 m_current_dimension;
    ScalarUInt32 m_sample_index;
    ScalarUInt32 m_resolution;
    ScalarFloat m_inv_resolution;

    std::vector<std::vector<UInt32>> m_permutations;
};

MTS_IMPLEMENT_CLASS_VARIANT(StratifiedSampler, Sampler)
MTS_EXPORT_PLUGIN(StratifiedSampler, "Stratified Sampler");
NAMESPACE_END(mitsuba)
