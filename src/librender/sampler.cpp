#include <mitsuba/render/sampler.h>
#include <mitsuba/core/properties.h>

NAMESPACE_BEGIN(mitsuba)

MTS_VARIANT Sampler<Float, Spectrum>::Sampler(const Properties &props) {
    m_sample_count = props.size_("sample_count", 4);
    m_base_seed = props.size_("seed", 0);
}

MTS_VARIANT Sampler<Float, Spectrum>::~Sampler() { }

MTS_VARIANT void Sampler<Float, Spectrum>::seed(UInt64) { NotImplementedError("seed"); }

MTS_VARIANT Float Sampler<Float, Spectrum>::next_1d(Mask) { NotImplementedError("next_1d"); }

MTS_VARIANT typename Sampler<Float, Spectrum>::Point2f Sampler<Float, Spectrum>::next_2d(Mask) {
    NotImplementedError("next_2d");
}

MTS_VARIANT RandomSampler<Float, Spectrum>::RandomSampler(const Properties &props) : Base(props) {
    /* Can't seed yet on the GPU because we don't know yet
        how many entries will be needed. */
    if (!is_dynamic_array_v<Float>)
        seed(PCG32_DEFAULT_STATE);
}

MTS_VARIANT void RandomSampler<Float, Spectrum>::seed(UInt64 seed_value) {
    if (!m_rng)
        m_rng = std::make_unique<PCG32>();

    seed_value += m_base_seed;

    if constexpr (is_dynamic_array_v<Float>) {
        UInt64 idx = arange<UInt64>(seed_value.size());
        m_rng->seed(sample_tea_64(seed_value, idx),
                    sample_tea_64(idx, seed_value));
    } else {
        m_rng->seed(seed_value, PCG32_DEFAULT_STREAM + arange<UInt64>());
    }
}

MTS_VARIANT size_t RandomSampler<Float, Spectrum>::wavefront_size() const {
    if (m_rng == nullptr)
        return 0;
    else
        return enoki::slices(m_rng->state);
}

MTS_IMPLEMENT_CLASS_VARIANT(Sampler, Object, "sampler")
MTS_IMPLEMENT_CLASS_VARIANT(RandomSampler, Sampler, "random sampler")

MTS_INSTANTIATE_CLASS(Sampler)
MTS_INSTANTIATE_CLASS(RandomSampler)
NAMESPACE_END(mitsuba)
