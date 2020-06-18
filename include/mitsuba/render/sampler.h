#pragma once

#include <mitsuba/mitsuba.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/core/random.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER Sampler : public Object {
public:
    MTS_IMPORT_TYPES()

    /**
     * \brief Create a clone of this sampler
     *
     * The clone is allowed to be different to some extent, e.g. a pseudorandom
     * generator should be based on a different random seed compared to the
     * original. All other parameters are copied exactly.
     *
     * May throw an exception if not supported. Cloning may also change the
     * state of the original sampler (e.g. by using the next 1D sample as a
     * seed for the clone).
     */
    virtual ref<Sampler> clone() = 0;

    /**
     * \brief Deterministically seed the underlying RNG, if applicable.
     *
     * In the context of wavefront ray tracing & dynamic arrays, this function
     * must be called with a \c seed_value matching the size of the wavefront.
     */
    virtual void seed(UInt64 seed_value);

    /// Start the next wavefront
    virtual void prepare_wavefront() { /*no op*/ }

    /// Retrieve the next component value from the current sample
    virtual Float next_1d(Mask active = true);

    /// Retrieve the next two component values from the current sample
    virtual Point2f next_2d(Mask active = true);

    /// Return the number of samples per pixel
    uint32_t sample_count() const { return m_sample_count; }

    /// Return the size of the wavefront (or 0, if not seeded)
    virtual uint32_t wavefront_size() const = 0;

    /// Set the number of samples per pass in the wavefront modes (default is 1)
    void set_samples_per_wavefront(uint32_t samples_per_wavefront) {
        m_samples_per_wavefront = samples_per_wavefront;
        m_wavefront_count = m_sample_count / m_samples_per_wavefront;
    };

    MTS_DECLARE_CLASS()
protected:
    Sampler(const Properties &props);
    virtual ~Sampler();

protected:
    uint32_t m_sample_count;
    uint32_t m_samples_per_wavefront;
    uint32_t m_wavefront_count;
    ScalarUInt64 m_base_seed;
};


template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER RandomSampler : public Sampler<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sampler, m_sample_count, m_base_seed)
    MTS_IMPORT_TYPES()
    using PCG32 = mitsuba::PCG32<UInt32>;

    virtual void seed(UInt64 seed_value) override;
    virtual uint32_t wavefront_size() const override;

    // Check that RNG is valid and initialized with the right size
    void check_rng(Mask active);

    MTS_DECLARE_CLASS()
protected:
    RandomSampler(const Properties &props);

protected:
    std::unique_ptr<PCG32> m_rng;
};

MTS_EXTERN_CLASS_RENDER(Sampler)
MTS_EXTERN_CLASS_RENDER(RandomSampler)
NAMESPACE_END(mitsuba)
