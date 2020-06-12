#include <mitsuba/core/random.h>
#include <mitsuba/python/python.h>

MTS_PY_EXPORT(sample_tea) {
    MTS_PY_IMPORT_TYPES()
    m.def("sample_tea_float32",
          vectorize(sample_tea_float32<UInt32>),
          "v0"_a, "v1"_a, "rounds"_a = 4, D(sample_tea_float32));

    m.def("sample_tea_float64",
          vectorize(sample_tea_float64<UInt32>),
          "v0"_a, "v1"_a, "rounds"_a = 4, D(sample_tea_float64));

    m.attr("sample_tea_float") = m.attr(
        sizeof(Float) != sizeof(Float64) ? "sample_tea_float32" : "sample_tea_float64");

    m.def("latin_hypercube_1", vectorize(latin_hypercube<Float, mitsuba::PCG32<Float>>),
          "rng"_a, "sample_count"_a, "jitter"_a = true);
    m.def("latin_hypercube_2", vectorize(latin_hypercube<Point2f, mitsuba::PCG32<Float>>),
          "rng"_a, "sample_count"_a, "jitter"_a = true);
    m.def("latin_hypercube_3", vectorize(latin_hypercube<Point3f, mitsuba::PCG32<Float>>),
          "rng"_a, "sample_count"_a, "jitter"_a = true);
}