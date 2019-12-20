import os

import numpy as np

import enoki as ek
import mitsuba
from mitsuba.gpu_rgb.core import Thread, xml
from mitsuba.gpu_rgb.core.xml import load_file
from mitsuba.gpu_rgb.render import (BSDF, BSDFContext, BSDFFlags,
                                    DirectionSample3f, Emitter, ImageBlock,
                                    SamplingIntegrator, has_flag,
                                    register_integrator)


def mis_weight(pdf_a, pdf_b):
    pdf_a *= pdf_a
    pdf_b *= pdf_b
    return ek.select(pdf_a > 0.0, pdf_a / (pdf_a + pdf_b), ek.FloatC(0.0))


def integrator_sample(scene, sampler, rays, active=True):
    si = scene.ray_intersect(rays)
    active = si.is_valid() & active

    # Visible emitters
    emitter_vis = si.emitter(scene, active)
    result = ek.select(active, Emitter.eval_vec(emitter_vis, si, active), ek.Vector3fC(0.0))

    ctx = BSDFContext()
    bsdf = si.bsdf(rays)

    # Emitter sampling
    sample_emitter = active & has_flag(BSDF.flags_vec(bsdf), BSDFFlags.Smooth)
    ds, emitter_val = scene.sample_emitter_direction(si, sampler.next_2d(sample_emitter), True, sample_emitter)
    active_e = sample_emitter & ek.neq(ds.pdf, 0.0)
    wo = si.to_local(ds.d)
    bsdf_val = BSDF.eval_vec(bsdf, ctx, si, wo, active_e)
    bsdf_pdf = BSDF.pdf_vec(bsdf, ctx, si, wo, active_e)
    mis = ek.select(ds.delta, ek.FloatC(1), mis_weight(ds.pdf, bsdf_pdf))
    result += ek.select(active_e, emitter_val * bsdf_val * mis, ek.Vector3fC(0))

    # BSDF sampling
    active_b = active
    bs, bsdf_val = BSDF.sample_vec(bsdf, ctx, si, sampler.next_1d(active), sampler.next_2d(active), active_b)
    si_bsdf = scene.ray_intersect(si.spawn_ray(si.to_world(bs.wo)), active_b)
    emitter = si_bsdf.emitter(scene, active_b)
    active_b &= ek.neq(emitter, 0)
    emitter_val = Emitter.eval_vec(emitter, si_bsdf, active_b)
    delta = has_flag(bs.sampled_type, BSDFFlags.Delta)
    ds = DirectionSample3f(si_bsdf, si)
    ds.object = emitter
    emitter_pdf = ek.select(delta, ek.FloatC(0), scene.pdf_emitter_direction(si, ds, active_b))
    result += ek.select(active_b, bsdf_val * emitter_val * mis_weight(bs.pdf, emitter_pdf), ek.Vector3fC(0))
    return result, si.is_valid(), ek.select(si.is_valid(), si.t, ek.FloatC(0.0))


class MyDirectIntegrator(SamplingIntegrator):
    def __init__(self, props):
        SamplingIntegrator.__init__(self, props)

    def sample(self, scene, sampler, ray, active):
        result, is_valid, depth = integrator_sample(scene, sampler, ray, active)
        return result, is_valid, [depth]

    def aov_names(self):
        return ["depth.Y"]

    def to_string(self):
        return "MyDirectIntegrator[]"


# Register our integrator such that the XML file loader can instantiate it when loading a scene
register_integrator("mydirectintegrator", lambda props: MyDirectIntegrator(props))

SCENE_DIR = '../../../resources/data/scenes/'

# Load an XML file which specifies "mydirectintegrator" as the scene's integrator
filename = os.path.join(SCENE_DIR, 'cbox/cbox-custom-integrator.xml')
directory_name = os.path.dirname(filename)
Thread.thread().file_resolver().append(directory_name)
scene = load_file(filename)

scene.integrator().render(scene, scene.sensors()[0])

film = scene.sensors()[0].film()
film.set_destination_file('my-direct-integrator.exr')
film.develop()
