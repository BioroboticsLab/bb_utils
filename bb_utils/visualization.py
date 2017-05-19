import numpy as np
import cairocffi as cairo
from io import BytesIO
from more_itertools import pairwise


class TagArtist:
    def __init__(self,
                 surface_width=256,
                 surface_height=256,
                 tag_radius=.48,
                 background_color=(.9, .9, .9, 1),
                 guideline_width=.001):
        self.surface_width = surface_width
        self.surface_height = surface_height
        self.tag_radius = tag_radius
        self.inner_radius = tag_radius * 1.1 / 3
        self.outer_radius = tag_radius * .9
        self.background_color = background_color
        self.guideline_width = guideline_width

    def draw(self, bits_12):
        bits = 1 - np.roll(bits_12, -3)

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                                     self.surface_width,
                                     self.surface_height)
        ctx = cairo.Context(surface)
        ctx.scale(self.surface_width, self.surface_height)

        ctx.set_source_rgba(*self.background_color)
        ctx.rectangle(0, 0, 1, 1)
        ctx.fill()

        center = np.array([.5, .5])
        ctx.move_to(center[0], center[1])
        ctx.set_source_rgba(1, 1, 1)
        ctx.arc(center[0], center[1], self.tag_radius, 0, 2 * np.pi)
        ctx.close_path()
        ctx.fill()

        for idx, (theta_start, theta_end) in enumerate(pairwise(np.linspace(0, 2 * np.pi, num=13))):
            vec = center + np.array([np.cos(theta_start), np.sin(theta_start)]) * \
                self.outer_radius

            ctx.set_source_rgb(bits[idx], bits[idx], bits[idx])
            ctx.set_line_width(0.)
            ctx.move_to(center[0], center[1])
            ctx.line_to(vec[0], vec[1])
            ctx.arc(center[0], center[1], self.outer_radius, theta_start, theta_end)
            ctx.line_to(center[0], center[1])
            ctx.close_path()
            ctx.fill()

            ctx.move_to(center[0], center[1])
            ctx.set_source_rgb(.5, .5, .5)
            ctx.set_line_width(self.guideline_width)
            ctx.line_to(vec[0], vec[1])
            ctx.close_path()
            ctx.stroke()

        ctx.set_source_rgba(.5, .5, .5)
        ctx.arc(center[0], center[1], self.outer_radius, 0, 2 * np.pi)
        ctx.close_path()
        ctx.set_line_width(self.guideline_width)
        ctx.stroke()

        ctx.move_to(center[0], center[1])
        ctx.set_source_rgba(1, 1, 1)
        ctx.arc(center[0], center[1], self.inner_radius, np.pi, 2 * np.pi)
        ctx.close_path()
        ctx.fill()

        ctx.move_to(center[0], center[1])
        ctx.set_source_rgba(0, 0, 0)
        ctx.arc(center[0], center[1], self.inner_radius, 0, np.pi)
        ctx.close_path()
        ctx.fill()

        ctx.set_source_rgba(.5, .5, .5)
        ctx.arc(center[0], center[1], self.inner_radius, 0, 2 * np.pi)
        ctx.close_path()
        ctx.set_line_width(self.guideline_width)
        ctx.stroke()

        b = BytesIO()

        surface.write_to_png(b)
        b.seek(0)

        return b.read()
