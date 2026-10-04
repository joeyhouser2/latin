"""Generate the Latin Library logo: SVG for the browser, .ico for the desktop.

One geometry, two renderers. The mark is defined once as a list of primitive
shapes (rounded rects, polygons, ellipses) in a 64x64 grid; ``to_svg`` writes
vector output for the page and favicon, ``to_image`` draws the same shapes with
Pillow for the Windows shortcut icon. Doing it this way rather than hand-keeping
an SVG and a PNG in sync matters because the two live in different places --
web/static and a .lnk on the Desktop -- and a logo that quietly diverges between
them is worse than no logo.

No SVG rasteriser is involved (cairosvg needs cairo DLLs on Windows). Pillow
draws the primitives directly, at 8x supersampling for antialiasing.

    python scripts/make_logo.py              # write the real assets
    python scripts/make_logo.py --sheet      # also write a comparison sheet of
                                             # every concept at 256/64/32/16 px

Concepts live in CONCEPTS; DEFAULT_CONCEPT is what ships. To change the logo,
change that and re-run -- then re-run scripts/install_shortcut.ps1 so the
Desktop shortcut picks up the new icon (Windows caches by path + timestamp).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "web", "static")

# The palette is the manuscript one, and it is not arbitrary: parchment ground,
# iron-gall ink for structure, and vermilion for the initial -- scribes
# rubricated initials in red precisely so a reader could find the start of a
# text at a glance, which is the same job a favicon does in a row of tabs.
PARCHMENT = "#f2e6d0"
PARCHMENT_EDGE = "#e2d1b2"
INK = "#2e2419"
RUBRIC = "#a8321f"
GOLD = "#c8a02c"

GRID = 64.0          # all coordinates are in a 64x64 box

Shape = Dict[str, Any]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def tile(fill: str = PARCHMENT, stroke: str = PARCHMENT_EDGE) -> List[Shape]:
    """The parchment ground every concept sits on.

    A filled tile rather than a transparent glyph so the mark survives both a
    white and a very dark browser tab strip without a second asset.
    """
    return [
        {"kind": "rrect", "x": 2, "y": 2, "w": 60, "h": 60, "r": 13,
         "fill": fill, "stroke": stroke, "width": 1.2},
    ]


def keyline(color: str = RUBRIC, inset: float = 6.5, width: float = 1.1) -> Shape:
    """The rubricated frame drawn inside an illuminated initial's field."""
    return {"kind": "rrect", "x": inset, "y": inset, "w": GRID - 2 * inset,
            "h": GRID - 2 * inset, "r": 8.5, "fill": None,
            "stroke": color, "width": width}


# A Roman capital L with bracketed wedge serifs, traced clockwise from the
# top-left of the head serif. Straight-line brackets rather than curves: at 16px
# the curve is invisible, and straight segments render identically in both
# renderers without a path-flattening step.
def versal_l(x: float = 0.0, scale: float = 1.0) -> List[Tuple[float, float]]:
    """A Roman capital L with bracketed wedge serifs, traced clockwise.

    Straight-line brackets rather than curves: at 16px the curve is invisible,
    and straight segments render identically in both renderers without a
    path-flattening step. The arm's terminal serif is kept shallow -- at this
    stroke weight a full-height one stops reading as a serif and starts reading
    as a notch cut out of the arm.
    """
    pts = [
        (20.0, 13.0), (38.0, 13.0), (38.0, 16.5), (33.5, 18.0),   # head serif
        (33.5, 44.2),                                              # stem, right side
        (46.2, 44.2), (47.4, 41.9), (50.0, 41.9),                  # arm + up-serif
        (50.0, 51.0), (20.0, 51.0),                                # baseline
        (20.0, 47.6), (24.5, 45.6),                                # foot serif
        (24.5, 18.0), (20.0, 16.5),                                # stem, left side
    ]
    if scale == 1.0 and x == 0.0:
        return pts
    # Scale about the glyph's own centre, then translate.
    cx, cy = 35.0, 32.0
    return [(cx + (px - cx) * scale + x, cy + (py - cy) * scale) for px, py in pts]


VERSAL_L = versal_l()


def concept_versal() -> Dict[str, List[Shape]]:
    """An illuminated initial: rubric L in a ruled field, with gold corner dots."""
    shapes = tile()
    shapes.append(keyline())
    shapes.append({"kind": "poly", "points": VERSAL_L, "fill": RUBRIC})
    # Gold dots in the corners of the field -- the cheap illumination a working
    # scribe actually used, as opposed to full gilding.
    for cx, cy in ((11.5, 11.5), (52.5, 11.5), (11.5, 52.5), (52.5, 52.5)):
        shapes.append({"kind": "ellipse", "cx": cx, "cy": cy, "rx": 1.7, "ry": 1.7,
                       "fill": GOLD})
    return {"full": shapes, "small": tile() + [
        {"kind": "poly", "points": VERSAL_L, "fill": RUBRIC}]}


def concept_scriptorium() -> Dict[str, List[Shape]]:
    """The initial *and* the text it opens -- a page, not just a letter.

    A rubric L on the left with ruled ink lines flowing to its right, which is
    how an initial actually appears on a manuscript page: the big coloured
    letter is the door into the text block. The small variant drops the lines
    and centres the letter, because below ~24px the ruling is grey fuzz that
    only costs the L its contrast.
    """
    shapes = tile()
    shapes.append({"kind": "poly", "points": versal_l(x=-12.0, scale=0.80),
                   "fill": RUBRIC})
    # Ruled text to the right of the initial: three lines, the last one short
    # and lighter -- a paragraph that ends, rather than a barcode. They stop
    # above the L's arm, which then runs on underneath them, the way a versal's
    # foot actually extends under the opening lines of its text block.
    for i, y in enumerate((20.5, 27.5, 34.5)):
        w = (19.0, 19.0, 12.0)[i]
        shapes.append({"kind": "rrect", "x": 34.0, "y": y, "w": w, "h": 2.2,
                       "r": 1.1, "fill": INK if i < 2 else "#7d6e58"})
    return {"full": shapes, "small": tile() + [
        {"kind": "poly", "points": VERSAL_L, "fill": RUBRIC}]}


def concept_codex() -> Dict[str, List[Shape]]:
    """An open book whose two pages are the reader's two columns.

    The left page is ruled in dense ink and the right in lighter, shorter
    lines: Latin on the left, its translation on the right, which is literally
    what the application shows.
    """
    shapes = tile()
    # Page blocks, splayed slightly from a central gutter.
    shapes.append({"kind": "poly", "fill": "#fbf5e9", "stroke": INK, "width": 1.1,
                   "points": [(8.0, 18.0), (31.0, 21.5), (31.0, 50.0), (8.0, 46.0)]})
    shapes.append({"kind": "poly", "fill": "#fbf5e9", "stroke": INK, "width": 1.1,
                   "points": [(33.0, 21.5), (56.0, 18.0), (56.0, 46.0), (33.0, 50.0)]})
    shapes.append({"kind": "rect", "x": 31.0, "y": 21.0, "w": 2.0, "h": 29.0,
                   "fill": INK})
    # Ruled lines: left dense (the source), right sparser (the translation).
    for i, y in enumerate((27.0, 31.0, 35.0, 39.0, 43.0)):
        shapes.append({"kind": "rect", "x": 12.0, "y": y - 0.7,
                       "w": 15.0 if i != 4 else 9.0, "h": 1.5, "fill": INK})
        shapes.append({"kind": "rect", "x": 36.5, "y": y - 0.6,
                       "w": 13.0 if i % 2 == 0 else 8.0, "h": 1.3, "fill": "#8a7a64"})
    # The rubric initial that opens the left page.
    shapes.append({"kind": "rect", "x": 12.0, "y": 24.0, "w": 3.4, "h": 3.4,
                   "fill": RUBRIC})
    # Small variant: drop the ruling, keep the book. Five hairlines a page is
    # four lines of grey at 16px.
    small = [s for s in shapes if not (s["kind"] == "rect" and s.get("h") in (1.5, 1.3))]
    return {"full": shapes, "small": small}


def concept_pilcrow() -> Dict[str, List[Shape]]:
    """The capitulum -- the scribe's paragraph mark, in rubric on parchment.

    Unusual as a logo and unmistakable as a mark of text scholarship: this is
    the sign that told a reader where one piece of argument stopped and the
    next began, centuries before anyone thought to indent a line.

    Built the way the glyph actually is: a filled bowl closed against a main
    stem, a second stem beside it, and a bar across the top joining the two.
    (The first pass drew one fat half-disc and a slab, which rendered as a
    blob with legs.)
    """
    bowl_cx, top, bottom = 28.0, 13.0, 51.0
    stem1_x, stem2_x, stem_w = 33.0, 41.0, 5.0
    shapes = tile()
    shapes.append(keyline(color=INK, inset=7.0, width=1.0))
    shapes.append({"kind": "ellipse", "cx": bowl_cx, "cy": 23.0, "rx": 10.0,
                   "ry": 10.0, "fill": RUBRIC})
    shapes.append({"kind": "rect", "x": bowl_cx, "y": top, "w": 10.0, "h": 20.0,
                   "fill": RUBRIC})
    shapes.append({"kind": "rect", "x": stem1_x, "y": top, "w": stem_w,
                   "h": bottom - top, "fill": RUBRIC})
    shapes.append({"kind": "rect", "x": stem2_x, "y": top, "w": stem_w,
                   "h": bottom - top, "fill": RUBRIC})
    # The bar that ties the second stem to the bowl along the top.
    shapes.append({"kind": "rect", "x": stem1_x, "y": top, "w": stem2_x - stem1_x,
                   "h": 5.0, "fill": RUBRIC})
    return {"full": shapes, "small": [s for s in shapes
                                      if not (s["kind"] == "rrect"
                                              and s.get("fill") is None)]}


CONCEPTS = {
    "versal": concept_versal,
    "scriptorium": concept_scriptorium,
    "codex": concept_codex,
    "pilcrow": concept_pilcrow,
}
DEFAULT_CONCEPT = "versal"


# ---------------------------------------------------------------------------
# renderers
# ---------------------------------------------------------------------------

def to_svg(shapes: List[Shape], size: int = 64, title: str = "Latin Library") -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" '
           f'width="{size}" height="{size}" role="img" aria-label="{title}">',
           f'<title>{title}</title>']
    for s in shapes:
        fill = s.get("fill") or "none"
        stroke = s.get("stroke")
        attrs = f'fill="{fill}"'
        if stroke:
            attrs += f' stroke="{stroke}" stroke-width="{s.get("width", 1)}"'
        if s["kind"] == "rrect":
            out.append(f'<rect x="{s["x"]}" y="{s["y"]}" width="{s["w"]}" '
                       f'height="{s["h"]}" rx="{s["r"]}" ry="{s["r"]}" {attrs}/>')
        elif s["kind"] == "rect":
            out.append(f'<rect x="{s["x"]}" y="{s["y"]}" width="{s["w"]}" '
                       f'height="{s["h"]}" {attrs}/>')
        elif s["kind"] == "poly":
            pts = " ".join(f"{x},{y}" for x, y in s["points"])
            out.append(f'<polygon points="{pts}" stroke-linejoin="round" {attrs}/>')
        elif s["kind"] == "ellipse":
            out.append(f'<ellipse cx="{s["cx"]}" cy="{s["cy"]}" rx="{s["rx"]}" '
                       f'ry="{s["ry"]}" {attrs}/>')
    out.append("</svg>")
    return "\n".join(out)


def to_image(shapes: List[Shape], size: int = 256, ss: int = 8) -> Image.Image:
    """Draw the same shapes with Pillow, supersampled then downscaled.

    Pillow has no antialiasing on its primitives, so a 16px icon drawn directly
    is a staircase. Drawing 8x large and using LANCZOS to come back down is the
    whole trick.
    """
    big = size * ss
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    k = big / GRID

    def sc(v: float) -> float:
        return v * k

    for s in shapes:
        fill = s.get("fill")
        stroke = s.get("stroke")
        width = max(1, int(round(s.get("width", 1) * k)))
        if s["kind"] == "rrect":
            draw.rounded_rectangle(
                [sc(s["x"]), sc(s["y"]), sc(s["x"] + s["w"]), sc(s["y"] + s["h"])],
                radius=sc(s["r"]), fill=fill, outline=stroke,
                width=width if stroke else 0)
        elif s["kind"] == "rect":
            draw.rectangle(
                [sc(s["x"]), sc(s["y"]), sc(s["x"] + s["w"]), sc(s["y"] + s["h"])],
                fill=fill, outline=stroke, width=width if stroke else 0)
        elif s["kind"] == "poly":
            pts = [(sc(x), sc(y)) for x, y in s["points"]]
            draw.polygon(pts, fill=fill, outline=stroke,
                         width=width if stroke else 0)
        elif s["kind"] == "ellipse":
            draw.ellipse([sc(s["cx"] - s["rx"]), sc(s["cy"] - s["ry"]),
                          sc(s["cx"] + s["rx"]), sc(s["cy"] + s["ry"])],
                         fill=fill, outline=stroke, width=width if stroke else 0)
    return img.resize((size, size), Image.LANCZOS)


ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def write_assets(concept: str) -> List[str]:
    art = CONCEPTS[concept]()
    shapes, small = art["full"], art["small"]
    written = []

    svg_path = os.path.join(STATIC, "logo.svg")
    with open(svg_path, "w", encoding="utf-8") as fh:
        fh.write(to_svg(shapes))
    written.append(svg_path)

    # The favicon uses the concept's own small variant -- each one decides for
    # itself what to shed below ~24px, because what survives at that size is
    # different for a letterform than for a book.
    fav_path = os.path.join(STATIC, "favicon.svg")
    with open(fav_path, "w", encoding="utf-8") as fh:
        fh.write(to_svg(small, title="Latin Library"))
    written.append(fav_path)

    # One .ico carrying every size Windows asks for: 16px in the taskbar, 256px
    # on the Desktop at large-icon settings. Small sizes use the simplified art.
    ico_path = os.path.join(STATIC, "latin-library.ico")
    frames = [to_image(small if s <= 32 else shapes, s) for s in ICO_SIZES]
    frames[-1].save(ico_path, format="ICO",
                    sizes=[(s, s) for s in ICO_SIZES], append_images=frames[:-1])
    written.append(ico_path)

    png_path = os.path.join(STATIC, "logo-256.png")
    to_image(shapes, 256).save(png_path)
    written.append(png_path)
    return written


def write_sheet(path: str) -> str:
    """A contact sheet: every concept at the sizes that decide a logo."""
    sizes = (256, 64, 32, 16)
    pad, label_h = 24, 0
    width = pad + sum(s + pad for s in sizes)
    height = pad + len(CONCEPTS) * (256 + pad) + label_h
    sheet = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    for row, (name, fn) in enumerate(CONCEPTS.items()):
        art = fn()
        y = pad + row * (256 + pad)
        x = pad
        for s in sizes:
            # Below 33px show the small variant: the point of the sheet is to
            # compare what you will really see, not four scalings of one image.
            sheet.alpha_composite(to_image(art["full"] if s > 32 else art["small"], s),
                                  (x, y + (256 - s) // 2))
            x += s + pad
    sheet.convert("RGB").save(path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", default=DEFAULT_CONCEPT, choices=sorted(CONCEPTS))
    ap.add_argument("--sheet", default="", metavar="PATH",
                    help="also write a comparison sheet of every concept here")
    args = ap.parse_args()

    for path in write_assets(args.concept):
        print("wrote", os.path.relpath(path))
    if args.sheet:
        print("wrote", write_sheet(args.sheet))


if __name__ == "__main__":
    main()
