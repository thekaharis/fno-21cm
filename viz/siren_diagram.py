"""Draw the SIREN weight-network schematic (figures/diagrams/siren_mechanism.svg).

Styled after the standard FNO architecture figure so the two can sit side by
side in the thesis: top row = the trunk of ``siren.SirenWeightNetwork``,
zoom-in = one ``siren.SineLayer``.

    python3 viz/siren_diagram.py
"""

import math, pathlib

W, H = 2000, 1000
SERIF = "'Latin Modern Roman','CMU Serif','Nimbus Roman',Georgia,'Times New Roman',Times,serif"

BLUE   = "#8FC0DE"
ORANGE = "#F2A85C"
GOLD   = "#F3CE72"
RED    = "#E4572E"
BOX    = "#F8F6C7"
INK    = "#111111"
GREY   = "#555555"

o = []
A = o.append

A(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{SERIF}">')
A('<defs>')
A(f'<marker id="ah" viewBox="0 0 10 10" refX="9.2" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>')
A(f'<marker id="ahs" viewBox="0 0 10 10" refX="9.2" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>')
A('</defs>')
A(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

def circ(cx, cy, r, fill, sw=3):
    A(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{INK}" stroke-width="{sw}"/>')

def box(x, y, w, h, fill, sw=3):
    A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{INK}" stroke-width="{sw}"/>')

def arrow(x1, y1, x2, y2, sw=3.2, marker="ah"):
    A(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK}" stroke-width="{sw}" marker-end="url(#{marker})"/>')

def txt(x, y, s, size=34, fill=INK, anchor="middle", style="normal", weight="normal"):
    A(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
      f'font-style="{style}" font-weight="{weight}">{s}</text>')

def it(s):
    return f'<tspan font-style="italic">{s}</tspan>'

def sub(s, size="0.66em", dy=9):
    # the reset tspan carries a zero-width space so renderers honour its dy
    return (f'<tspan font-size="{size}" dy="{dy}">{s}</tspan>'
            f'<tspan font-size="1em" dy="{-dy}">&#8203;</tspan>')

# ---------------------------------------------------------------- top row
CY = 125
R  = 57

circ(133, CY, R, BLUE)
txt(133, CY + 14, it('k'), 40, weight="bold")
txt(133, 218, 'Fourier-mode', 20, GREY)
txt(133, 243, 'coordinate', 20, GREY)

arrow(193, CY, 251, CY)

circ(310, CY, R, ORANGE)
txt(310, CY + 15, '&#947;', 42, style="italic")
txt(310, 218, 'random Fourier', 20, GREY)
txt(310, 243, 'features', 20, GREY)

arrow(370, CY, 432, CY)

box(435, CY - 51, 283, 102, BOX)
txt(576, CY + 12, 'Sine layer 1', 34)

arrow(721, CY, 762, CY)

box(765, CY - 51, 285, 102, BOX)
txt(907, CY + 12, 'Sine layer 2', 34)

arrow(1053, CY, 1097, CY)

for cx in (1132, 1187, 1242):
    A(f'<circle cx="{cx}" cy="{CY}" r="13" fill="{INK}"/>')

arrow(1272, CY, 1297, CY)

box(1300, CY - 51, 290, 102, BOX)
txt(1445, CY + 12, 'Sine layer ' + '<tspan font-style="italic">L</tspan>', 34)

arrow(1593, CY, 1680, CY)

circ(1740, CY, R, ORANGE)
txt(1740, CY + 13, it('W') + sub('out'), 36)
txt(1740, 218, 'linear', 20, GREY)
txt(1740, 243, 'read-out', 20, GREY)

arrow(1800, CY, 1840, CY)

circ(1900, CY, R, BLUE)
txt(1900, CY + 13, it('R') + '(' + it('k') + ')', 36)
txt(1900, 218, 'per-mode channel', 20, GREY)
txt(1900, 243, 'mixing weight', 20, GREY)

# ------------------------------------------------------- zoom-in dashes
PX0, PY0, PX1, PY1 = 230, 302, 1790, 940
A(f'<line x1="765" y1="178" x2="{PX0+2}" y2="{PY0-2}" stroke="{INK}" stroke-width="2" stroke-dasharray="11 11"/>')
A(f'<line x1="1050" y1="178" x2="{PX1-2}" y2="{PY0-2}" stroke="{INK}" stroke-width="2" stroke-dasharray="11 11"/>')

box(PX0, PY0, PX1 - PX0, PY1 - PY0, "none")
txt(PX1 - 35, PY0 + 55, 'Sine layer', 36, anchor="end")

# ------------------------------------------------------------ zoom body
BY = 537
BX0, BX1 = 470, 1425
BY0, BY1 = 372, 702
A(f'<rect x="{BX0}" y="{BY0}" width="{BX1-BX0}" height="{BY1-BY0}" fill="{BOX}" stroke="{INK}" stroke-width="2.5"/>')

def mini_axes(x0, y0, w):
    A(f'<line x1="{x0}" y1="{y0}" x2="{x0+w}" y2="{y0}" stroke="{INK}" stroke-width="1.8" marker-end="url(#ahs)"/>')
    A(f'<line x1="{x0+w/2:.1f}" y1="{y0+40}" x2="{x0+w/2:.1f}" y2="{y0-40}" stroke="{INK}" stroke-width="1.8" marker-end="url(#ahs)"/>')

def ramp(x0, y0, w, dy):
    A(f'<line x1="{x0}" y1="{y0+dy:.1f}" x2="{x0+w}" y2="{y0-dy:.1f}" stroke="{INK}" stroke-width="2.4" stroke-linecap="round"/>')

def sine(x0, y0, w, cycles, amp=24):
    pts = []
    n = 90
    for i in range(n + 1):
        t = i / n
        x = x0 + t * w
        y = y0 - amp * math.sin(2 * math.pi * cycles * (t - 0.5))
        pts.append(f'{x:.1f},{y:.1f}')
    A(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{INK}" stroke-width="2.4" stroke-linejoin="round"/>')

# W (bias-free linear)
circ(552, BY, 52, GOLD)
txt(552, BY + 13, it('W') + sub('&#8467;'), 36)

# pre-activation ramps
for dy, slope in ((-92, 11), (0, 23), (92, 35)):
    mini_axes(632, BY + dy, 150)
    ramp(632, BY + dy, 150, slope)

# omega (learnable frequency scale)
circ(862, BY, 52, ORANGE)
txt(862, BY + 14, '&#969;' + sub('&#8467;'), 40, style="italic")

# omega -> sin
arrow(920, BY, 954, BY, sw=3.0)

# sin nonlinearity
circ(1012, BY, 52, RED)
txt(1012, BY + 12, 'sin', 34)

# post-activation sinusoids
for dy, c in ((-92, 1.0), (0, 2.2), (92, 4.0)):
    mini_axes(1102, BY + dy, 255)
    sine(1102, BY + dy, 255, c)

# state in / out
circ(340, 560, 58, BLUE)
txt(340, 573, it('h') + sub('&#8467;&#8722;1'), 34)
A(f'<path d="M 396 550 C 440 536 465 537 497 537" fill="none" stroke="{INK}" stroke-width="3.2" marker-end="url(#ah)"/>')

circ(1600, 560, 58, BLUE)
txt(1600, 573, it('h') + sub('&#8467;'), 34)
A(f'<path d="M {BX1} 537 C 1465 537 1508 547 1538 556" fill="none" stroke="{INK}" stroke-width="3.2" marker-end="url(#ah)"/>')

# equations
eq = (it('h') + sub('&#8467;') + ' = sin' + '&#8202;' + '(&#8201;' + it('&#969;') + sub('&#8467;')
      + ' &#8857; ' + it('W') + sub('&#8467;') + ' ' + it('h') + sub('&#8467;&#8722;1') + '&#8201;)')
txt(1010, 810, eq, 40)
note = ('bias-free ' + it('W') + sub('&#8467;', dy=6)
        + '&#8195;&#183;&#8195;learnable per-feature ' + it('&#969;') + sub('&#8467;', dy=6)
        + '&#8195;&#183;&#8195;' + it('W') + sub('&#8467;', dy=6)
        + ' &#8764; &#119984;&#8202;(&#177;&#8730;(6/' + it('n') + ')&#8201;/&#8201;'
        + it('&#969;') + sub('&#8467;', dy=6) + ')')
txt(1010, 870, note, 26, GREY)

A('</svg>')

out = pathlib.Path(__file__).resolve().parent.parent / 'figures' / 'diagrams' / 'siren_mechanism.svg'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\n'.join(o))
print(out)
