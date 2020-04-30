__author__ = 'Wasiur R. KhudaBukhsh'

from colour import Color

def rawrgb2rgb(a,b,c):
    return Color(rgb = (a/255,b/255,c/255))

blues = dict(
    blue1 = rawrgb2rgb(113,199,236),
    blue2 = rawrgb2rgb(30,187,215),
    blue3 = rawrgb2rgb(24,154,211),
    blue4 = rawrgb2rgb(16,125,172),
    blue5 = rawrgb2rgb(0,80,115)
)

reds = dict(
    red1 = rawrgb2rgb(236,30,30),
    red2 = rawrgb2rgb(204,29,29),
    red3 = rawrgb2rgb(159,37,37),
    red4 = rawrgb2rgb(120,31,31),
    red5 = rawrgb2rgb(94,22,22)
)

greys = dict(
    grey1 = rawrgb2rgb(162,162,162),
    grey2 = rawrgb2rgb(81,81,81),
    grey3 = rawrgb2rgb(59,59,59),
    grey4 = rawrgb2rgb(37,37,37),
    grey5 = rawrgb2rgb(16,16,16)
)

forest = dict(
    forest1 = rawrgb2rgb(186,221,215),
    forest2 = rawrgb2rgb(44,53,73),
    forest3 = rawrgb2rgb(48,74,90),
    forest4 = rawrgb2rgb(94,131,110),
    forest5 = rawrgb2rgb(135,171,112)
)

bluegreys = dict(
    bluegrey1 = rawrgb2rgb(194,205,216),
    bluegrey2 = rawrgb2rgb(161,169,180),
    bluegrey3 = rawrgb2rgb(56,129,184),
    bluegrey4 = rawrgb2rgb(35,81,116),
    bluegrey5 = rawrgb2rgb(29,43,73)
)

coffee = dict(
    coffee1 = rawrgb2rgb(236,224,209),
    coffee2 = rawrgb2rgb(219,193,172),
    coffee3 = rawrgb2rgb(216,197,166),
    coffee4 = rawrgb2rgb(112,64,65),
    coffee5 = rawrgb2rgb(56,34,15)
)

pinks = dict(
    pink1 = rawrgb2rgb(250,236,230),
    pink2 = rawrgb2rgb(238,207,200),
    pink3 = rawrgb2rgb(217,178,169),
    pink4 = rawrgb2rgb(163,126,113)
)

browns = dict(
    brown1 = rawrgb2rgb(219,201,184),
    brown2 = rawrgb2rgb(161,126,97),
    brown3 = rawrgb2rgb(133,88,50),
    brown4 = rawrgb2rgb(116,72,42),
    brown5 = rawrgb2rgb(54,41,37)
)

browngreen = dict(
    browngreen1 = rawrgb2rgb(221,213,199),
    browngreen2 = rawrgb2rgb(184,171,139),
    browngreen3 = rawrgb2rgb(139,138,104),
    browngreen4 = rawrgb2rgb(105,103,61),
    browngreen5 = rawrgb2rgb(60,56,34)
)

purplybrown = dict(
    purplybrown1 = rawrgb2rgb(182,138,130),
    purplybrown2 = rawrgb2rgb(161,125,132),
    purplybrown3 = rawrgb2rgb(139,114,134),
    purplybrown4 = rawrgb2rgb(116,99,124),
    purplybrown5 = rawrgb2rgb(92,89,114)
)

junglegreen = dict(
    green1 = rawrgb2rgb(133,170,155),
    green2 = rawrgb2rgb(88,139,118),
    green3 = rawrgb2rgb(41,95,72),
    green4 = rawrgb2rgb(32,76,57),
    green5 = rawrgb2rgb(24,57,43)
)

greens = dict(
    green1 = rawrgb2rgb(148,206,152),
    green2 = rawrgb2rgb(97,175,102),
    green3 = rawrgb2rgb(56,142,62),
    green4 = rawrgb2rgb(27,112,33),
    green5 = rawrgb2rgb(6,78,10)
)

maroons = dict(
    maroon1 = rawrgb2rgb(193,113,113),
    maroon2 = rawrgb2rgb(169,76,76),
    maroon3 = rawrgb2rgb(146,68,68),
    maroon4 = rawrgb2rgb(109,54,54),
    maroon5 = rawrgb2rgb(86,36,36)
)

cyans = dict(
    cyan1 = rawrgb2rgb(138,187,187),
    cyan2 = rawrgb2rgb(101,155,150),
    cyan3 = rawrgb2rgb(60,131,132),
    cyan4 = rawrgb2rgb(29,91,95),
    cyan5 = rawrgb2rgb(0,78,82)
)