"""
The demo and benchmark building sets. Spec 11, 15.

DEMO: the six addresses photographed on site and rehearsed for judging.
Judging is in NCB, so that one has to land.

BENCHMARK: spec 11's twenty buildings, stratified by MEASURED shape
(rectangular / complex / near-square) — see the comment above the lists.
"""

# MEASURED values, from scripts/prefetch_footprints.py against live OSM on
# 2026-09-19, AFTER the multi-way ring-stitching fix. Do not trust any earlier
# figures for the two relations: Burruss and Torgersen were both truncated to
# their first member way, which understated Burruss by 29% and inverted its
# rectilinearity (0.713 -> 1.000, because the missing wings made the partial
# ring look irregular).
DEMO = [
    # addr                                  R     WxL m    height
    "Classroom Building, Blacksburg, VA",  # 0.99  35x78   3 levels      <- NCB
    "Burruss Hall, Blacksburg, VA",        # 1.00  71x102  20.7 m tag
    "Patton Hall, Blacksburg, VA",         # 1.00  22x65   16.4 m tag
    "War Memorial Hall, Blacksburg, VA",   # 1.00  98x113  18.8 m tag    near-square
    "Whittemore Hall, Blacksburg, VA",     # 1.00  43x79   37.4 m tag
    "Goodwin Hall, Blacksburg, VA",        # 1.00  72x91   4 levels
]

# Why this set (2026-09-19, after the benchmark):
#   NCB          judging happens in it. It has to land. Levels-only height, so
#                it routes to review once the non-authoritative-height rule is in.
#   Burruss,     explicit metre `height` tags (spec 7 tier 1) and R = 1.00, the
#   Patton,      shapes the solver handles cleanly. War Memorial is near-square
#   War Memorial, (aspect 1.14), so orientation rests on the photo — the case
#   Whittemore   EXIF exists for.
#   Goodwin      replaces McBryde, whose R is 0.249 (non-orthogonal wings): the
#                gate would send it to review whatever photo was taken.
# Moss Arts, Torgersen: dropped from the demo, still in the benchmark or
# prefetch cache; no photo walk budget for them.

# Spec 11's strata, assigned from MEASURED shape (scripts/run_benchmark.py,
# 2026-09-19), not from building names. The name-based guesses were wrong for
# 8 of 20: McBryde (R 0.249 — non-orthogonal wings, not a rectangle), Newman
# (R 0.425), Lane Stadium (R 0.393) are complex; Goodwin, Davidson, Robeson,
# Hutcheson and Patton are NOT near-square (aspect 1.24-2.87); Holden and War
# Memorial are. The rule, applied in this order:
#     complex      R < 0.85
#     near_square  aspect < 1.15   (spec says 1.1; only Newman clears that,
#                                   and it is already complex)
#     rectangular  everything else
# run_benchmark.measured_stratum() applies the same rule at run time, so a
# changed footprint re-stratifies itself instead of silently mislabelling.
#
# "Sloped" is NOT a footprint property — slope only affects terrain height,
# which the 2D benchmark does not exercise. SLOPED_SITES is kept as a tag for
# the terrain check; the slopes themselves are unverified.
BENCHMARK_RECTANGULAR = [
    "Burruss Hall, Blacksburg, VA",          # R 1.000  aspect 1.44
    "Major Williams Hall, Blacksburg, VA",   # 0.964  1.31  (Randolph: demolished)
    "Whittemore Hall, Blacksburg, VA",       # 1.000  1.86
    "Hancock Hall, Blacksburg, VA",          # 0.882  1.77
    "Torgersen Hall, Blacksburg, VA",        # 0.919  2.42
    "Goodwin Hall, Blacksburg, VA",          # 0.999  1.26
    "Davidson Hall, Blacksburg, VA",         # 0.878  1.46
    "Robeson Hall, Blacksburg, VA",          # 0.999  1.64
    "Hutcheson Hall, Blacksburg, VA",        # 1.000  1.24
    "Patton Hall, Blacksburg, VA",           # 0.997  2.87  (Femoyer: not geocodable)
    "Cassell Coliseum, Blacksburg, VA",      # 1.000  1.34
    "Derring Hall, Blacksburg, VA",          # 1.000  3.10
    "Price Hall, Blacksburg, VA",            # 1.000  2.35
]

BENCHMARK_COMPLEX = [
    "McBryde Hall, Blacksburg, VA",          # 0.249  1.13
    "Newman Library, Blacksburg, VA",        # 0.425  1.06  courtyard
    "Lane Stadium, Blacksburg, VA",          # 0.393  2.24  multi-part (4 stands)
    "Squires Student Center, Blacksburg, VA",  # 0.787  1.90
    "Durham Hall, Blacksburg, VA",           # 0.826  2.80
]

BENCHMARK_NEAR_SQUARE = [
    "Holden Hall, Blacksburg, VA",           # 0.997  1.11
    "War Memorial Hall, Blacksburg, VA",     # 1.000  1.14
]

SLOPED_SITES = [
    "Lane Stadium, Blacksburg, VA",
    "Cassell Coliseum, Blacksburg, VA",
    "War Memorial Hall, Blacksburg, VA",
    "Derring Hall, Blacksburg, VA",
    "Price Hall, Blacksburg, VA",
]

BENCHMARK = (
    BENCHMARK_RECTANGULAR
    + BENCHMARK_COMPLEX
    + BENCHMARK_NEAR_SQUARE
)

STRATA = {
    "rectangular": BENCHMARK_RECTANGULAR,
    "complex": BENCHMARK_COMPLEX,
    "near_square": BENCHMARK_NEAR_SQUARE,
}
