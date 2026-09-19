"""
The demo and benchmark building sets. Spec 11, 15.

DEMO: the five addresses rehearsed for judging. Judging is in NCB, so that one
has to land. Each is chosen to exercise a different part of the solver, and one
(Moss Arts Center) is chosen to FAIL informatively — it is organic, so its
rectilinearity R should come in low and the pipeline should flag it for review
rather than confidently placing it wrong. Demonstrating that is worth more than
a fifth clean success.

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
    # addr                                    area    R      aspect  height
    "Burruss Hall, Blacksburg, VA",          # 6136  1.000   1.44   20.7 m tag
    "Torgersen Hall, Blacksburg, VA",        # 5353  0.919   2.42   6 levels
    "Goodwin Hall, Blacksburg, VA",          # 4068  0.999   1.26   4 levels
    "Classroom Building, Blacksburg, VA",    # 2272  0.993   2.22   3 levels  <- NCB
    "Moss Arts Center, Blacksburg, VA",      # 7897  1.000   1.08   NO TAG
]

# Why this set, given the measurements:
#   NCB          aspect 2.22, R 0.993 — the easy case, and judging happens in
#                it. This one has to land.
#   Burruss      the largest relation in the demo set and the one that exercises
#                multi-way ring stitching. R 1.000 once assembled correctly, and
#                it carries an explicit `height` tag, so it exercises tier 1 of
#                spec 7. If this ever reads 4384 m2 again, the stitching broke.
#   Moss Arts    aspect 1.08 is BELOW spec 6.6 Filter 1's 1.1 cutoff, so the
#                90-degree candidates cannot be excluded on scale grounds. This
#                is the genuinely ambiguous case and the whole disambiguation
#                chain has to carry it. It also has NO height tag, so it
#                exercises spec 7's fallback. The best demo in the set.
#   Torgersen    R 0.919 is the lowest of the five — a genuinely complex plan
#                with the bridge over Alumni Mall.
#   Goodwin      clean rectilinear control, and a plain way rather than a
#                relation, so it isolates solver bugs from parsing bugs.

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
