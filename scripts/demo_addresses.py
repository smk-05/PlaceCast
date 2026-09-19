"""
The demo and benchmark building sets. Spec 11, 15.

DEMO: the five addresses rehearsed for judging. Judging is in NCB, so that one
has to land. Each is chosen to exercise a different part of the solver, and one
(Moss Arts Center) is chosen to FAIL informatively — it is organic, so its
rectilinearity R should come in low and the pipeline should flag it for review
rather than confidently placing it wrong. Demonstrating that is worth more than
a fifth clean success.

BENCHMARK: spec 11's twenty buildings across four strata — five rectangular,
five L-shaped or complex, five near-square, five on sloped ground. Include
buildings the judges will recognise.
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

# Spec 11's four strata. Hand-annotate ground truth for each by manually
# aligning a reference box; the resulting rows are also addendum B's training
# set (20 buildings x 4 ablation configs = 80 labelled placements).
#
# CAVEAT: these stratum assignments are guesses from building names, and the
# demo set already proved two such guesses wrong. Before the hour-30 benchmark
# run, verify each one against its MEASURED aspect and R — "near-square" means
# aspect < 1.1, "complex" means R below about 0.85 — and move buildings between
# strata accordingly. A stratum that does not actually contain near-square
# buildings tests nothing.
BENCHMARK_RECTANGULAR = [
    "Burruss Hall, Blacksburg, VA",
    "McBryde Hall, Blacksburg, VA",
    # Randolph Hall removed: it is DEMOLISHED. OSM now carries "Randolph Hall
    # Demolition, Mitchell Hall Construction" at that point and no building
    # polygon, so the selection rule correctly refuses it. That is a bad
    # benchmark entry, not a solver failure — benchmarking it would measure
    # nothing. Verified live 2026-09-19.
    "Major Williams Hall, Blacksburg, VA",
    "Whittemore Hall, Blacksburg, VA",
    "Hancock Hall, Blacksburg, VA",
]

BENCHMARK_COMPLEX = [
    "Torgersen Hall, Blacksburg, VA",
    "Squires Student Center, Blacksburg, VA",
    "Newman Library, Blacksburg, VA",
    "Durham Hall, Blacksburg, VA",
    "Holden Hall, Blacksburg, VA",
]

BENCHMARK_NEAR_SQUARE = [
    "Goodwin Hall, Blacksburg, VA",
    "Davidson Hall, Blacksburg, VA",
    "Robeson Hall, Blacksburg, VA",
    "Hutcheson Hall, Blacksburg, VA",
    "Patton Hall, Blacksburg, VA",   # was Femoyer Hall — Nominatim cannot resolve it
]

BENCHMARK_SLOPED = [
    # Lane Stadium is also the MULTI-PART case: relation/2417911 is four
    # disjoint outer rings (the stands) with the field as a genuine gap, so the
    # geocoded point is honestly not contained by the footprint. Keep it — it is
    # the only benchmark entry exercising MultiPolygon handling.
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
    + BENCHMARK_SLOPED
)

STRATA = {
    "rectangular": BENCHMARK_RECTANGULAR,
    "complex": BENCHMARK_COMPLEX,
    "near_square": BENCHMARK_NEAR_SQUARE,
    "sloped": BENCHMARK_SLOPED,
}
