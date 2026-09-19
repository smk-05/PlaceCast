"""Generation — spec section 5. Hosted on Replicate, per addendum D.2.

Running FLUX and TRELLIS locally alongside a segmentation model does not fit in
8 GB. Hosting both removes the two largest models from the machine entirely and
makes the local GPU load trivial.

Everything here is cached to disk under the asset UUID. Re-running the solver
must never re-run a model — you will re-run the solver hundreds of times while
debugging and each generation costs 10-60 s and real money.
"""
