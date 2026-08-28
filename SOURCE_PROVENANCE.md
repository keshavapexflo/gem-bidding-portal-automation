# Source provenance

This unified deployment was assembled from the following local references:

- `D:\gem-bidding-portal_phase1`
  - GitHub: `https://github.com/keshavapexflo/gem-bidding-portal_phase1`
  - Commit: `fbfddaa0514d53ad435076147be93a78f1fe79de`
- `D:\gem-bidding-portal_phase2`
  - GitHub: `https://github.com/keshavapexflo/gem-bidding-portal_phase2`
  - Commit: `012ef412a3a782d2f1ab9addcef0ddf978dee04c`
- Initial embedding workflow reviewed from `D:\downloads\embedding_40k.ipynb`.

The five shared application modules in the two phase repositories were
identical when consolidated. Phase 2 supplied the original embedding,
maintenance, cleanup, and Windows setup scripts. This deployment adds a unified
entry point, restart-safe sync journaling, safer expiry behavior, deterministic
chunk IDs, durable boilerplate classification, absolute runtime paths,
validation, backup, and deployment wrappers.
