# -*- coding: utf-8 -*-
"""jsonpath.py — dotted-path lookup that tolerates dots *inside* JSON keys.

Why this exists (round-3 self-check, found by gate G1):

    results/chi_lip.json contains entries keyed

        "F_fixed_ag_0.5"        <-- note the period in the key itself

    A naive lookup splits the request path on "." and therefore looks for
    chi_lip_constrained -> "F_fixed_ag_0" -> "5" -> "rho_chi_to_chi", finds nothing,
    returns None, and gen_numbers.py turns None into the LaTeX placeholder "--".
    The result is a compiled paper that prints "--" where a measurement belongs --
    exactly the failure class that the deliverable gate's G1 exists to catch, and it
    did catch it.  The fix belongs here rather than at the call site, because every
    *_get helper in gen_numbers.py shares this lookup.

Rule: at each level, try the longest join of the remaining path segments that names an
existing key; recurse on the remainder.  The rule is strictly more permissive than the
naive split, so no path that used to resolve can stop resolving, and a genuinely absent
path still returns None.  `_probe_jsonpath.py` is the falsifiability probe for exactly
those three properties (dot-bearing key resolves; plain key still resolves; absent path
still returns None).

Ambiguity (round-3 R3 n3.2).  The same rule that fixes the dot-bearing key can also
*over-match*: for

    {"alpha.beta": 99, "alpha": {"gamma": 1}}

the request "alpha.beta" resolves to 99 through the flat key, while a reader of the JSON
would equally plausibly mean the nested one.  Returning a value silently is then as
dangerous as returning None, because a wrong number is worse than a visible placeholder.
`dig_path` keeps the documented precedence (longest key first) for compatibility, and
`dig_candidates` enumerates *every* interpretation that resolves, so callers can detect
the situation instead of guessing.  gen_numbers.py records every ambiguous lookup it
performs into results/ambiguity.json, which the deliverable gate reads back.
"""


def dig_path(d, path):
    """Resolve `path` (dot-separated) inside nested dict `d`; None if not found."""
    def rec(cur, parts):
        if not parts:
            return cur
        if not isinstance(cur, dict):
            return None
        # longest match first: a key may itself contain dots
        for j in range(len(parts), 0, -1):
            key = ".".join(parts[:j])
            if key in cur:
                r = rec(cur[key], parts[j:])
                if r is not None:
                    return r
        return None
    return rec(d, path.split("."))


def dig_candidates(d, path):
    """All interpretations of `path` that resolve, as (key_chain, value) pairs.

    `key_chain` is the tuple of keys actually traversed, so the two readings of an
    ambiguous path are distinguishable (a dotted label would collide: the flat key
    "alpha.beta" and the chain ("alpha", "beta") both render as "alpha.beta").

    Ordered by precedence: longest-key-first, exactly as `dig_path` picks.  The first
    entry therefore equals `dig_path(...)` whenever anything resolves.  len() > 1 means
    the request is ambiguous.
    """
    out = []

    def rec(cur, parts, chain):
        if not parts:
            out.append((chain, cur))
            return
        if not isinstance(cur, dict):
            return
        for j in range(len(parts), 0, -1):
            key = ".".join(parts[:j])
            if key in cur:
                rec(cur[key], parts[j:], chain + (key,))

    rec(d, path.split("."), ())
    return out

