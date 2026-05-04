#!/usr/bin/env python3
"""
optimize_products_fewer_FFT.py

Variant of optimize_products.py that ensures every Phase 3 term contains
exactly ONE standalone factor of <w|gamma_mu|v>, named prod_vec1_mu. This
universal factor is FFT'd once globally in the C++/CUDA stage and reused
across all terms.

For each term, the OTHER <w|gamma|v> insertion is split into bare <w|(pos)
and gamma|v>(pos) pieces and absorbed into a chained vector-matrix product
via the same machinery as the parent optimizer (only matrix-vector ops at
single-site closure for vec-vec).

The choice of which vertex to keep standalone is made per term to maximize
global reuse of the resulting absorbed-vertex chain (single-pass greedy on
canonical chain counts; iterative fix-point optional).

Usage:
    python3 optimize_products_fewer_FFT.py <input_mom.txt>
"""

import sys
import re
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional


# =============================================================================
# Parsing
# =============================================================================

def parse_trace_elements(trace_str: str) -> List[str]:
    """
    Parse a trace string like 'Tr[elem1 . elem2 . elem3]' into a list of
    element strings: ['elem1', 'elem2', 'elem3'].
    """
    match = re.match(r'Tr\[(.+)\]', trace_str)
    if not match:
        return [trace_str]
    content = match.group(1)
    elements = [e.strip() for e in content.split(' . ')]
    return elements


def split_current_vertices(elements: List[str]) -> List[str]:
    """
    Split <w|gamma_X|v>(pos) into two elements: <w|(pos) and gamma_X |v>(pos).
    The gamma matrix stays with |v>.
    Pi elements are left unchanged.
    """
    new_elements = []
    for elem in elements:
        match = re.match(r'<w\|(gamma_\w+)\|v>\((.+)\)', elem)
        if match:
            gamma = match.group(1)
            pos = match.group(2)
            new_elements.append(f"<w|({pos})")
            new_elements.append(f"{gamma} |v>({pos})")
        else:
            new_elements.append(elem)
    return new_elements


def extract_traces_from_rest(rest: str) -> List[str]:
    """
    Extract all Tr[...] blocks from the RHS of a term (after the coefficient).
    Returns a list of trace strings like ['Tr[...]', 'Tr[...]'].
    """
    traces = []
    i = 0
    while i < len(rest):
        idx = rest.find('Tr[', i)
        if idx == -1:
            break
        bracket_count = 0
        j = idx + 2
        while j < len(rest):
            if rest[j] == '[':
                bracket_count += 1
            elif rest[j] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    break
            j += 1
        traces.append(rest[idx:j + 1])
        i = j + 1
    return traces


def parse_term_line(line: str) -> Optional[Tuple[str, str, List[Tuple[str, List[str]]]]]:
    """
    Parse a term line like:
        term_X = coef_X * Tr[...] * Tr[...]

    Returns (name, coef, traces) where traces is a list of
    (original_trace_str, elements) tuples. Current vertices are NOT split here;
    the per-term carve-out pass decides which one stays unsplit (standalone)
    and splits the other.
    Returns None if the line doesn't match.
    """
    match = re.match(r'(term_\w+)\s*=\s*(coef_\w+)\s*\*\s*(.*)', line)
    if not match:
        return None

    name = match.group(1)
    coef = match.group(2)
    rest = match.group(3).strip()

    trace_strings = extract_traces_from_rest(rest)

    traces = []
    for trace_str in trace_strings:
        elements = parse_trace_elements(trace_str)
        traces.append((trace_str, elements))

    return (name, coef, traces)


# =============================================================================
# Per-term vertex carve-out
# =============================================================================
#
# Every EM term has two current vertices <w|gamma|v>(pos), one at x_1 and one
# at x_2. The carve-out picks ONE per term to keep unsplit (the standalone
# universal vertex), and splits the other into <w|(pos) + gamma|v>(pos) so
# Phase 2 can absorb it into a chain at that position.
#
# The choice between (a) "keep V_mu(x_1) standalone" and (b) "keep V_nu(x_2)
# standalone" is made greedily, maximizing the global reuse of the
# resulting absorbed-chain canonical key.
#
# For connected traces (1 trace, 2 vertices), the trace is rotated so the
# chosen standalone is at index 0 — this puts the absorbed chain in a
# linear segment that Phase 2's adjacent-pair greedy can absorb. The
# absorbed-vertex chain may have Pi matrices on BOTH sides of <w|/|v>
# (e.g., Type5), which is why this variant allows mat . <w| pairs that the
# parent optimizer's "orphan-bra" rule forbids.

def _find_unsplit_vertices(traces):
    """Return [(trace_idx, elem_idx, gamma, position), ...] for every
    <w|gamma_X|v>(pos) element across the given traces."""
    out = []
    for ti, (_, elements) in enumerate(traces):
        for ei, elem in enumerate(elements):
            m = re.match(r'<w\|gamma_(\w+)\|v>\((x_\d+)\)', elem)
            if m:
                out.append((ti, ei, m.group(1), m.group(2)))
    return out


def _split_one_vertex(elements: List[str], idx: int) -> List[str]:
    """Split elements[idx] = <w|gamma_X|v>(pos) into <w|(pos) and gamma_X |v>(pos)."""
    elem = elements[idx]
    m = re.match(r'<w\|(gamma_\w+)\|v>\((.+)\)', elem)
    if not m:
        return list(elements)
    gamma = m.group(1)
    pos = m.group(2)
    return list(elements[:idx]) + [f"<w|({pos})", f"{gamma} |v>({pos})"] + list(elements[idx + 1:])


def _rotate(elements: List[str], start: int) -> List[str]:
    return list(elements[start:]) + list(elements[:start])


def _strip_pos_gamma(elem: str) -> str:
    """Strip x_1/x_2 and mu/nu from an element for canonical-key comparison."""
    s = re.sub(r'\(x_\d+\)', '(x)', elem)
    s = re.sub(r'gamma_\w+', 'gamma', s)
    return s


def _canonical_chain_key(traces, standalone_loc):
    """Build a canonical hashable key for the absorbed-chain shape that
    results from keeping the vertex at standalone_loc=(trace_idx, elem_idx)
    unsplit and splitting any other vertex.

    For the trace containing the standalone, the trace is rotated so the
    standalone sits at index 0 and EXCLUDED from the key (only the absorbed
    chain — the rest of the trace, with the other vertex split — is keyed).
    Other traces are included verbatim with any vertex split.
    Position (x_1/x_2) and gamma (mu/nu) labels are stripped.
    """
    s_ti, s_ei = standalone_loc
    per_trace_keys = []
    for ti, (_, elements) in enumerate(traces):
        if ti == s_ti:
            rotated = _rotate(list(elements), s_ei)
            for ei in range(1, len(rotated)):
                if re.match(r'<w\|gamma_\w+\|v>\(', rotated[ei]):
                    rotated = _split_one_vertex(rotated, ei)
                    break
            chain = rotated[1:]
        else:
            chain = list(elements)
            for ei in range(len(chain)):
                if re.match(r'<w\|gamma_\w+\|v>\(', chain[ei]):
                    chain = _split_one_vertex(chain, ei)
                    break
        per_trace_keys.append(tuple(_strip_pos_gamma(e) for e in chain))
    return tuple(sorted(per_trace_keys))


def carve_out_per_term(terms, iterate: bool = False):
    """Per-term, choose which current vertex stays unsplit (standalone) and
    split the other into <w|(pos) + gamma_X |v>(pos). Greedy single-pass
    on canonical chain reuse counts; iterate=True runs a fix-point loop.

    Connected traces are rotated so the standalone sits at index 0 (so the
    absorbed chain forms a linear segment for Phase 2).
    """
    vertex_locs = []  # per-term list of (trace_idx, elem_idx)
    for name, coef, traces in terms:
        locs = _find_unsplit_vertices(traces)
        vertex_locs.append([(l[0], l[1]) for l in locs])

    candidate_keys = []
    for i, (name, coef, traces) in enumerate(terms):
        if len(vertex_locs[i]) < 2:
            candidate_keys.append((None, None))
            continue
        ka = _canonical_chain_key(traces, vertex_locs[i][0])
        kb = _canonical_chain_key(traces, vertex_locs[i][1])
        candidate_keys.append((ka, kb))

    # Initial counts: union over all candidates (treat each term as if it
    # had picked option_a OR option_b — count both).
    union_counts: Dict[tuple, int] = {}
    for ka, kb in candidate_keys:
        if ka is not None:
            union_counts[ka] = union_counts.get(ka, 0) + 1
        if kb is not None and kb != ka:
            union_counts[kb] = union_counts.get(kb, 0) + 1

    choices = [0] * len(terms)
    for i, (ka, kb) in enumerate(candidate_keys):
        if ka is None:
            continue
        ca = union_counts.get(ka, 0)
        cb = union_counts.get(kb, 0)
        if cb > ca:
            choices[i] = 1
        elif ca > cb:
            choices[i] = 0
        else:
            choices[i] = 0 if ka <= kb else 1

    if iterate:
        while True:
            counts: Dict[tuple, int] = {}
            for i, (ka, kb) in enumerate(candidate_keys):
                k = (ka, kb)[choices[i]] if ka is not None else None
                if k is not None:
                    counts[k] = counts.get(k, 0) + 1
            changed = False
            for i, (ka, kb) in enumerate(candidate_keys):
                if ka is None:
                    continue
                ca = counts.get(ka, 0)
                cb = counts.get(kb, 0)
                if choices[i] == 0:
                    new = 0 if ca >= cb else 1
                else:
                    new = 1 if cb >= ca else 0
                if new != choices[i]:
                    changed = True
                    choices[i] = new
            if not changed:
                break

    # Apply chosen split.
    new_terms = []
    for i, (name, coef, traces) in enumerate(terms):
        if len(vertex_locs[i]) < 2:
            # No two vertices found — pass through with all vertices split.
            new_traces = []
            for trace_str, elements in traces:
                new_traces.append((trace_str, split_current_vertices(elements)))
            new_terms.append((name, coef, new_traces))
            continue
        s_ti, s_ei = vertex_locs[i][choices[i]]
        new_traces = []
        for ti, (trace_str, elements) in enumerate(traces):
            if ti == s_ti:
                rotated = _rotate(list(elements), s_ei)
                for ei in range(1, len(rotated)):
                    if re.match(r'<w\|gamma_\w+\|v>\(', rotated[ei]):
                        rotated = _split_one_vertex(rotated, ei)
                        break
                new_traces.append((trace_str, rotated))
            else:
                new_elements = list(elements)
                ei = 0
                while ei < len(new_elements):
                    if re.match(r'<w\|gamma_\w+\|v>\(', new_elements[ei]):
                        new_elements = _split_one_vertex(new_elements, ei)
                        ei += 2
                    else:
                        ei += 1
                new_traces.append((trace_str, new_elements))
        new_terms.append((name, coef, new_traces))

    return new_terms


# =============================================================================
# Element classification and position tracking
# =============================================================================

def is_unsplit_vertex(elem: str) -> bool:
    """Check if an element is an unsplit current vertex <w|gamma_X|v>(pos).
    These are the per-term standalone vertices preserved by the carve-out
    pass; they must be skipped by all pairing logic.
    """
    return bool(re.match(r'<w\|gamma_\w+\|v>\(', elem))


def is_vector_element(elem: str) -> bool:
    """Check if an element is a raw vector (<w| or |v>) rather than a matrix.
    Unsplit standalone vertices <w|gamma_X|v>(pos) are NOT vectors here —
    they are matrix-shaped objects at a position and must be left untouched.
    """
    if is_unsplit_vertex(elem):
        return False
    return elem.startswith("<w|(") or "|v>(" in elem


def is_vector_vector_pair(a: str, b: str) -> bool:
    """
    Check if both elements are raw vectors. This catches:
      - <w|(pos) . gamma_X |v>(pos)  (w_v: original current vertex)
      - gamma_X |v>(pos) . <w|(pos)  (v_w: outer product, expensive)
    Vector-vector products must always be avoided.
    """
    return is_vector_element(a) and is_vector_element(b)


def get_element_position(elem: str,
                         product_positions: Dict[str, Optional[str]]) -> Optional[str]:
    """
    Get the spatial position (e.g. 'x_1', 'x_2', or None) of an element.

    - Pi(...) fields are position-free (momentum-space) -> None
    - <w|(x_1), gamma_mu |v>(x_1) -> 'x_1'
    - Product names are looked up in the product_positions dict
    """
    # Check if it's a known product
    if elem in product_positions:
        return product_positions[elem]
    # Pi fields are position-free
    if elem.startswith("Pi("):
        return None
    # Raw vector elements: extract position from parentheses
    # <w|(x_1) -> x_1, gamma_mu |v>(x_1) -> x_1
    match = re.search(r'\((x_\d+)\)', elem)
    if match:
        return match.group(1)
    return None


def positions_compatible(pos_a: Optional[str], pos_b: Optional[str]) -> bool:
    """
    Check if two positions can be combined in a product.
    None (position-free) is compatible with anything.
    Two positions are only compatible if they are the same.
    """
    if pos_a is None or pos_b is None:
        return True
    return pos_a == pos_b


def combined_position(pos_a: Optional[str], pos_b: Optional[str]) -> Optional[str]:
    """Get the position of a product of two elements."""
    if pos_a is not None:
        return pos_a
    return pos_b


def get_element_gamma(elem: str,
                      product_gammas: Dict[str, Optional[str]]) -> Optional[str]:
    """
    Get the gamma label (e.g. 'mu', 'nu') of an element, or None.

    - gamma_mu |v>(x_1) -> 'mu'
    - gamma_nu |v>(x_2) -> 'nu'
    - Product names are looked up in the product_gammas dict
    - Everything else (Pi, <w|, prod_Pi) -> None
    """
    if elem in product_gammas:
        return product_gammas[elem]
    match = re.match(r'gamma_(\w+)\s+\|v>', elem)
    if match:
        return match.group(1)
    return None


def combined_gamma(gamma_a: Optional[str], gamma_b: Optional[str]) -> Optional[str]:
    """Get the gamma label of a product of two elements."""
    if gamma_a is not None:
        return gamma_a
    return gamma_b


def make_pair_key(a: str, b: str) -> str:
    """Create a string key for an adjacent pair."""
    return f"{a} . {b}"


# =============================================================================
# Pair collection
# =============================================================================

def collect_pairs(terms,
                  matrix_only: bool = False,
                  product_positions: Optional[Dict[str, Optional[str]]] = None
                  ) -> Dict[str, int]:
    """
    Collect all adjacent pairs across all terms and count occurrences.

    Args:
        matrix_only: If True, only collect pairs where NEITHER element is
                     a raw vector. Restricts to Pi-chain products.
        product_positions: If provided, enforce position locality — skip pairs
                          that would combine elements at different positions.
    """
    pair_counts = {}
    for name, coef, traces in terms:
        for trace_str, elements in traces:
            for i in range(len(elements) - 1):
                a, b = elements[i], elements[i + 1]
                # Skip pairs involving the unsplit standalone vertex
                if is_unsplit_vertex(a) or is_unsplit_vertex(b):
                    continue
                # Always skip vector-vector pairs
                if is_vector_vector_pair(a, b):
                    continue
                # In matrix-only mode, skip any pair involving a vector
                if matrix_only and (is_vector_element(a) or is_vector_element(b)):
                    continue
                # NB: this variant ALLOWS matrix . <w|(pos) pairs (parent
                # forbids them via an "orphan-bra" rule). We need bra-side
                # dressing because the absorbed-vertex chain in connected
                # types like Type5 has Pi matrices on both sides of the
                # bra/ket.
                # Enforce position locality
                if product_positions is not None:
                    pos_a = get_element_position(a, product_positions)
                    pos_b = get_element_position(b, product_positions)
                    if not positions_compatible(pos_a, pos_b):
                        continue
                pair_key = make_pair_key(a, b)
                pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
    return pair_counts


# =============================================================================
# Substitution
# =============================================================================

def substitute_pairs(elements: List[str], product_map: Dict[str, str]) -> List[str]:
    """
    Greedily substitute adjacent pairs with product names, left to right.
    When two overlapping pairs are both defined, the leftmost one wins.
    """
    result = []
    i = 0
    while i < len(elements):
        if i < len(elements) - 1:
            pair_key = make_pair_key(elements[i], elements[i + 1])
            if pair_key in product_map:
                result.append(product_map[pair_key])
                i += 2
                continue
        result.append(elements[i])
        i += 1
    return result


def apply_substitution_to_terms(terms, product_map: Dict[str, str]):
    """Apply pair substitutions to all terms, returning updated terms."""
    new_terms = []
    for name, coef, traces in terms:
        new_traces = []
        for trace_str, elements in traces:
            substituted = substitute_pairs(elements, product_map)
            new_traces.append((trace_str, substituted))
        new_terms.append((name, coef, new_traces))
    return new_terms


# =============================================================================
# Two-phase optimization
# =============================================================================

def optimize_phase1(terms):
    """
    Phase 1: Matrix-matrix products only (Pi chains).
    No position tracking needed since Pi fields are position-free.

    Returns (terms, product_defs, product_positions)
    """
    product_defs = []
    product_positions = {}  # all Phase 1 products are position-free
    counter = 0
    level = 0

    while True:
        pair_counts = collect_pairs(terms, matrix_only=True)
        repeated = {k: v for k, v in pair_counts.items() if v >= 2}
        if not repeated:
            break

        level += 1
        product_names = OrderedDict()
        for pair_key in sorted(repeated.keys()):
            counter += 1
            prod_name = f"prod_Pi{counter}"
            product_names[pair_key] = prod_name
            product_positions[prod_name] = None  # position-free

        terms = apply_substitution_to_terms(terms, product_names)
        product_defs.append((level, product_names,
                             {k: pair_counts[k] for k in product_names}))

    return terms, product_defs, product_positions


def optimize_phase2(terms, product_positions: Dict[str, Optional[str]]):
    """
    Phase 2: Vector-matrix products with position locality enforcement.
    Never combines elements at different spatial positions.
    Products carry their position in their name: prod_vecN(x_1).

    Returns (terms, product_defs)
    """
    product_defs = []
    product_gammas = {}
    counter = 0
    level = 0

    while True:
        pair_counts = collect_pairs(terms, matrix_only=False,
                                    product_positions=product_positions)
        repeated = {k: v for k, v in pair_counts.items() if v >= 2}
        if not repeated:
            break

        level += 1
        product_names = OrderedDict()
        for pair_key in sorted(repeated.keys()):
            counter += 1
            # Determine position of this product
            a, b = pair_key.split(' . ', 1)
            pos_a = get_element_position(a, product_positions)
            pos_b = get_element_position(b, product_positions)
            position = combined_position(pos_a, pos_b)

            # Determine gamma label of this product
            gamma_a = get_element_gamma(a, product_gammas)
            gamma_b = get_element_gamma(b, product_gammas)
            gamma = combined_gamma(gamma_a, gamma_b)
            gamma_suffix = f"_{gamma}" if gamma else ""

            # Include gamma label and position in the product name
            if position:
                prod_name = f"prod_vec{counter}{gamma_suffix}({position})"
            else:
                prod_name = f"prod_vec{counter}{gamma_suffix}"

            product_names[pair_key] = prod_name
            product_positions[prod_name] = position
            product_gammas[prod_name] = gamma

        terms = apply_substitution_to_terms(terms, product_names)
        product_defs.append((level, product_names,
                             {k: pair_counts[k] for k in product_names}))

    return terms, product_defs, product_gammas


# =============================================================================
# Phase 2 deduplication: merge products differing only by position/gamma
# =============================================================================

def _canonicalize_element(elem: str, old_to_canonical: Dict[str, str]) -> str:
    """Strip position and gamma label from an element for deduplication.

    <w|(x_1) -> <w|
    gamma_mu |v>(x_1) -> gamma |v>
    prod_vec7_mu(x_1) -> lookup canonical name
    Pi(...), prod_PiN -> unchanged
    """
    # <w|(x_N) -> <w|
    if re.match(r'<w\|\(x_\d+\)', elem):
        return '<w|'
    # gamma_X |v>(x_N) -> gamma |v>
    if re.match(r'gamma_\w+\s+\|v>\(x_\d+\)', elem):
        return 'gamma |v>'
    # prod_vec reference: look up canonical name
    if elem in old_to_canonical:
        return old_to_canonical[elem]
    # Pi(...), prod_PiN -> unchanged
    return elem


def _canonicalize_pair_key(pair_key: str, old_to_canonical: Dict[str, str]) -> str:
    """Canonicalize a pair key by stripping position/gamma from both sides."""
    parts = pair_key.split(' . ', 1)
    return ' . '.join(_canonicalize_element(p.strip(), old_to_canonical)
                      for p in parts)


def _rewrite_term_element(elem: str,
                          old_to_canonical: Dict[str, str],
                          old_name_info: Dict[str, tuple]) -> str:
    """Rewrite a prod_vec reference with canonical name, re-attaching position/gamma.

    Raw elements (<w|(x_1), gamma_mu |v>(x_1), Pi(...), prod_PiN) pass through.
    """
    if elem in old_to_canonical:
        canonical = old_to_canonical[elem]
        pos, gamma = old_name_info[elem]
        # Strip canonical _mu suffix before re-attaching actual gamma label
        base_name = re.sub(r'_mu$', '', canonical)
        gamma_suffix = f"_{gamma}" if gamma else ""
        pos_suffix = f"({pos})" if pos else ""
        return f"{base_name}{gamma_suffix}{pos_suffix}"
    return elem


def deduplicate_phase2(phase2_defs, terms,
                       product_positions: Dict[str, Optional[str]],
                       product_gammas: Dict[str, Optional[str]]):
    """Merge Phase 2 products that differ only by position (x_1/x_2)
    and gamma label (mu/nu).

    Definitions become position/gamma-free; position and gamma are
    re-attached in the term assembly lines.

    Returns (new_phase2_defs, new_terms).
    """
    old_to_canonical = {}   # full old name -> canonical base name
    old_name_info = {}      # full old name -> (position, gamma)

    new_phase2_defs = []
    counter = 0

    for level, product_names, pair_counts in phase2_defs:
        canonical_groups = OrderedDict()  # canonical_def -> [(pair_key, prod_name)]

        for pair_key, prod_name in product_names.items():
            pos = product_positions.get(prod_name)
            gamma = product_gammas.get(prod_name)
            old_name_info[prod_name] = (pos, gamma)

            canon_key = _canonicalize_pair_key(pair_key, old_to_canonical)

            if canon_key not in canonical_groups:
                canonical_groups[canon_key] = []
            canonical_groups[canon_key].append((pair_key, prod_name))

        new_product_names = OrderedDict()
        new_pair_counts = {}

        for canon_key, members in canonical_groups.items():
            counter += 1
            # Check if any member has gamma — if so, use _mu in canonical name
            has_gamma = any(product_gammas.get(old_name) is not None
                           for _, old_name in members)
            canonical_name = f"prod_vec{counter}" + ("_mu" if has_gamma else "")
            # Display key: restore gamma_mu label in definitions
            display_key = (canon_key.replace('gamma |v>', 'gamma_mu |v>')
                           if has_gamma else canon_key)

            total_count = sum(pair_counts[pk] for pk, _ in members)
            new_product_names[display_key] = canonical_name
            new_pair_counts[display_key] = total_count

            for _, old_name in members:
                old_to_canonical[old_name] = canonical_name

        new_phase2_defs.append((level, new_product_names, new_pair_counts))

    # Rewrite terms
    new_terms = []
    for name, coef, traces in terms:
        new_traces = []
        for trace_str, elements in traces:
            new_elements = [_rewrite_term_element(e, old_to_canonical, old_name_info)
                            for e in elements]
            new_traces.append((trace_str, new_elements))
        new_terms.append((name, coef, new_traces))

    return new_phase2_defs, new_terms


def _is_bare_matrix(elem: str) -> bool:
    """Check if element is a bare matrix (Pi field or prod_Pi), not a vector or prod_vec."""
    return elem.startswith("Pi(") or re.match(r'prod_Pi\d+$', elem) is not None


def _absorb_bare_matrices(phase2_defs, terms, product_positions, product_gammas):
    """Absorb bare matrices adjacent to gamma |v> in connected traces.

    For each gamma_X |v>(pos) . bare_matrix pair:
      Step 1: Create gamma_mu |v> . bare_matrix as a new Phase 2 product
      Step 2: Create <w| . new_product as a Level 2 product (bra attaches)

    Returns (phase2_defs, terms).
    """
    max_num = 0
    for level, product_names, _ in phase2_defs:
        for _, prod_name in product_names.items():
            m = re.match(r'prod_vec(\d+)', prod_name)
            if m:
                max_num = max(max_num, int(m.group(1)))
    counter = max_num

    existing_levels = [level for level, _, _ in phase2_defs]
    next_level = max(existing_levels) + 1 if existing_levels else 1

    # ---- Step 1: gamma_X |v>(pos) . bare_matrix → new prod_vec ----

    pair_counts = {}
    pair_info = {}

    for name, coef, traces in terms:
        for trace_str, elements in traces:
            if len(elements) <= 1:
                continue
            for i in range(len(elements) - 1):
                a, b = elements[i], elements[i + 1]
                a_match = re.match(r'gamma_(\w+)\s+\|v>\((x_\d+)\)', a)
                if a_match and _is_bare_matrix(b):
                    pair_key = make_pair_key(a, b)
                    pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
                    pair_info[pair_key] = (a_match.group(2), a_match.group(1))

    if not pair_counts:
        return phase2_defs, terms

    # Group by canonical form (strip position/gamma from gamma |v>)
    canonical_groups = OrderedDict()
    for pair_key in sorted(pair_counts.keys()):
        bare = pair_key.split(' . ', 1)[1].strip()
        canon_key = f"gamma_mu |v> . {bare}"
        if canon_key not in canonical_groups:
            canonical_groups[canon_key] = []
        canonical_groups[canon_key].append(pair_key)

    product_map = {}
    new_L1_names = OrderedDict()
    new_L1_counts = {}
    l1_full_to_canonical = {}

    for canon_key, member_keys in canonical_groups.items():
        counter += 1
        canonical_name = f"prod_vec{counter}_mu"
        base_name = f"prod_vec{counter}"

        total_count = sum(pair_counts[pk] for pk in member_keys)
        new_L1_names[canon_key] = canonical_name
        new_L1_counts[canon_key] = total_count

        for pk in member_keys:
            pos, gamma = pair_info[pk]
            sub_name = f"{base_name}_{gamma}({pos})"
            product_map[pk] = sub_name
            product_positions[sub_name] = pos
            product_gammas[sub_name] = gamma
            l1_full_to_canonical[sub_name] = canonical_name

    terms = apply_substitution_to_terms(terms, product_map)

    # ---- Step 2: <w|(pos) . new_prod(pos) → Level 2 prod_vec ----

    pair_counts2 = {}
    pair_info2 = {}

    for name, coef, traces in terms:
        for trace_str, elements in traces:
            if len(elements) <= 1:
                continue
            for i in range(len(elements) - 1):
                a, b = elements[i], elements[i + 1]
                a_match = re.match(r'<w\|\((x_\d+)\)', a)
                if not a_match:
                    continue
                if b not in l1_full_to_canonical:
                    continue
                pos_a = a_match.group(1)
                pos_b = product_positions.get(b)
                if pos_a != pos_b:
                    continue
                pair_key = make_pair_key(a, b)
                pair_counts2[pair_key] = pair_counts2.get(pair_key, 0) + 1
                pair_info2[pair_key] = (pos_a, product_gammas.get(b))

    if not pair_counts2:
        phase2_defs = list(phase2_defs) + [(next_level, new_L1_names, new_L1_counts)]
        return phase2_defs, terms

    canonical_groups2 = OrderedDict()
    for pair_key in sorted(pair_counts2.keys()):
        b_part = pair_key.split(' . ', 1)[1].strip()
        b_canon = l1_full_to_canonical[b_part]
        canon_key = f"<w| . {b_canon}"
        if canon_key not in canonical_groups2:
            canonical_groups2[canon_key] = []
        canonical_groups2[canon_key].append(pair_key)

    product_map2 = {}
    new_L2_names = OrderedDict()
    new_L2_counts = {}

    for canon_key, member_keys in canonical_groups2.items():
        counter += 1
        canonical_name = f"prod_vec{counter}_mu"
        base_name = f"prod_vec{counter}"

        total_count = sum(pair_counts2[pk] for pk in member_keys)
        new_L2_names[canon_key] = canonical_name
        new_L2_counts[canon_key] = total_count

        for pk in member_keys:
            pos, gamma = pair_info2[pk]
            sub_name = f"{base_name}_{gamma}({pos})"
            product_map2[pk] = sub_name
            product_positions[sub_name] = pos
            product_gammas[sub_name] = gamma

    terms = apply_substitution_to_terms(terms, product_map2)

    phase2_defs = list(phase2_defs) + [
        (next_level, new_L1_names, new_L1_counts),
        (next_level + 1, new_L2_names, new_L2_counts),
    ]

    return phase2_defs, terms


def _reorder_phase2_levels(phase2_defs, terms):
    """Reorder Phase 2 levels so matrix-vector levels come before vector-vector
    levels. Swaps Level 2 (vector-vector) and Level 3 (matrix-vector) so that
    Levels 1 and 2 are both matrix-vector products.

    After reordering, renumbers all prod_vec names sequentially by level
    and updates all references in both phase2_defs and terms.

    Returns (new_phase2_defs, new_terms).
    """
    if len(phase2_defs) < 3:
        return phase2_defs, terms

    # Swap entries at index 1 (Level 2) and index 2 (Level 3)
    reordered = list(phase2_defs)
    reordered[1], reordered[2] = reordered[2], reordered[1]

    # Collect prod_vec numbers in original order
    old_order_nums = []
    for _, product_names, _ in phase2_defs:
        for _, prod_name in product_names.items():
            m = re.match(r'prod_vec(\d+)', prod_name)
            if m:
                old_order_nums.append(int(m.group(1)))

    # Collect prod_vec numbers in new (reordered) order
    new_order_nums = []
    for _, product_names, _ in reordered:
        for _, prod_name in product_names.items():
            m = re.match(r'prod_vec(\d+)', prod_name)
            if m:
                new_order_nums.append(int(m.group(1)))

    # Build rename map: old_number -> new_sequential_number
    rename_map = {}
    for new_seq, old_num in enumerate(new_order_nums, start=1):
        rename_map[old_num] = new_seq

    # Helper to rename prod_vec references in a string
    def rename(text):
        def replace(m):
            old_num = int(m.group(1))
            if old_num in rename_map:
                return f"prod_vec{rename_map[old_num]}"
            return m.group(0)
        return re.sub(r'prod_vec(\d+)', replace, text)

    # Apply renaming to phase2_defs
    new_phase2_defs = []
    for idx, (_, product_names, pair_counts) in enumerate(reordered):
        new_level = idx + 1
        new_product_names = OrderedDict()
        new_pair_counts = {}
        for pair_key, prod_name in product_names.items():
            new_pair_key = rename(pair_key)
            new_prod_name = rename(prod_name)
            new_product_names[new_pair_key] = new_prod_name
            new_pair_counts[new_pair_key] = pair_counts[pair_key]
        new_phase2_defs.append((new_level, new_product_names, new_pair_counts))

    # Apply renaming to terms
    new_terms = []
    for name, coef, traces in terms:
        new_traces = []
        for trace_str, elements in traces:
            new_elements = [rename(e) for e in elements]
            new_traces.append((trace_str, new_elements))
        new_terms.append((name, coef, new_traces))

    return new_phase2_defs, new_terms


def _create_current_vertex_product(phase2_defs, terms):
    """Factor out the universal current vertex <w|gamma_X|v>(pos).

    In this variant every term has exactly ONE standalone unsplit vertex
    (preserved by the carve-out pass). Replace each such element with a
    reference to a single canonical product. Also fold any adjacent
    <w|(pos) . gamma_X |v>(pos) split pairs into the same canonical
    product — these can survive Phase 2 when no Pi sits between bra/ket
    in the absorbed chain (e.g., Type 1 with the V_nu-end form).

    A new Phase 2 level is appended. The renumbering pass moves it to
    Level 1 as prod_vec1_mu.
    """
    max_num = 0
    for level, product_names, pair_counts in phase2_defs:
        for canon_key, prod_name in product_names.items():
            m = re.match(r'prod_vec(\d+)', prod_name)
            if m:
                max_num = max(max_num, int(m.group(1)))

    universal_num = max_num + 1
    canonical_name = f"prod_vec{universal_num}_mu"
    base_name = f"prod_vec{universal_num}"
    canon_key = "<w| . gamma_mu |v>"

    pattern_count = 0
    new_terms = []
    for name, coef, traces in terms:
        new_traces = []
        for trace_str, elements in traces:
            new_elements = []
            i = 0
            while i < len(elements):
                elem = elements[i]
                # 1. Unsplit standalone vertex.
                m = re.match(r'<w\|gamma_(\w+)\|v>\((x_\d+)\)', elem)
                if m:
                    gamma = m.group(1)
                    pos = m.group(2)
                    new_elements.append(f"{base_name}_{gamma}({pos})")
                    pattern_count += 1
                    i += 1
                    continue
                # 2. Split bra+ket adjacent pair at same position.
                if i < len(elements) - 1:
                    b = elements[i + 1]
                    a_match = re.match(r'<w\|\((x_\d+)\)', elem)
                    b_match = re.match(r'gamma_(\w+)\s+\|v>\((x_\d+)\)', b)
                    if a_match and b_match and a_match.group(1) == b_match.group(2):
                        pos = a_match.group(1)
                        gamma = b_match.group(1)
                        new_elements.append(f"{base_name}_{gamma}({pos})")
                        pattern_count += 1
                        i += 2
                        continue
                new_elements.append(elem)
                i += 1
            new_traces.append((trace_str, new_elements))
        new_terms.append((name, coef, new_traces))

    if pattern_count == 0:
        return phase2_defs, new_terms

    new_product_names = OrderedDict()
    new_product_names[canon_key] = canonical_name
    new_pair_counts = {canon_key: pattern_count}

    existing_levels = [level for level, _, _ in phase2_defs]
    new_level = max(existing_levels) + 1 if existing_levels else 1
    phase2_defs = list(phase2_defs) + [(new_level, new_product_names, new_pair_counts)]

    return phase2_defs, new_terms


def _renumber_with_universal_first(phase2_defs, terms):
    """Renumber prod_vec products so the universal current-vertex product
    (defined as <w| . gamma_mu |v>) becomes prod_vec1_mu, and is placed
    as the first Phase 2 level. All other prod_vec products are renumbered
    sequentially from 2, preserving their relative order across levels.

    Returns (new_phase2_defs, new_terms).
    """
    # Locate the universal product's old number.
    universal_old_num = None
    for level, product_names, _ in phase2_defs:
        for canon_key, prod_name in product_names.items():
            if canon_key == "<w| . gamma_mu |v>":
                m = re.match(r'prod_vec(\d+)', prod_name)
                if m:
                    universal_old_num = int(m.group(1))
                    break
        if universal_old_num is not None:
            break
    if universal_old_num is None:
        return phase2_defs, terms

    # Collect every prod_vec number in order of appearance.
    seen_order = []
    for level, product_names, _ in phase2_defs:
        for canon_key, prod_name in product_names.items():
            m = re.match(r'prod_vec(\d+)', prod_name)
            if m:
                n = int(m.group(1))
                if n not in seen_order:
                    seen_order.append(n)

    rename = {universal_old_num: 1}
    next_num = 2
    for n in seen_order:
        if n in rename:
            continue
        rename[n] = next_num
        next_num += 1

    def rename_text(text):
        return re.sub(
            r'prod_vec(\d+)',
            lambda m: f"prod_vec{rename.get(int(m.group(1)), int(m.group(1)))}",
            text,
        )

    # Split phase2_defs into the universal entry and everything else,
    # then put the universal as Level 1.
    universal_entry = None  # (pair_key, prod_name, count)
    other_levels = []  # list of (product_names, pair_counts)
    for level, product_names, pair_counts in phase2_defs:
        kept_names = OrderedDict()
        kept_counts = {}
        for canon_key, prod_name in product_names.items():
            new_pair = rename_text(canon_key)
            new_prod = rename_text(prod_name)
            if canon_key == "<w| . gamma_mu |v>":
                universal_entry = (new_pair, new_prod, pair_counts[canon_key])
            else:
                kept_names[new_pair] = new_prod
                kept_counts[new_pair] = pair_counts[canon_key]
        if kept_names:
            other_levels.append((kept_names, kept_counts))

    new_phase2_defs = []
    if universal_entry is not None:
        univ_names = OrderedDict()
        univ_names[universal_entry[0]] = universal_entry[1]
        univ_counts = {universal_entry[0]: universal_entry[2]}
        new_phase2_defs.append((1, univ_names, univ_counts))
    for idx, (names, counts) in enumerate(other_levels):
        new_phase2_defs.append((idx + 2, names, counts))

    new_terms = []
    for name, coef, traces in terms:
        new_traces = []
        for trace_str, elements in traces:
            new_traces.append((trace_str, [rename_text(e) for e in elements]))
        new_terms.append((name, coef, new_traces))

    return new_phase2_defs, new_terms


def _swap_position_gamma(elem: str) -> str:
    """Swap x_1 <-> x_2 and mu <-> nu in an element string."""
    # Temporary placeholders to avoid double-swap
    s = elem.replace('x_1', 'X_TEMP_1').replace('x_2', 'X_TEMP_2')
    s = s.replace('X_TEMP_1', 'x_2').replace('X_TEMP_2', 'x_1')
    s = s.replace('_mu', '_MU_TEMP').replace('_nu', '_NU_TEMP')
    s = s.replace('_MU_TEMP', '_nu').replace('_NU_TEMP', '_mu')
    return s


def _term_signature(traces):
    """Return a hashable signature for a term's traces, invariant to:
    - trace ordering (disconnected: Tr[A]*Tr[B] = Tr[B]*Tr[A])
    - cyclic permutations within a trace (Tr[A.B] = Tr[B.A])

    Each trace is normalized to its lexicographically smallest cyclic rotation.
    """
    trace_sigs = []
    for trace_str, elements in traces:
        n = len(elements)
        if n <= 1:
            trace_sigs.append(tuple(elements))
        else:
            rotations = [tuple(elements[i:] + elements[:i]) for i in range(n)]
            trace_sigs.append(min(rotations))
    return tuple(sorted(trace_sigs))


def _dedup_position_swap(terms):
    """Identify terms that are identical under x_1 <-> x_2 (and mu <-> nu) swap.

    Since both positions are summed over, swapped terms give the same result.
    Keep the first occurrence and mark the duplicate as a redirect.

    Returns list of (name, coef, traces_or_redirect) where traces_or_redirect is
    either the normal traces list or a string 'term_X (x1 <-> x2)'.
    """
    new_terms = []
    # Map from (coef, swapped_signature) -> name of the kept term
    seen = {}

    for name, coef, traces in terms:
        sig = _term_signature(traces)

        # Build the swapped version
        swapped_traces = []
        for trace_str, elements in traces:
            swapped_elements = [_swap_position_gamma(e) for e in elements]
            swapped_traces.append((trace_str, swapped_elements))
        swapped_sig = _term_signature(swapped_traces)

        # Check if we already have this term or its swap
        key_self = (coef, sig)
        key_swap = (coef, swapped_sig)

        if key_self in seen:
            # Exact duplicate (shouldn't happen, but handle it)
            new_terms.append((name, coef, f"{seen[key_self]} (x1 <-> x2)"))
        elif key_swap in seen and key_swap != key_self:
            # This term is the position-swapped version of an earlier term
            new_terms.append((name, coef, f"{seen[key_swap]} (x1 <-> x2)"))
        else:
            # New unique term — register under its own signature
            seen[key_self] = name
            new_terms.append((name, coef, traces))

    return new_terms


def _swap_phase2_levels(phase2_defs, terms, idx_a: int, idx_b: int):
    """Swap two Phase 2 level blocks (0-indexed) and renumber prod_vec
    sequentially in the new order. References in subsequent levels and in
    terms are updated accordingly.
    """
    if max(idx_a, idx_b) >= len(phase2_defs):
        return phase2_defs, terms

    reordered = list(phase2_defs)
    reordered[idx_a], reordered[idx_b] = reordered[idx_b], reordered[idx_a]

    new_order_nums = []
    for _, product_names, _ in reordered:
        for _, prod_name in product_names.items():
            m = re.match(r'prod_vec(\d+)', prod_name)
            if m:
                n = int(m.group(1))
                if n not in new_order_nums:
                    new_order_nums.append(n)

    rename_map = {old_num: new_seq
                  for new_seq, old_num in enumerate(new_order_nums, start=1)}

    def rename(text):
        return re.sub(
            r'prod_vec(\d+)',
            lambda m: f"prod_vec{rename_map.get(int(m.group(1)), int(m.group(1)))}",
            text,
        )

    new_phase2_defs = []
    for idx, (_, product_names, pair_counts) in enumerate(reordered):
        new_level = idx + 1
        new_product_names = OrderedDict()
        new_pair_counts = {}
        for pair_key, prod_name in product_names.items():
            new_pair_key = rename(pair_key)
            new_prod_name = rename(prod_name)
            new_product_names[new_pair_key] = new_prod_name
            new_pair_counts[new_pair_key] = pair_counts[pair_key]
        new_phase2_defs.append((new_level, new_product_names, new_pair_counts))

    new_terms = []
    for name, coef, traces in terms:
        new_traces = []
        for trace_str, elements in traces:
            new_traces.append((trace_str, [rename(e) for e in elements]))
        new_terms.append((name, coef, new_traces))

    return new_phase2_defs, new_terms


def optimize(terms, iterate: bool = False):
    """
    Pipeline:
      Carve-out:   per term, choose which current vertex stays unsplit
                   (standalone) so the C++/CUDA stage can FFT it once
                   globally; split the other for absorption.
      Phase 1:     matrix-matrix prod_Pi chains (cheap, position-free).
      Phase 2:     vector-matrix prod_vec chains, position-local. This
                   variant ALLOWS mat . <w|(pos) pairs (parent forbids
                   them) so bra-side dressing of the absorbed chain works.
      Dedup:       merge Phase 2 products differing only by position/gamma.
      Bare-mat:    absorb leftover bare Pi/prod_Pi adjacent to gamma|v>.
      Vertex:      replace every standalone unsplit vertex (and any
                   leftover adjacent <w|.gamma|v> split pairs) with a
                   reference to a single canonical product.
      Renumber:    promote the universal vertex product to prod_vec1_mu.
      Reorder:     swap Phase 2 Levels 3 and 4 so all bra-side closure
                   steps (<w|.prod_vec_X, prod_vec_X.gamma|v>) sit in the
                   final two levels and mat-vec dressing comes first.
      Pos-swap:    dedup terms equivalent under x_1 <-> x_2 / mu <-> nu.

    Returns (terms, phase1_defs, phase2_defs).
    """
    terms = carve_out_per_term(terms, iterate=iterate)
    terms, phase1_defs, product_positions = optimize_phase1(terms)
    terms, phase2_defs, product_gammas = optimize_phase2(terms, product_positions)
    phase2_defs, terms = deduplicate_phase2(phase2_defs, terms,
                                            product_positions, product_gammas)
    phase2_defs, terms = _absorb_bare_matrices(phase2_defs, terms,
                                               product_positions, product_gammas)
    phase2_defs, terms = _create_current_vertex_product(phase2_defs, terms)
    phase2_defs, terms = _renumber_with_universal_first(phase2_defs, terms)
    phase2_defs, terms = _swap_phase2_levels(phase2_defs, terms, 2, 3)
    terms = _dedup_position_swap(terms)
    return terms, phase1_defs, phase2_defs


# =============================================================================
# Output
# =============================================================================

def write_optimized_output(output_path: str, terms, phase1_defs, phase2_defs):
    """Write the optimized output file."""
    with open(output_path, 'w') as f:
        f.write("# Optimized pion meson field form (momentum space)\n")
        f.write("# <w|gamma_X|v>(pos) split into <w|(pos) and gamma_X |v>(pos)\n")
        f.write("# '.' denotes matrix multiplication (A2A index contraction)\n")
        f.write("# Tr[...] denotes trace over A2A indices\n")
        f.write("#\n")
        f.write("# Optimization order (compute in this order):\n")
        f.write("#   1. prod_Pi products  — matrix-matrix (Pi chain) products (cheap, position-free)\n")
        f.write("#   2. prod_vec products — vector-matrix products (position/gamma-independent)\n")
        f.write("#      Definitions are stripped of position (x_1/x_2) and gamma (mu/nu)\n")
        f.write("#      Position and gamma are re-attached in Phase 3 term assembly\n")
        f.write("#   3. Term assembly     — combine products into final traces\n")
        f.write("#\n")
        f.write("# Momentum conventions:\n")
        f.write("#   src_1 -> (p, t_src)\n")
        f.write("#   src_2 -> (-p, t_src + Delta)\n")
        f.write("#   snk_1 -> (k, t_snk)\n")
        f.write("#   snk_2 -> (-k, t_snk + Delta)\n")
        f.write("#   x_1, x_2 -> remain in position space (summed over)\n")
        f.write("#\n")

        # Phase 1 products
        if phase1_defs:
            f.write("# =========================================================\n")
            f.write("# Phase 1: Matrix-matrix products (Pi field chains)\n")
            f.write("# =========================================================\n")
            for level, product_names, pair_counts in phase1_defs:
                f.write(f"#\n")
                f.write(f"# Level {level}:\n")
                for pair_key, prod_name in product_names.items():
                    count = pair_counts[pair_key]
                    f.write(f"{prod_name} = {pair_key}    # appears {count} times\n")
            f.write("\n")

        # Phase 2 products
        if phase2_defs:
            f.write("# =========================================================\n")
            f.write("# Phase 2: Vector-matrix products (position/gamma-independent)\n")
            f.write("# =========================================================\n")
            for level, product_names, pair_counts in phase2_defs:
                f.write(f"#\n")
                f.write(f"# Level {level}:\n")
                for pair_key, prod_name in product_names.items():
                    count = pair_counts[pair_key]
                    f.write(f"{prod_name} = {pair_key}    # appears {count} times\n")
            f.write("\n")

        # Term definitions
        f.write("# =========================================================\n")
        f.write("# Phase 3: Term assembly\n")
        f.write("# =========================================================\n")
        for name, coef, traces in terms:
            if isinstance(traces, str):
                # Redirect: traces is e.g. "term_Type12_1 (x1 <-> x2)"
                f.write(f"{name} = {traces}\n")
            else:
                trace_strs = []
                for trace_str, elements in traces:
                    chain_str = " . ".join(elements)
                    trace_strs.append(f"Tr[{chain_str}]")
                f.write(f"{name} = {coef} * {' * '.join(trace_strs)}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 optimize_products.py <input_mom.txt>")
        sys.exit(1)

    input_path = sys.argv[1]

    if input_path.endswith('_mom.txt'):
        output_path = input_path.replace('_mom.txt', '_mom_optimized.txt')
    else:
        output_path = input_path.replace('.txt', '_optimized.txt')

    with open(input_path, 'r') as f:
        lines = f.readlines()

    terms = []
    for line in lines:
        line = line.rstrip('\n')
        if line.startswith('#') or not line.strip():
            continue
        parsed = parse_term_line(line)
        if parsed:
            terms.append(parsed)

    if not terms:
        print("No terms found in input file.")
        sys.exit(1)

    terms, phase1_defs, phase2_defs = optimize(terms)

    write_optimized_output(output_path, terms, phase1_defs, phase2_defs)

    # Summary
    p1_total = sum(len(pd) for _, pd, _ in phase1_defs)
    p2_total = sum(len(pd) for _, pd, _ in phase2_defs)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Terms:  {len(terms)}")
    print(f"\nPhase 1 (matrix-matrix): {p1_total} products in {len(phase1_defs)} level(s)")
    for level, product_names, pair_counts in phase1_defs:
        print(f"  Level {level}: {len(product_names)} products")
        for pair_key, prod_name in product_names.items():
            print(f"    {prod_name} ({pair_counts[pair_key]}x): {pair_key}")
    print(f"\nPhase 2 (vector-matrix, position/gamma-independent): {p2_total} products in {len(phase2_defs)} level(s)")
    for level, product_names, pair_counts in phase2_defs:
        print(f"  Level {level}: {len(product_names)} products")
        for pair_key, prod_name in product_names.items():
            print(f"    {prod_name} ({pair_counts[pair_key]}x): {pair_key}")


if __name__ == "__main__":
    main()
