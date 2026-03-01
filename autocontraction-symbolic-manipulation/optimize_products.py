#!/usr/bin/env python3
"""
optimize_products.py

Post-processing script that reads momentum-space meson field expressions
from *_mom.txt and produces *_mom_optimized.txt with repeated adjacent
products factored out.

Usage:
    python3 optimize_products.py <input_mom.txt>

The optimization is structured in two phases to minimize expensive operations:
  Phase 1 (matrix-matrix): Build up Pi field chain products first (cheap)
  Phase 2 (vector-matrix): Attach <w| and |v> vectors, respecting position
           locality — never combine elements at different positions (x_1 vs x_2)
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
    (original_trace_str, split_elements) tuples.
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
        split_elements = split_current_vertices(elements)
        traces.append((trace_str, split_elements))

    return (name, coef, traces)


# =============================================================================
# Element classification and position tracking
# =============================================================================

def is_vector_element(elem: str) -> bool:
    """Check if an element is a raw vector (<w| or |v>) rather than a matrix."""
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
                # Always skip vector-vector pairs
                if is_vector_vector_pair(a, b):
                    continue
                # In matrix-only mode, skip any pair involving a vector
                if matrix_only and (is_vector_element(a) or is_vector_element(b)):
                    continue
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
    """Factor out remaining <w|(pos) . gamma_X |v>(pos) pairs as a single
    canonical product (the current vertex). Added as a new Phase 2 level.

    Returns (phase2_defs, terms).
    """
    # Count occurrences
    pattern_count = 0
    for name, coef, traces in terms:
        for trace_str, elements in traces:
            for i in range(len(elements) - 1):
                a, b = elements[i], elements[i + 1]
                a_match = re.match(r'<w\|\((x_\d+)\)', a)
                b_match = re.match(r'gamma_(\w+)\s+\|v>\((x_\d+)\)', b)
                if a_match and b_match and a_match.group(1) == b_match.group(2):
                    pattern_count += 1

    if pattern_count < 2:
        return phase2_defs, terms

    # Find the next prod_vec number
    max_num = 0
    for level, product_names, pair_counts in phase2_defs:
        for canon_key, prod_name in product_names.items():
            m = re.match(r'prod_vec(\d+)', prod_name)
            if m:
                max_num = max(max_num, int(m.group(1)))

    canonical_name = f"prod_vec{max_num + 1}_mu"
    canon_key = "<w| . gamma_mu |v>"

    new_product_names = OrderedDict()
    new_product_names[canon_key] = canonical_name
    new_pair_counts = {canon_key: pattern_count}

    existing_levels = [level for level, _, _ in phase2_defs]
    new_level = max(existing_levels) + 1 if existing_levels else 1
    phase2_defs = list(phase2_defs) + [(new_level, new_product_names, new_pair_counts)]

    # Substitute in terms
    new_terms = []
    for name, coef, traces in terms:
        new_traces = []
        for trace_str, elements in traces:
            new_elements = []
            i = 0
            while i < len(elements):
                if i < len(elements) - 1:
                    a, b = elements[i], elements[i + 1]
                    a_match = re.match(r'<w\|\((x_\d+)\)', a)
                    b_match = re.match(r'gamma_(\w+)\s+\|v>\((x_\d+)\)', b)
                    if a_match and b_match and a_match.group(1) == b_match.group(2):
                        pos = a_match.group(1)
                        gamma = b_match.group(1)
                        base_name = re.sub(r'_mu$', '', canonical_name)
                        new_elements.append(f"{base_name}_{gamma}({pos})")
                        i += 2
                        continue
                new_elements.append(elements[i])
                i += 1
            new_traces.append((trace_str, new_elements))
        new_terms.append((name, coef, new_traces))

    return phase2_defs, new_terms


def optimize(terms):
    """
    Two-phase optimization:
      Phase 1: Matrix-matrix products only (Pi chains) — cheap, position-free
      Phase 2: Vector-matrix products — expensive, position-local only
      Dedup: Merge Phase 2 products differing only by position/gamma
      Current vertex: Factor out remaining <w| . gamma |v> pairs

    Returns:
        (terms, phase1_defs, phase2_defs)
    """
    terms, phase1_defs, product_positions = optimize_phase1(terms)
    terms, phase2_defs, product_gammas = optimize_phase2(terms, product_positions)
    phase2_defs, terms = deduplicate_phase2(phase2_defs, terms,
                                            product_positions, product_gammas)
    phase2_defs, terms = _absorb_bare_matrices(phase2_defs, terms,
                                               product_positions, product_gammas)
    phase2_defs, terms = _create_current_vertex_product(phase2_defs, terms)
    phase2_defs, terms = _reorder_phase2_levels(phase2_defs, terms)
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
