"""
Parse *_mom_optimized.txt files and extract Phase 1 (matrix-matrix)
product definitions and Phase 3 (term assembly) into structured Python dicts.

Usage:
    python3 parse_contractions.py I2_pipi_cexpr_mom_optimized.txt
    python3 parse_contractions.py I2_pipi_EM_cexpr_mom_optimized.txt
"""

import sys
import re
from pprint import pprint


def parse_pi_operand(text):
    """Parse a single operand like 'Pi(-k, t_snk + Delta)' or 'prod_Pi3'.

    Returns a dict. For a Pi field:
        {'type': 'source', 'momentum': '-k', 'time': 't_snk + Delta'}
    For a reference to another product:
        {'type': 'product', 'ref': 'prod_Pi3'}
    """
    pi_match = re.match(r'Pi\((.+),\s*(.+)\)', text.strip())
    if pi_match:
        return {
            'type': 'source',
            'momentum': pi_match.group(1).strip(),
            'time': pi_match.group(2).strip()
        }

    prod_match = re.match(r'(prod_Pi\d+)', text.strip())
    if prod_match:
        return {'type': 'product', 'ref': prod_match.group(1)}

    raise ValueError(f"Could not parse operand: '{text}'")


def parse_product_line(line):
    """Parse a line like 'prod_Pi1 = Pi(-k, t_snk + Delta) . Pi(-p, t_src + Delta)'.

    Returns (name, left_operand_dict, right_operand_dict).
    """
    # strip trailing comments
    line = line.split('#')[0].strip()

    name, rhs = line.split('=', 1)
    name = name.strip()
    rhs = rhs.strip()

    # split on ' . ' to get left and right operands
    parts = rhs.split(' . ')
    if len(parts) != 2:
        raise ValueError(f"Expected exactly 2 operands in: '{line}'")

    left = parse_pi_operand(parts[0])
    right = parse_pi_operand(parts[1])

    return name, left, right


def parse_phase1(filepath):
    """Parse Phase 1 (matrix-matrix products) from an optimized contraction file.

    Returns:
        level1: dict of dicts for Level 1 products
        level2: dict of dicts for Level 2 products (may be empty)
    """
    level1 = {}
    level2 = {}

    current_level = None

    with open(filepath) as f:
        for line in f:
            stripped = line.strip()

            # detect level markers
            if stripped == '# Level 1:':
                current_level = 1
                continue
            if stripped == '# Level 2:':
                current_level = 2
                continue

            # stop at Phase 2 or Phase 3
            if 'Phase 2' in stripped or 'Phase 3' in stripped:
                current_level = None
                continue

            # skip comments and blank lines
            if stripped.startswith('#') or stripped == '':
                continue

            # only parse prod_Pi lines while in a level
            if current_level and stripped.startswith('prod_Pi'):
                name, left, right = parse_product_line(stripped)

                product = {'left': left, 'right': right}

                if current_level == 1:
                    level1[name] = product
                elif current_level == 2:
                    level2[name] = product

    return level1, level2


def parse_trace(text):
    """Parse the contents of a single Tr[...] block.

    E.g. 'prod_Pi4 . prod_Pi1' -> list of operand dicts.
    A single operand like 'prod_Pi4' also works.

    Returns a list of operand dicts (parsed by parse_pi_operand).
    """
    parts = text.split(' . ')
    return [parse_pi_operand(p) for p in parts]


def parse_term_line(line):
    """Parse a term assembly line like:
        term_ADT1_0 = coef_ADT1 * Tr[prod_Pi4] * Tr[prod_Pi1]
        term_ADT2_0 = coef_ADT2 * Tr[prod_Pi4 . prod_Pi1]

    Returns (name, coef, traces) where traces is a list of lists of operand dicts.
    """
    line = line.split('#')[0].strip()

    name, rhs = line.split('=', 1)
    name = name.strip()
    rhs = rhs.strip()

    # extract all Tr[...] blocks
    trace_blocks = re.findall(r'Tr\[([^\]]+)\]', rhs)

    # coefficient is everything before the first Tr, stripped of trailing *
    coef_part = rhs[:rhs.index('Tr[')].strip().rstrip('*').strip()
    coef = coef_part

    traces = [parse_trace(block) for block in trace_blocks]

    return name, coef, traces


def parse_phase3(filepath):
    """Parse Phase 3 (term assembly) from an optimized contraction file.

    Returns:
        terms: dict mapping term name to {'coef': str, 'traces': list of lists}
    """
    terms = {}
    in_phase3 = False

    with open(filepath) as f:
        for line in f:
            stripped = line.strip()

            if 'Phase 3' in stripped:
                in_phase3 = True
                continue

            if not in_phase3:
                continue

            if stripped.startswith('#') or stripped == '':
                continue

            if stripped.startswith('term_'):
                name, coef, traces = parse_term_line(stripped)
                terms[name] = {'coef': coef, 'traces': traces}

    return terms


# =========================================================
# Index resolution: map symbolic labels to flat buffer indices
# =========================================================

def momentum_time_index_to_flattened_index(p, t, Nmodes, Nt):
    return p * Nt * Nmodes * Nmodes + t * Nmodes * Nmodes


def make_momentum_map(p, min_p, k, min_k):
    return {
        'p': p,
        '-p': min_p,
        'k': k,
        '-k': min_k
    }


def make_time_map(tsrc, tsnk, Delta):
    return {
        't_src': tsrc,
        't_snk': tsnk,
        't_src + Delta': tsrc + Delta,
        't_snk + Delta': tsnk + Delta
    }


def level1_to_contractions(momenta_set, time_set, level1, Nmodes, Nt,
                           Ac={}, Bc={}, Cc={}):
    """Resolve Level 1 products to flat buffer indices for batched BLAS.

    Args:
        momenta_set: [p, -p, k, -k] buffer indices
        time_set: [tsrc, tsnk, Delta] numerical values
        level1: parsed Level 1 dict from parse_phase1
        Nmodes: number of A2A modes
        Nt: number of time slices
        Ac, Bc, Cc: accumulator dicts from previous calls

    Returns [Ac, Bc, Cc] where:
        Ac: {(momentum, time): flat_index} for left operands
        Bc: {(momentum, time): flat_index} for right operands
        Cc: {(prod_name, left_mom, left_t, right_mom, right_t): sequential_index}
    """
    Aadd = {}
    Badd = {}
    Cadd = {}

    momenta_map = make_momentum_map(momenta_set[0], momenta_set[1], momenta_set[2], momenta_set[3])
    time_map = make_time_map(time_set[0], time_set[1], time_set[2])

    for n, i in enumerate(level1):
        momentum = momenta_map[level1[i]['left']['momentum']]
        time = time_map[level1[i]['left']['time']]

        key = (level1[i]['left']['momentum'], level1[i]['left']['time'])
        Aadd[key] = momentum_time_index_to_flattened_index(momentum, time, Nmodes, Nt)

        momentum = momenta_map[level1[i]['right']['momentum']]
        time = time_map[level1[i]['right']['time']]

        key = (level1[i]['right']['momentum'], level1[i]['right']['time'])
        Badd[key] = momentum_time_index_to_flattened_index(momentum, time, Nmodes, Nt)

        key = (i, level1[i]['left']['momentum'], level1[i]['left']['time'],
               level1[i]['right']['momentum'], level1[i]['right']['time'])
        Cadd[key] = n

    Ac |= Aadd
    Bc |= Badd
    Cc |= Cadd

    return [Ac, Bc, Cc]


def level2_to_contractions(momenta_set, time_set, level1, level2, terms, Nmodes, Nt, Cc):
    """Resolve Level 2 contractions from both:
      1. Explicit level2 dict from parse_phase1 (EM case)
      2. Connected traces in Phase 3 terms with 2 product operands (zeroth order case)

    Returns [Aadd, Badd, flag_A, flag_B, Cadd]
      - Aadd/Badd: dict of {(name, 'left'/'right', mom_labels...) : buffer_index}
      - flag_A/flag_B: list of ints (0 = source A buffer, 1 = Level 1 result C buffer)
      - Cadd: dict mapping (name, left_labels..., right_labels...) to sequential Level 2 result index
    """
    Aadd = {}
    Badd = {}
    flag_A = []
    flag_B = []
    Cadd = {}

    momenta_map = make_momentum_map(momenta_set[0], momenta_set[1], momenta_set[2], momenta_set[3])
    time_map = make_time_map(time_set[0], time_set[1], time_set[2])

    # Build name -> C buffer index lookup from Cc
    cc_by_name = {key[0]: val for key, val in Cc.items()}

    def operand_labels(op):
        """Return a tuple of momentum/time labels for an operand.
        For a source: (momentum, time)
        For a product ref: look up in level1 to get (left_mom, left_time, right_mom, right_time)
        """
        if op['type'] == 'source':
            return (op['momentum'], op['time'])
        else:
            ref = op['ref']
            if ref in level1:
                L = level1[ref]['left']
                R = level1[ref]['right']
                return (L['momentum'], L['time'], R['momentum'], R['time'])
            else:
                return (ref,)

    # Collect all Level 2 products to compute
    products = {}

    # 1. Explicit Level 2 products (EM case)
    for name, product in level2.items():
        products[name] = product

    # 2. Connected traces from Phase 3 (traces with 2 product operands)
    for term_name, term in terms.items():
        for trace in term['traces']:
            if len(trace) == 2 and trace[0]['type'] == 'product' and trace[1]['type'] == 'product':
                pair = (trace[0]['ref'], trace[1]['ref'])
                already_exists = any(
                    p['left'].get('ref') == pair[0] and p['right'].get('ref') == pair[1]
                    for p in products.values()
                )
                if not already_exists:
                    synth_name = f"{pair[0]}.{pair[1]}"
                    products[synth_name] = {'left': trace[0], 'right': trace[1]}

    # Resolve each product
    for n, (name, product) in enumerate(products.items()):
        left = product['left']
        right = product['right']

        # Left operand — key includes resolved momentum/time labels
        a_key = (name, 'left') + operand_labels(left)
        if left['type'] == 'product':
            Aadd[a_key] = cc_by_name[left['ref']]
            flag_A.append(1)
        else:
            momentum = momenta_map[left['momentum']]
            time = time_map[left['time']]
            Aadd[a_key] = momentum_time_index_to_flattened_index(momentum, time, Nmodes, Nt)
            flag_A.append(0)

        # Right operand — key includes resolved momentum/time labels
        b_key = (name, 'right') + operand_labels(right)
        if right['type'] == 'product':
            Badd[b_key] = cc_by_name[right['ref']]
            flag_B.append(1)
        else:
            momentum = momenta_map[right['momentum']]
            time = time_map[right['time']]
            Badd[b_key] = momentum_time_index_to_flattened_index(momentum, time, Nmodes, Nt)
            flag_B.append(0)

        # Descriptive Cadd key: (name, left_labels..., right_labels...)
        c_key = (name,) + operand_labels(left) + operand_labels(right)
        Cadd[c_key] = n

    return [Aadd, Badd, flag_A, flag_B, Cadd]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <optimized_file.txt>")
        sys.exit(1)

    filepath = sys.argv[1]
    level1, level2 = parse_phase1(filepath)

    print(f"=== Level 1: {len(level1)} products ===")
    pprint(level1)

    if level2:
        print(f"\n=== Level 2: {len(level2)} products ===")
        pprint(level2)

    terms = parse_phase3(filepath)
    print(f"\n=== Phase 3: {len(terms)} terms ===")
    pprint(terms)
