"""
Parse *_mom_optimized.txt files and extract Phase 1 (matrix-matrix),
Phase 2 (vector-matrix), and Phase 3 (term assembly) into structured Python dicts.

Usage:
    python3 parse_contractions.py I2_pipi_cexpr_mom_optimized.txt
    python3 parse_contractions.py I2_pipi_EM_cexpr_mom_optimized.txt
"""

import sys
import re
from pprint import pprint


def parse_pi_operand(text):
    """Parse a single operand in a trace.

    Returns a dict:
      Pi(-k, t_snk + Delta)     -> {'type': 'source', 'momentum': '-k', 'time': 't_snk + Delta'}
      prod_Pi3                  -> {'type': 'product', 'ref': 'prod_Pi3'}
      prod_vec34_mu(x_1)        -> {'type': 'vec_product', 'ref': 'prod_vec34',
                                    'gamma': 'mu', 'position': 'x_1'}
      prod_vec1(x_2)            -> {'type': 'vec_product', 'ref': 'prod_vec1',
                                    'gamma': None, 'position': 'x_2'}
    """
    pi_match = re.match(r'Pi\((.+),\s*(.+)\)', text.strip())
    if pi_match:
        return {
            'type': 'source',
            'momentum': pi_match.group(1).strip(),
            'time': pi_match.group(2).strip()
        }

    prod_match = re.match(r'(prod_Pi\d+)$', text.strip())
    if prod_match:
        return {'type': 'product', 'ref': prod_match.group(1)}

    vec_match = re.match(r'(prod_vec\d+)(?:_(\w+))?\((x_\d+)\)', text.strip())
    if vec_match:
        return {
            'type': 'vec_product',
            'ref': vec_match.group(1),
            'gamma': vec_match.group(2),
            'position': vec_match.group(3)
        }

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

            # stop at Phase 2 or Phase 3 section headers
            if stripped.startswith('# Phase 2:') or stripped.startswith('# Phase 3:'):
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


def parse_phase2(filepath):
    """Parse all Phase 2 vector-matrix products (position/gamma-independent format).

    Handles all levels including the Level 3 current vertex product.

    Returns dict: {name: (vector_flag, matrix_label)}
        vector_flag: 0 = <w| (bra), 1 = gamma |v> (gamma-ket),
                     2 = current vertex (<w| . gamma |v>)
        matrix_label: tuple ('momentum', 'time') for source Pi,
                      string 'prod_PiN' or 'prod_vecN' for product ref,
                      or None for current vertex (flag=2)
    """
    result = {}
    in_phase2 = False

    def parse_matrix_label(text):
        """Convert matrix operand string to structured label.
        'Pi(-k, t_snk + Delta)' -> ('-k', 't_snk + Delta')
        'prod_Pi1' -> 'prod_Pi1'
        'prod_vec10' -> 'prod_vec10'
        """
        pi_match = re.match(r'Pi\((.+),\s*(.+)\)', text.strip())
        if pi_match:
            return (pi_match.group(1).strip(), pi_match.group(2).strip())
        return text.strip()

    with open(filepath) as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith('# Phase 2:'):
                in_phase2 = True
                continue
            if stripped.startswith('# Phase 3:'):
                break

            if not in_phase2:
                continue
            if stripped.startswith('#') or stripped == '':
                continue

            # Strip comments
            stripped = stripped.split('#')[0].strip()

            # Parse product name: prod_vecN or prod_vecN_mu
            name_match = re.match(r'(prod_vec\d+(?:_\w+)?)\s*=', stripped)
            if not name_match:
                continue
            name = name_match.group(1)

            # Get RHS
            _, rhs = stripped.split('=', 1)
            rhs = rhs.strip()
            parts = rhs.split(' . ')
            left = parts[0].strip()
            right = parts[1].strip()

            # Identify operand types
            left_is_bra = (left == '<w|')
            left_is_ket = bool(re.match(r'gamma_\w+\s+\|v>', left))
            right_is_bra = (right == '<w|')
            right_is_ket = bool(re.match(r'gamma_\w+\s+\|v>', right))

            # Current vertex: <w| . gamma_mu |v>
            if left_is_bra and right_is_ket:
                result[name] = (2, None)
            elif left_is_bra:
                result[name] = (0, parse_matrix_label(right))
            elif left_is_ket:
                result[name] = (1, parse_matrix_label(right))
            elif right_is_bra:
                result[name] = (0, parse_matrix_label(left))
            elif right_is_ket:
                result[name] = (1, parse_matrix_label(left))
            else:
                raise ValueError(f"No vector operand found in: {stripped}")

    return result


def phase2_to_buffer_indices(filepath):
    """Parse Phase 2 products into buffer index mappings for C++ code generation.

    For a single (p, k, t_src, t_snk) combination, the matrix buffer is indexed:
        0: Pi(p, t_src)
        1: Pi(-p, t_src + Delta)
        2: Pi(k, t_snk)
        3: Pi(-k, t_snk + Delta)
        4: prod_Pi1
        5: prod_Pi2
        ...
        J+3: prod_PiJ

    Returns (gamma_ket_indices, bra_indices):
        gamma_ket_indices: dict {prod_name: buffer_index} for L1+L2
            e.g. {'prod_vec1_mu': 3, 'prod_vec5_mu': 4, ...}
        bra_indices: dict {prod_name: (bra_label, ref)} for L3+L4+L5
            e.g. {'prod_vec29_mu': ('w', 'prod_vec10_mu'), ...,
                   'prod_vec57_mu': ('w', 'v')}
    """
    source_pi_map = {
        ('p', 't_src'): 0,
        ('-p', 't_src + Delta'): 1,
        ('k', 't_snk'): 2,
        ('-k', 't_snk + Delta'): 3,
    }

    gamma_ket_indices = {}
    bra_indices = {}

    in_phase2 = False
    current_level = None

    with open(filepath) as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith('# Phase 2:'):
                in_phase2 = True
                continue
            if stripped.startswith('# Phase 3:'):
                break
            if not in_phase2:
                continue

            # Track level changes
            level_match = re.match(r'# Level (\d+):', stripped)
            if level_match:
                current_level = int(level_match.group(1))
                continue

            if stripped.startswith('#') or stripped == '':
                continue

            # Strip comments
            stripped = stripped.split('#')[0].strip()

            # Parse: prod_vecN_mu = RHS
            name_match = re.match(r'(prod_vec\d+(?:_\w+)?)\s*=\s*(.*)', stripped)
            if not name_match:
                continue

            name = name_match.group(1)
            rhs = name_match.group(2).strip()
            parts = rhs.split(' . ')
            left = parts[0].strip()
            right = parts[1].strip() if len(parts) > 1 else ''

            if current_level in (1, 2):
                # gamma_mu |v> . matrix → extract matrix operand
                matrix = right

                # Source Pi → index 0..3
                pi_match = re.match(r'Pi\((.+),\s*(.+)\)', matrix)
                if pi_match:
                    mom = pi_match.group(1).strip()
                    time = pi_match.group(2).strip()
                    gamma_ket_indices[name] = source_pi_map[(mom, time)]
                else:
                    # prod_PiJ → index J + 3
                    j_match = re.match(r'prod_Pi(\d+)', matrix)
                    if j_match:
                        gamma_ket_indices[name] = int(j_match.group(1)) + 3

            elif current_level in (3, 4):
                # <w| . prod_vecM_mu → ('w', 'prod_vecM_mu')
                bra_indices[name] = ('w', right)

            elif current_level == 5:
                # <w| . gamma_mu |v> → ('w', 'v')
                bra_indices[name] = ('w', 'v')

    return gamma_ket_indices, bra_indices


def parse_phase3(filepath):
    """Parse Phase 3 (term assembly) from an optimized contraction file.

    Returns:
        terms: dict mapping term name to {'coef': str, 'traces': list of lists}
        duplicates: dict mapping redirect term name to target term name
            e.g. {'term_Type10_4': 'term_Type10_0', ...}
    """
    terms = {}
    duplicates = {}
    in_phase3 = False

    with open(filepath) as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith('# Phase 3:'):
                in_phase3 = True
                continue

            if not in_phase3:
                continue

            if stripped.startswith('#') or stripped == '':
                continue

            if stripped.startswith('term_'):
                redirect_match = re.match(r'(term_\S+)\s*=\s*(term_\S+)\s*\(x1\s*<->\s*x2\)', stripped)
                if redirect_match:
                    duplicates[redirect_match.group(1)] = redirect_match.group(2)
                    continue
                name, coef, traces = parse_term_line(stripped)
                terms[name] = {'coef': coef, 'traces': traces}

    return terms, duplicates


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

        key = (i, level1[i]['left']['momentum'], level1[i]['left']['time'])
        Aadd[key] = momentum_time_index_to_flattened_index(momentum, time, Nmodes, Nt)

        momentum = momenta_map[level1[i]['right']['momentum']]
        time = time_map[level1[i]['right']['time']]

        key = (i, level1[i]['right']['momentum'], level1[i]['right']['time'])
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


def parse_expr_factors(filepath, expr_index=4):
    """Parse the exprs[N] block from the original cexpr file.

    Extracts the numerical factor for each term_TypeX_Y line between
    the exprs[N] header and the next exprs[N+1] header.

    Args:
        filepath: path to the original cexpr file (e.g. I2_pipi_EM_cexpr_original.txt)
        expr_index: which exprs block to parse (default 4 for j_0 * j_0)

    Returns:
        factors: dict mapping term name to its factor as a string
            e.g. {'term_Type5_3': '2/9', 'term_Type1_0': '-1/9', ...}
    """
    factors = {}
    in_block = False
    header = f'exprs[{expr_index}]'
    next_header = f'exprs[{expr_index + 1}]'

    with open(filepath) as f:
        for line in f:
            stripped = line.strip()

            # Detect start of our block (the comment line)
            if stripped.startswith('#') and header in stripped:
                in_block = True
                continue

            # Stop at next block
            if in_block and next_header in stripped:
                break

            if not in_block:
                continue

            # Parse lines like: exprs[4] += 2/9*term_Type5_3
            match = re.match(
                r'exprs\[\d+\]\s*\+=\s*([+-]?\s*\d+(?:/\d+)?)\s*\*\s*(term_\S+)',
                stripped
            )
            if match:
                factor_str = match.group(1).replace(' ', '')
                term_name = match.group(2)
                factors[term_name] = factor_str

    return factors


def choose_appropriate_matrix(matrix_index, level1_len, level2_len=0):
    """Map a flat buffer index to the appropriate C++ Pion buffer expression.

    Buffer layout:
        0-3: Pion_source_fields (4 source Pi fields)
        4 .. 4+level1_len-1: Pion_product_fields_level_1
        4+level1_len .. : Pion_product_fields_level_2
    """
    if matrix_index < 4:
        return f"Pion_source_fields[{matrix_index}]"
    elif matrix_index >= 4 + level1_len:
        return f"Pion_product_fields_level_2[{matrix_index - level1_len - 4}]"
    else:
        return f"Pion_product_fields_level_1[{matrix_index - 4}]"


def prod_pi_to_matrix(name, level1_len):
    """Map a prod_Pi string (e.g. 'prod_Pi1') to the appropriate C++ buffer expression.

    Source Pi fields (Pi(p,t_src), etc.) are indexed 0-3 in Pion_source_fields.
    Level 1 products (prod_Pi1..prod_Pi8) start at index 4.
    Level 2 products (prod_Pi9+) follow after Level 1.
    """
    j = int(re.match(r'prod_Pi(\d+)', name).group(1))
    return choose_appropriate_matrix(j + 3, level1_len)


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

    phase2 = parse_phase2(filepath)
    print(f"\n=== Phase 2: {len(phase2)} products ===")
    for name, val in phase2.items():
        print(f"  {name}: {val}")

    gamma_ket_indices, bra_indices = phase2_to_buffer_indices(filepath)
    print(f"\n=== Phase 2 buffer indices: {len(gamma_ket_indices)} gamma-ket, {len(bra_indices)} bra ===")
    print("Gamma-ket (L1+L2):")
    for name, val in gamma_ket_indices.items():
        print(f"  {name}: {val}")
    print("Bra (L3+L4+L5):")
    for name, val in bra_indices.items():
        print(f"  {name}: {val}")

    terms, duplicates = parse_phase3(filepath)
    print(f"\n=== Phase 3: {len(terms)} terms, {len(duplicates)} duplicates ===")
    pprint(terms)
    if duplicates:
        print(f"\n=== Duplicates (x1 <-> x2): ===")
        pprint(duplicates)
