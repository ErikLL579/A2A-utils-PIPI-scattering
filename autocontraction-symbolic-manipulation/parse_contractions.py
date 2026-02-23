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
