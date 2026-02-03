"""
Transform Wick contraction expressions from quark propagator form to pion meson field form.

This script parses autocontraction output files and transforms them using A2A factorization.

Usage:
    python transform_wick.py <input_file>

Example:
    python transform_wick.py luchang-qlat-AC-output/I2_pipi_cexpr_original.txt
"""

import re
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Propagator:
    """
    Represents a quark propagator S_l(start, end).

    In lattice QCD notation, S(x,y) propagates a quark from y to x.
    """
    start: str  # e.g., "snk_1"
    end: str    # e.g., "src_1"

    def __repr__(self):
        return f"S_l({self.start},{self.end})"


@dataclass
class Trace:
    """
    Represents a trace over spinor indices: tr(gamma_5 * S * gamma_5 * S * ...)

    The propagators list contains the propagators in order as they appear in the trace.
    Each propagator is sandwiched between gamma_5 matrices.
    """
    propagators: List[Propagator]

    def __repr__(self):
        props = "*gamma_5*".join(str(p) for p in self.propagators)
        return f"tr(gamma_5*{props})"


@dataclass
class Term:
    """
    Represents a term in the Wick contraction: coefficient * (product of traces).

    For direct diagrams (ADT1): two separate traces multiplied together
    For cross diagrams (ADT2): single trace with four propagators
    """
    name: str              # e.g., "term_ADT1_0"
    coef_name: str         # e.g., "coef_ADT1"
    traces: List[Trace]    # list of traces (product of traces)

    def __repr__(self):
        traces_str = "*".join(str(t) for t in self.traces)
        return f"{self.name} = {self.coef_name} * {traces_str}"


@dataclass
class MesonField:
    """
    Represents a pion meson field Pi(position) or Pi(momentum, time).

    In A2A notation: Pi^{ij}(x) = <w^i(x)|gamma_5|v^j(x)>
    """
    position: str              # position label like "snk_1" or "src_1"
    momentum: Optional[str] = None   # momentum label like "p" or "-p"
    time: Optional[str] = None       # time label like "t_src" or "t_src + delta_t"

    def to_position_str(self) -> str:
        return f"Pi({self.position})"

    def to_momentum_str(self) -> str:
        if self.momentum and self.time:
            return f"Pi({self.momentum}, {self.time})"
        return self.to_position_str()


@dataclass
class MesonChain:
    """
    Represents a chain of meson fields with implicit A2A index contraction.

    For example: Pi(snk_1) . Pi(src_1) means sum over shared A2A index.
    """
    fields: List[MesonField]
    is_traced: bool = False  # True if this is Tr[...] over A2A indices

    def to_position_str(self) -> str:
        chain = " . ".join(f.to_position_str() for f in self.fields)
        if self.is_traced:
            return f"Tr[{chain}]"
        return f"({chain})"

    def to_momentum_str(self) -> str:
        chain = " . ".join(f.to_momentum_str() for f in self.fields)
        if self.is_traced:
            return f"Tr[{chain}]"
        return f"({chain})"


@dataclass
class MesonTerm:
    """
    Represents a transformed term in meson field notation.
    """
    name: str
    coef_name: str
    chains: List[MesonChain]  # product of meson chains

    def to_position_str(self) -> str:
        chains_str = " * ".join(c.to_position_str() for c in self.chains)
        return f"{self.name} = {self.coef_name} * {chains_str}"

    def to_momentum_str(self) -> str:
        chains_str = " * ".join(c.to_momentum_str() for c in self.chains)
        return f"{self.name} = {self.coef_name} * {chains_str}"


# =============================================================================
# Parser
# =============================================================================

def parse_propagator(prop_str: str) -> Propagator:
    """
    Parse a propagator string like 'S_l(snk_1,src_1)' into a Propagator object.
    """
    match = re.match(r'S_l\((\w+),(\w+)\)', prop_str)
    if not match:
        raise ValueError(f"Cannot parse propagator: {prop_str}")
    return Propagator(start=match.group(1), end=match.group(2))


def parse_trace(trace_str: str) -> Trace:
    """
    Parse a trace string like 'tr(gamma_5*S_l(snk_1,src_1)*gamma_5*S_l(src_1,snk_1))'
    into a Trace object.
    """
    # Extract content inside tr(...)
    match = re.match(r'tr\((.*)\)', trace_str)
    if not match:
        raise ValueError(f"Cannot parse trace: {trace_str}")

    content = match.group(1)

    # Find all propagators S_l(...) in the trace
    prop_pattern = r'S_l\(\w+,\w+\)'
    prop_matches = re.findall(prop_pattern, content)

    propagators = [parse_propagator(p) for p in prop_matches]
    return Trace(propagators=propagators)


def find_matching_paren(s: str, start: int) -> int:
    """
    Find the index of the closing parenthesis matching the opening paren at 'start'.
    """
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1


def parse_term_expression(expr_str: str) -> List[Trace]:
    """
    Parse the right-hand side of a term assignment (after the coefficient).
    Returns a list of Trace objects.

    Example input: 'tr(...)*tr(...)' or just 'tr(...)'
    """
    traces = []

    # Find all 'tr(' occurrences and extract full trace with nested parens
    i = 0
    while i < len(expr_str):
        idx = expr_str.find('tr(', i)
        if idx == -1:
            break

        # Find matching closing paren
        open_paren = idx + 2  # position of '(' in 'tr('
        close_paren = find_matching_paren(expr_str, open_paren)

        if close_paren != -1:
            trace_str = expr_str[idx:close_paren + 1]
            traces.append(parse_trace(trace_str))
            i = close_paren + 1
        else:
            i = idx + 1

    return traces


def parse_input_file(filepath: str) -> Tuple[Dict[str, str], List[Term], Dict[str, List[Tuple[int, str]]]]:
    """
    Parse the autocontraction output file.

    Returns:
        - diagram_types: dict mapping term prefixes to diagram type names
        - terms: list of Term objects
        - expressions: dict mapping expression names to list of (coefficient, term_name)
    """
    with open(filepath, 'r') as f:
        content = f.read()

    diagram_types = {}
    terms = []
    expressions = {}
    coef_values = {}

    lines = content.split('\n')

    for line in lines:
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue

        # Parse coefficient definitions
        coef_match = re.match(r'(coef_\w+)\s*=\s*(\d+)', line)
        if coef_match:
            coef_values[coef_match.group(1)] = int(coef_match.group(2))
            continue

        # Parse term definitions
        term_match = re.match(r'(term_(\w+)_(\d+))\s*=\s*(.+)', line)
        if term_match:
            term_name = term_match.group(1)
            diagram_type = term_match.group(2)
            term_index = term_match.group(3)
            rhs = term_match.group(4)

            # Extract coefficient name (appears before the first 'tr' or at start)
            coef_match = re.match(r'(coef_\w+)\s*\*\s*(.+)', rhs)
            if coef_match:
                coef_name = coef_match.group(1)
                expr_part = coef_match.group(2)
            else:
                coef_name = f"coef_{diagram_type}"
                expr_part = rhs

            # Skip ADT0 terms (vacuum/normalization)
            if 'tr(' not in expr_part:
                continue

            traces = parse_term_expression(expr_part)
            if traces:
                terms.append(Term(name=term_name, coef_name=coef_name, traces=traces))
            continue

        # Parse expression accumulations
        expr_match = re.match(r'exprs\[(\d+)\]\s*\+=\s*(-?\d+)\*(term_\w+)', line)
        if expr_match:
            expr_idx = int(expr_match.group(1))
            coef = int(expr_match.group(2))
            term_name = expr_match.group(3)

            expr_key = f"exprs[{expr_idx}]"
            if expr_key not in expressions:
                expressions[expr_key] = []
            expressions[expr_key].append((coef, term_name))
            continue

    return diagram_types, terms, expressions


# =============================================================================
# Transformer
# =============================================================================

def transform_trace_to_meson_chain(trace: Trace) -> MesonChain:
    """
    Transform a single spinor trace into a meson field chain.

    For a trace Tr[gamma_5 * S(a,b) * gamma_5 * S(c,d) * ...]:
    - Each propagator S(x,y) contributes to meson fields
    - The positions where meson fields sit are determined by the propagator endpoints

    Key insight from A2A factorization:
    - S(x,y) = sum_i |v^i(x)><w^i(y)|
    - After cyclic trace manipulation, we get products of Pi fields
    - Pi^{ij}(x) = <w^i(x)|gamma_5|v^j(x)>

    For Tr[gamma_5 * S(a,b) * gamma_5 * S(b,a)]:
    -> Pi(a) . Pi(b)  (with implicit A2A index contraction)
    """
    # Extract the positions where meson fields will be located
    # For each propagator S(start, end), the meson field sits at 'start'

    fields = []
    for prop in trace.propagators:
        fields.append(MesonField(position=prop.start))

    # All meson chains require a trace over A2A indices
    # - Direct diagram (2 propagators per spinor trace) -> Tr[Π · Π]
    # - Cross diagram (4 propagators in single spinor trace) -> Tr[Π · Π · Π · Π]
    is_traced = True

    return MesonChain(fields=fields, is_traced=is_traced)


def transform_term(term: Term) -> MesonTerm:
    """
    Transform a Wick contraction term to meson field notation.

    Direct diagram (2 separate spinor traces):
        Tr[...] * Tr[...] -> (Pi . Pi) * (Pi . Pi)

    Cross diagram (1 spinor trace with 4 propagators):
        Tr[...] -> Tr[Pi . Pi . Pi . Pi]  (A2A trace)
    """
    chains = []
    for trace in term.traces:
        chain = transform_trace_to_meson_chain(trace)
        chains.append(chain)

    return MesonTerm(name=term.name, coef_name=term.coef_name, chains=chains)


# =============================================================================
# Momentum Space Conversion
# =============================================================================

# Position to momentum/time mapping for I=2 pipi scattering
# Based on center-of-mass frame conventions from the reference paper
# Sources have momenta p, -p; sinks have momenta k, -k
MOMENTUM_MAP = {
    # Source positions (x1, x2)
    "src_1": ("p", "t_src"),
    "src_2": ("-p", "t_src + Delta"),
    # Sink positions (y1, y2)
    "snk_1": ("k", "t_snk"),
    "snk_2": ("-k", "t_snk + Delta"),
}


def apply_momentum_projection(meson_term: MesonTerm) -> MesonTerm:
    """
    Convert position-space meson fields to momentum space.

    Uses center-of-mass frame conventions:
    - p_src2 = -p_src1
    - p_snk2 = -p_snk1
    """
    new_chains = []

    for chain in meson_term.chains:
        new_fields = []
        for field in chain.fields:
            if field.position in MOMENTUM_MAP:
                mom, time = MOMENTUM_MAP[field.position]
                new_fields.append(MesonField(
                    position=field.position,
                    momentum=mom,
                    time=time
                ))
            else:
                new_fields.append(field)
        new_chains.append(MesonChain(fields=new_fields, is_traced=chain.is_traced))

    return MesonTerm(name=meson_term.name, coef_name=meson_term.coef_name, chains=new_chains)


# =============================================================================
# Output Generation
# =============================================================================

def generate_output(terms: List[Term],
                    expressions: Dict[str, List[Tuple[int, str]]],
                    output_pos_path: str,
                    output_mom_path: str):
    """
    Generate both position-space and momentum-space output files.
    """
    # Transform all terms
    meson_terms_pos = {}
    meson_terms_mom = {}

    for term in terms:
        meson_term = transform_term(term)
        meson_terms_pos[term.name] = meson_term
        meson_terms_mom[term.name] = apply_momentum_projection(meson_term)

    # Generate position-space output
    with open(output_pos_path, 'w') as f:
        f.write("# Pion meson field form (position space)\n")
        f.write("# Transformed from quark propagator form using A2A factorization\n")
        f.write("# Pi(x) represents the pion meson field matrix with implicit A2A indices\n")
        f.write("# '.' denotes matrix multiplication (A2A index contraction)\n")
        f.write("# Tr[...] denotes trace over A2A indices\n")
        f.write("#\n")
        f.write("# Coefficient definitions:\n")

        # Write coefficient definitions
        coef_names = sorted(set(t.coef_name for t in meson_terms_pos.values()))
        for coef in coef_names:
            f.write(f"{coef} = 1\n")
        f.write("\n")

        f.write("# Term definitions:\n")
        for term_name in sorted(meson_terms_pos.keys()):
            f.write(meson_terms_pos[term_name].to_position_str() + "\n")
        f.write("\n")

        f.write("# Expression definitions:\n")
        for expr_name, term_list in sorted(expressions.items()):
            for coef, term_name in term_list:
                if term_name in meson_terms_pos:
                    sign = "+" if coef >= 0 else ""
                    f.write(f"{expr_name} += {sign}{coef}*{term_name}\n")

    # Generate momentum-space output
    with open(output_mom_path, 'w') as f:
        f.write("# Pion meson field form (momentum space)\n")
        f.write("# Transformed from quark propagator form using A2A factorization\n")
        f.write("# Pi(p, t) represents the momentum-projected pion meson field matrix\n")
        f.write("# '.' denotes matrix multiplication (A2A index contraction)\n")
        f.write("# Tr[...] denotes trace over A2A indices\n")
        f.write("#\n")
        f.write("# Momentum conventions (center-of-mass frame):\n")
        f.write("#   src_1 -> (p, t_src)\n")
        f.write("#   src_2 -> (-p, t_src + Delta)\n")
        f.write("#   snk_1 -> (k, t_snk)\n")
        f.write("#   snk_2 -> (-k, t_snk + Delta)\n")
        f.write("#\n")
        f.write("# Coefficient definitions:\n")

        # Write coefficient definitions
        for coef in coef_names:
            f.write(f"{coef} = 1\n")
        f.write("\n")

        f.write("# Term definitions:\n")
        for term_name in sorted(meson_terms_mom.keys()):
            f.write(meson_terms_mom[term_name].to_momentum_str() + "\n")
        f.write("\n")

        f.write("# Expression definitions:\n")
        for expr_name, term_list in sorted(expressions.items()):
            for coef, term_name in term_list:
                if term_name in meson_terms_mom:
                    sign = "+" if coef >= 0 else ""
                    f.write(f"{expr_name} += {sign}{coef}*{term_name}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python transform_wick.py <input_file>")
        print("Example: python transform_wick.py luchang-qlat-AC-output/I2_pipi_cexpr_original.txt")
        sys.exit(1)

    input_path = sys.argv[1]

    # Derive output paths from input path
    base_name = input_path.replace('_original.txt', '').replace('.txt', '')
    if '/' in base_name:
        # Keep just the filename part for output
        base_name = base_name.split('/')[-1]

    output_pos_path = f"{base_name}_pos.txt"
    output_mom_path = f"{base_name}_mom.txt"

    print(f"Parsing input file: {input_path}")
    diagram_types, terms, expressions = parse_input_file(input_path)

    print(f"Found {len(terms)} terms")
    for term in terms:
        print(f"  {term.name}: {len(term.traces)} trace(s)")

    print(f"\nGenerating output files:")
    print(f"  Position space: {output_pos_path}")
    print(f"  Momentum space: {output_mom_path}")

    generate_output(terms, expressions, output_pos_path, output_mom_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
