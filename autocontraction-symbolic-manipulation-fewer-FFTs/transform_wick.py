"""
Transform Wick contraction expressions from quark propagator form to pion meson field form.

This script parses autocontraction output files and transforms them using A2A factorization.
Handles both standard pipi scattering and EM corrections with gamma_mu insertions.

Usage:
    python transform_wick.py <input_file>

Example:
    python transform_wick.py luchang-qlat-AC-output/I2_pipi_cexpr_original.txt
    python transform_wick.py luchang-qlat-AC-output/I2_pipi_EM_cexpr_original.txt
"""

import re
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union


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
class GammaPropPair:
    """
    Represents a (gamma_matrix, propagator) pair in a trace.

    The trace structure is: gamma * S * gamma * S * ...
    Each pair represents the gamma that precedes the propagator.
    """
    gamma: str        # e.g., "gamma_5", "gamma_x", "gamma_y", etc.
    propagator: Propagator

    def __repr__(self):
        return f"({self.gamma}, {self.propagator})"


@dataclass
class Trace:
    """
    Represents a trace over spinor indices: tr(gamma * S * gamma * S * ...)

    Contains a list of (gamma, propagator) pairs in order.
    """
    pairs: List[GammaPropPair]

    def __repr__(self):
        parts = []
        for pair in self.pairs:
            parts.append(f"{pair.gamma}*{pair.propagator}")
        return f"tr({'*'.join(parts)})"

    def get_gamma_types(self) -> List[str]:
        """Return list of gamma matrix types in this trace."""
        return [pair.gamma for pair in self.pairs]

    def count_gamma_x(self) -> int:
        """Count occurrences of gamma_x in this trace."""
        return sum(1 for pair in self.pairs if pair.gamma == "gamma_x")


@dataclass
class Term:
    """
    Represents a term in the Wick contraction: coefficient * (product of traces).
    """
    name: str              # e.g., "term_Type10_0"
    coef_name: str         # e.g., "coef_Type10"
    traces: List[Trace]    # list of traces (product of traces)

    def __repr__(self):
        traces_str = "*".join(str(t) for t in self.traces)
        return f"{self.name} = {self.coef_name} * {traces_str}"

    def count_gamma_x(self) -> int:
        """Count total gamma_x occurrences across all traces."""
        return sum(t.count_gamma_x() for t in self.traces)


@dataclass
class MesonField:
    """
    Represents a pion meson field Pi(position) or Pi(momentum, time).

    In A2A notation: Pi^{ij}(x) = <w^i(x)|gamma_5|v^j(x)>
    """
    position: str
    momentum: Optional[str] = None
    time: Optional[str] = None

    def to_position_str(self) -> str:
        return f"Pi({self.position})"

    def to_momentum_str(self) -> str:
        if self.momentum and self.time:
            return f"Pi({self.momentum}, {self.time})"
        return self.to_position_str()


@dataclass
class CurrentVertex:
    """
    Represents a current vertex <w^i|gamma_mu|v^j>(z).

    This is NOT converted to a meson field - it remains as a vertex
    for EM current insertions.
    """
    position: str      # position label like "x_1" or "x_2"
    gamma: str         # gamma matrix type, e.g., "gamma_x"

    def to_position_str(self) -> str:
        # Convert gamma_x to gamma_mu or gamma_nu based on position
        # x_1 (z_1) gets gamma_mu, x_2 (z_2) gets gamma_nu
        if self.gamma == "gamma_x":
            if self.position == "x_1":
                gamma_out = "gamma_mu"
            elif self.position == "x_2":
                gamma_out = "gamma_nu"
            else:
                gamma_out = "gamma_mu"  # fallback
        else:
            gamma_out = self.gamma
        return f"<w|{gamma_out}|v>({self.position})"

    def to_momentum_str(self) -> str:
        # Current vertices stay in position space (summed over)
        return self.to_position_str()


# Union type for chain elements
ChainElement = Union[MesonField, CurrentVertex]


@dataclass
class MesonChain:
    """
    Represents a chain of meson fields and/or current vertices with implicit A2A index contraction.
    """
    elements: List[ChainElement]
    is_traced: bool = True

    def to_position_str(self) -> str:
        chain = " . ".join(e.to_position_str() for e in self.elements)
        if self.is_traced:
            return f"Tr[{chain}]"
        return f"({chain})"

    def to_momentum_str(self) -> str:
        chain = " . ".join(e.to_momentum_str() for e in self.elements)
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
    chains: List[MesonChain]

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
    Parse a trace string into a Trace object with (gamma, propagator) pairs.

    Example: 'tr(gamma_x*S_l(x_1,snk_1)*gamma_5*S_l(snk_1,x_2)*gamma_x*S_l(x_2,src_1)*gamma_5*S_l(src_1,x_1))'
    """
    # Extract content inside tr(...)
    match = re.match(r'tr\((.*)\)', trace_str)
    if not match:
        raise ValueError(f"Cannot parse trace: {trace_str}")

    content = match.group(1)

    # Pattern to match gamma_* followed by *S_l(...)
    # This handles: gamma_5, gamma_x, gamma_y, gamma_z, gamma_t
    pattern = r'(gamma_[a-z0-9]+)\*S_l\((\w+),(\w+)\)'

    pairs = []
    for match in re.finditer(pattern, content):
        gamma = match.group(1)
        prop_start = match.group(2)
        prop_end = match.group(3)
        pairs.append(GammaPropPair(
            gamma=gamma,
            propagator=Propagator(start=prop_start, end=prop_end)
        ))

    return Trace(pairs=pairs)


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
    """
    traces = []

    i = 0
    while i < len(expr_str):
        idx = expr_str.find('tr(', i)
        if idx == -1:
            break

        open_paren = idx + 2
        close_paren = find_matching_paren(expr_str, open_paren)

        if close_paren != -1:
            trace_str = expr_str[idx:close_paren + 1]
            traces.append(parse_trace(trace_str))
            i = close_paren + 1
        else:
            i = idx + 1

    return traces


def parse_input_file(filepath: str, filter_gamma_x: bool = False) -> Tuple[Dict[str, str], List[Term], Dict[str, List[Tuple[int, str]]]]:
    """
    Parse the autocontraction output file.

    Args:
        filepath: path to input file
        filter_gamma_x: if True, only keep terms with exactly 2 gamma_x matrices

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

            # Extract coefficient name
            coef_match = re.match(r'(coef_\w+)\s*\*\s*(.+)', rhs)
            if coef_match:
                coef_name = coef_match.group(1)
                expr_part = coef_match.group(2)
            else:
                coef_name = f"coef_{diagram_type}"
                expr_part = rhs

            # Skip terms without traces
            if 'tr(' not in expr_part:
                continue

            traces = parse_term_expression(expr_part)
            if traces:
                term = Term(name=term_name, coef_name=coef_name, traces=traces)

                # Filter for gamma_x if requested
                if filter_gamma_x:
                    if term.count_gamma_x() == 2:
                        terms.append(term)
                else:
                    terms.append(term)
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
    Transform a single spinor trace into a chain of meson fields and/or current vertices.

    For each (gamma, propagator) pair in the trace:
    - The position is the 'start' of the propagator (where <w| and |v> meet)
    - If gamma is gamma_5: create a MesonField Pi(position)
    - If gamma is gamma_x (or other): create a CurrentVertex <w|gamma|v>(position)
    """
    elements = []

    for pair in trace.pairs:
        position = pair.propagator.start

        if pair.gamma == "gamma_5":
            elements.append(MesonField(position=position))
        else:
            # gamma_x, gamma_y, gamma_z, gamma_t -> current vertex
            elements.append(CurrentVertex(position=position, gamma=pair.gamma))

    return MesonChain(elements=elements, is_traced=True)


def transform_term(term: Term) -> MesonTerm:
    """
    Transform a Wick contraction term to meson field notation.
    """
    chains = []
    for trace in term.traces:
        chain = transform_trace_to_meson_chain(trace)
        chains.append(chain)

    return MesonTerm(name=term.name, coef_name=term.coef_name, chains=chains)


# =============================================================================
# Momentum Space Conversion
# =============================================================================

MOMENTUM_MAP = {
    "src_1": ("p", "t_src"),
    "src_2": ("-p", "t_src + Delta"),
    "snk_1": ("k", "t_snk"),
    "snk_2": ("-k", "t_snk + Delta"),
}


def apply_momentum_projection(meson_term: MesonTerm) -> MesonTerm:
    """
    Convert position-space meson fields to momentum space.
    CurrentVertex elements remain in position space (they are summed over).
    """
    new_chains = []

    for chain in meson_term.chains:
        new_elements = []
        for elem in chain.elements:
            if isinstance(elem, MesonField):
                if elem.position in MOMENTUM_MAP:
                    mom, time = MOMENTUM_MAP[elem.position]
                    new_elements.append(MesonField(
                        position=elem.position,
                        momentum=mom,
                        time=time
                    ))
                else:
                    new_elements.append(elem)
            else:
                # CurrentVertex stays as is
                new_elements.append(elem)
        new_chains.append(MesonChain(elements=new_elements, is_traced=chain.is_traced))

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
        f.write("# Pi(x) = pion meson field matrix\n")
        f.write("# <w|gamma_mu|v>(z) = current vertex (summed over position z)\n")
        f.write("# '.' denotes matrix multiplication (A2A index contraction)\n")
        f.write("# Tr[...] denotes trace over A2A indices\n")
        f.write("#\n")

        f.write("# Term definitions:\n")
        for term_name in sorted(meson_terms_pos.keys()):
            f.write(meson_terms_pos[term_name].to_position_str() + "\n")

    # Generate momentum-space output
    with open(output_mom_path, 'w') as f:
        f.write("# Pion meson field form (momentum space)\n")
        f.write("# Transformed from quark propagator form using A2A factorization\n")
        f.write("# Pi(p, t) = momentum-projected pion meson field matrix\n")
        f.write("# <w|gamma_mu|v>(z) = current vertex (summed over position z)\n")
        f.write("# '.' denotes matrix multiplication (A2A index contraction)\n")
        f.write("# Tr[...] denotes trace over A2A indices\n")
        f.write("#\n")
        f.write("# Momentum conventions:\n")
        f.write("#   src_1 -> (p, t_src)\n")
        f.write("#   src_2 -> (-p, t_src + Delta)\n")
        f.write("#   snk_1 -> (k, t_snk)\n")
        f.write("#   snk_2 -> (-k, t_snk + Delta)\n")
        f.write("#   x_1, x_2 -> remain in position space (summed over)\n")
        f.write("#\n")

        f.write("# Term definitions:\n")
        for term_name in sorted(meson_terms_mom.keys()):
            f.write(meson_terms_mom[term_name].to_momentum_str() + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python transform_wick.py <input_file> [--filter-gamma-x]")
        print("Example: python transform_wick.py luchang-qlat-AC-output/I2_pipi_cexpr_original.txt")
        print("         python transform_wick.py luchang-qlat-AC-output/I2_pipi_EM_cexpr_original.txt --filter-gamma-x")
        sys.exit(1)

    input_path = sys.argv[1]
    filter_gamma_x = "--filter-gamma-x" in sys.argv

    # Derive output paths from input path
    base_name = input_path.replace('_original.txt', '').replace('.txt', '')
    if '/' in base_name:
        base_name = base_name.split('/')[-1]

    output_pos_path = f"{base_name}_pos.txt"
    output_mom_path = f"{base_name}_mom.txt"

    print(f"Parsing input file: {input_path}")
    if filter_gamma_x:
        print("Filtering for terms with exactly 2 gamma_x matrices")

    diagram_types, terms, expressions = parse_input_file(input_path, filter_gamma_x=filter_gamma_x)

    print(f"Found {len(terms)} terms")
    if len(terms) <= 20:
        for term in terms:
            gamma_count = term.count_gamma_x()
            print(f"  {term.name}: {len(term.traces)} trace(s), {gamma_count} gamma_x")

    print(f"\nGenerating output files:")
    print(f"  Position space: {output_pos_path}")
    print(f"  Momentum space: {output_mom_path}")

    generate_output(terms, expressions, output_pos_path, output_mom_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
