#!/usr/bin/env python3
"""
Simple parser/analyzer for OpenFOAM internalField lists (U and p)
Usage:
    python analyze_openfoam_fields.py <U_file> <p_file>

Outputs descriptive statistics for the vector field U and scalar field p.
"""
import sys
import re
import math
from statistics import mean, median, pstdev


def parse_list_file(file_path):
    """Return (type, n, list_of_strings) where type is 'vector' or 'scalar' and list contains raw entries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Find the internalField header up to the opening '(' that precedes the entries
    header_pattern = re.search(r"internalField\s+nonuniform\s+List<(?P<dtype>[^>]+)>\s*\n(?P<count>\d+)\s*\(\s*", text, re.S)
    if not header_pattern:
        raise ValueError('Could not find internalField header in file: ' + file_path)
    m = header_pattern

    dtype = m.group('dtype').strip()
    count = int(m.group('count'))
    # To avoid matching vectors in boundaryField sections, parse only the 'count' entries that
    # directly follow the opening '(' after the count. We'll find the '(' after the count and
    # then read entries (lines or vector entries) until we obtain `count` entries.
    # the '(' is included in our header match, so entries start immediately after header_pattern.end()
    open_paren_pos = text.find('(', m.start())
    # but the actual content starts at m.end(), which is after '(' and optional whitespace
    if open_paren_pos == -1:
        raise ValueError('Opening ( for list not found in: ' + file_path)
    # substring starting after '('
    after = text[m.end():]

    # split entries: for vector, entries are (x y z), for scalar each line is number
    if dtype.lower().startswith('vector'):
        # robustly find exactly `count` (x y z) entries
        vec_pattern = re.compile(r"\(\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s+([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s+([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*\)")
        entries = []
        for match in vec_pattern.finditer(after):
            entries.append("{} {} {}".format(match.group(1), match.group(2), match.group(3)))
            if len(entries) >= count:
                break
    else:
        # scalars: parse lines until we've read exactly `count` scalars
        scalar_pattern = re.compile(r"([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)")
        entries = []
        for match in scalar_pattern.finditer(after):
            entries.append(match.group(1))
            if len(entries) >= count:
                break

    return dtype, count, entries


def parse_vector(entry_str):
    parts = re.split(r"\s+", entry_str.strip())
    if len(parts) < 3:
        # sometimes vectors may contain trailing data; try parse first three
        raise ValueError('Bad vector entry: ' + entry_str)
    return float(parts[0]), float(parts[1]), float(parts[2])


def parse_scalar(entry_str):
    # handles numbers with optional trailing chars
    return float(entry_str.strip())


def stats(values):
    n = len(values)
    if n == 0:
        return None
    mn = min(values)
    mx = max(values)
    avg = mean(values)
    med = median(values)
    try:
        sd = pstdev(values)
    except Exception:
        sd = 0.0
    return {'count': n, 'min': mn, 'max': mx, 'mean': avg, 'median': med, 'std': sd}


def format_num(x):
    return f"{x:.8g}"


def analyze_U(file_path):
    dtype, count, entries = parse_list_file(file_path)
    if not dtype.lower().startswith('vector'):
        raise ValueError('U file is not vector type: ' + dtype)
    ux = []
    uy = []
    uz = []
    mag = []
    for e in entries:
        x, y, z = parse_vector(e)
        ux.append(x)
        uy.append(y)
        uz.append(z)
        mag.append(math.sqrt(x*x + y*y + z*z))

    stats_all = {
        'Ux': stats(ux),
        'Uy': stats(uy),
        'Uz': stats(uz),
        'mag': stats(mag),
    }

    # top 10 magnitudes
    indexed = list(enumerate(mag))
    indexed_sorted = sorted(indexed, key=lambda it: it[1], reverse=True)
    top10 = indexed_sorted[:10]

    return stats_all, top10, entries


def analyze_p(file_path):
    dtype, count, entries = parse_list_file(file_path)
    if not dtype.lower().startswith('scalar'):
        raise ValueError('p file not scalar type: ' + dtype)
    vals = [parse_scalar(e) for e in entries]
    s = stats(vals)
    # top10 largest
    indexed = list(enumerate(vals))
    indexed_sorted = sorted(indexed, key=lambda it: it[1], reverse=True)
    top10 = indexed_sorted[:10]

    # counts for duplicates: simple frequency of top 10
    from collections import Counter
    ctr = Counter(vals)
    return s, top10, ctr


def main(u_path, p_path):
    print('Analyzing files:')
    print('  U:', u_path)
    print('  p:', p_path)
    print('\n')

    u_stats, u_top10, u_entries = analyze_U(u_path)
    p_stats, p_top10, p_ctr = analyze_p(p_path)

    print('U (vector field) statistics:')
    for k, v in u_stats.items():
        print(f'  {k}: count={v["count"]}, min={format_num(v["min"])}, max={format_num(v["max"])}, mean={format_num(v["mean"])}, median={format_num(v["median"])}, std={format_num(v["std"]) if v["std"] is not None else "N/A"}')

    print('\nTop 10 largest magnitudes in U (index, magnitude, vector):')
    for idx, m in u_top10:
        v = u_entries[idx]
        print(f'  idx={idx}, mag={format_num(m)}, vec={v}')

    print('\nPressure p (scalar field) statistics:')
    print(f'  count={p_stats["count"]}, min={format_num(p_stats["min"])}, max={format_num(p_stats["max"])}, mean={format_num(p_stats["mean"])}, median={format_num(p_stats["median"])}, std={format_num(p_stats["std"]) if p_stats["std"] is not None else "N/A"}')

    print('\nTop 10 largest p values (index, value, count-in-dataset if duplicate):')
    for idx, v in p_top10:
        print(f'  idx={idx}, val={format_num(v)}, count={p_ctr[v]}')

    # other counts
    print('\nAdditional p counts summary:')
    total = p_stats['count']
    # negative values count
    neg_count = sum(1 for v in p_ctr if v < 0)
    print(f'  total values: {total}, unique values: {len(p_ctr)}, negative values unique count: {neg_count}')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python analyze_openfoam_fields.py <U_file> <p_file>')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
