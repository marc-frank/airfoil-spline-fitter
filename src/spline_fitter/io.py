"""
Airfoil File I/O
================

Functions for parsing and exporting airfoil coordinate data files.

This module handles multiple airfoil file formats:
    - Selig format: Name header followed by x y coordinate pairs
    - Lednicer format: Point counts header followed by upper/lower surfaces

The coordinates are normalized without turning them (the file's x axis
stays the chord line): coordinates already in chords are kept as written,
others are moved and scaled so the leading edge (the smallest x) is at
x = 0 and the trailing-edge midpoint at (1, 0).
"""
import math
import os
import re


def get_precision(s_val):
    """Decimal places of a number as written: ``0.0123`` has 4, ``1.2346e-03`` has 7
    and ``1.23e+01`` has 1."""
    mantissa, _, exponent = s_val.lower().replace('d', 'e').partition('e')
    decimals = len(mantissa.split('.')[1]) if '.' in mantissa else 0
    try:
        return decimals - int(exponent or 0)
    except ValueError:
        return decimals


def _file_precision(values):
    """Decimal places of a file's coordinates.

    Plain numbers may drop trailing zeros, so the most decimals any of them
    has is the precision. Numbers in e-notation keep their significant
    digits, so the largest ones are rounded the coarsest: the fewest
    decimals any of them has is the precision.
    """
    plain = [get_precision(v) for v in values if 'e' not in v.lower().replace('d', 'e')]
    scientific = [get_precision(v) for v in values if 'e' in v.lower().replace('d', 'e')]
    candidates = ([max(plain)] if plain else []) + ([min(scientific)] if scientific else [])
    return max(0, min(candidates)) if candidates else 0


def filter_coordinate_outliers(coords, threshold_factor=5.0):
    """
    Filters out significant outliers from a list of coordinates.
    An outlier is a point that is unusually far from the median of the points.
    """
    if not coords or len(coords) < 4:
        return coords

    # Find the median point (approximated by the point at the median index)
    # This is a simple but effective way to find a central point robust to outliers
    sorted_by_x = sorted(coords)
    median_point = sorted_by_x[len(sorted_by_x) // 2]

    # Calculate distances from the median point
    distances = [math.sqrt((p[0] - median_point[0])**2 + (p[1] - median_point[1])**2) for p in coords]

    # Use Interquartile Range (IQR) to find a robust measure of spread
    sorted_distances = sorted(distances)
    q1 = sorted_distances[len(sorted_distances) // 4]
    q3 = sorted_distances[3 * len(sorted_distances) // 4]
    iqr = q3 - q1

    # Define a distance threshold to identify outliers
    # Points far beyond the typical spread are considered outliers
    distance_threshold = q3 + threshold_factor * iqr

    # Filter out the outliers
    filtered_coords = [p for i, p in enumerate(coords) if distances[i] <= distance_threshold]

    # As a final sanity check, if filtering removes too many points, return original
    if len(filtered_coords) < len(coords) * 0.8:
        return coords

    return filtered_coords


# Coordinates whose leading edge (smallest x) and trailing-edge midpoint are
# this close to (0, 0) and (1, 0), in chords, are already normalized and kept
# exactly as written
UNIT_CHORD_TOLERANCE = 5e-3


def normalize_coords(coords):
    """
    Normalizes airfoil coordinates without turning them, so the x axis of
    the file stays the chord line (thickness and camber are then measured
    as airfoil databases quote them).
    1. Handles 0-100 scaling.
    2. Keeps coordinates that already run from x = 0 to a trailing-edge
       midpoint at (1, 0).
    3. Otherwise moves and scales them uniformly so the leading edge (the
       smallest x) is at x = 0 and the trailing-edge midpoint (the average
       of the first and last points) at (1, 0).
    """
    return _normalize(coords)[0]


def _normalize(coords):
    """:func:`normalize_coords`, and the factor it scaled lengths by."""
    if not coords or len(coords) < 3:
        return coords, 1.0

    scale = 1.0
    # Preliminary check for 0-100 scaling
    large_coord_threshold = 2.0
    large_point_count = sum(1 for x, y in coords if abs(x) > large_coord_threshold or abs(y) > large_coord_threshold)
    if large_point_count > len(coords) / 2:
        coords = [(x / 100.0, y / 100.0) for x, y in coords]
        scale = 0.01

    te_x = (coords[0][0] + coords[-1][0]) / 2.0
    te_y = (coords[0][1] + coords[-1][1]) / 2.0
    le_x = min(x for x, _ in coords)
    chord = te_x - le_x
    if chord < 1e-6:  # degenerate, or not in Selig order
        return coords, scale
    if (abs(le_x) <= UNIT_CHORD_TOLERANCE and abs(te_x - 1.0) <= UNIT_CHORD_TOLERANCE
            and abs(te_y) <= UNIT_CHORD_TOLERANCE):
        return coords, scale
    return [((x - le_x) / chord, (y - te_y) / chord) for x, y in coords], scale / chord


def _scaled_precision(precision, scale):
    """Decimal places of data written with ``precision`` decimals once its
    lengths are multiplied by ``scale``: a 0-100 file with 4 decimals has 6
    at unit chord. Rounded down unless within a few percent of a whole
    number, so the rounding error it implies is never too small."""
    if not scale > 0 or not math.isfinite(scale):
        return precision
    value = precision - math.log10(scale)
    nearest = round(value)
    return max(0, int(nearest if abs(value - nearest) < 0.05 else math.floor(value)))


def _is_point_counts(x, y):
    """Whether a coordinate pair is a Lednicer header: two whole point counts.

    A Selig file's first point is the trailing edge, near (1, 0) or (100, 0).
    """
    return x >= 2 and y >= 2 and x == int(x) and y == int(y)


def _is_lednicer(rows):
    """Whether parsed rows are a Lednicer file: point counts that roughly
    match the rows after them, which start at the leading edge (a Selig file
    starts at the trailing edge, as far from the leading edge as x goes)."""
    if len(rows) <= 4 or not _is_point_counts(rows[0][0], rows[0][1]):
        return False
    n_points = len(rows) - 1
    if abs(rows[0][0] + rows[0][1] - n_points) > max(2, n_points // 10):
        return False
    xs = [r[0] for r in rows[1:]]
    return xs[0] - min(xs) < max(xs) - xs[0]


def _is_stray_first_row(rows):
    """Whether the first row is a name that reads as numbers ("0012 64")
    rather than a point: it lies far outside the box of all other rows."""
    if len(rows) <= 4:
        return False
    xs, ys = [r[0] for r in rows[1:]], [r[1] for r in rows[1:]]
    margin = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys))
    x, y = rows[0][0], rows[0][1]
    return not (min(xs) - margin <= x <= max(xs) + margin
                and min(ys) - margin <= y <= max(ys) + margin)


def _lednicer_to_selig(coords, n_upper, n_lower):
    """
    Joins Lednicer surfaces (upper then lower, both LE -> TE) into Selig order.
    The header counts split the surfaces when they account for every point;
    otherwise the split is where x jumps back from the trailing to the leading edge.
    The leading-edge point is kept once when both surfaces list it.
    """
    if n_upper + n_lower != len(coords):
        steps = [coords[i + 1][0] - coords[i][0] for i in range(len(coords) - 1)]
        n_upper = steps.index(min(steps)) + 1
    upper, lower = coords[:n_upper], coords[n_upper:]
    if lower and lower[0] == upper[0]:
        lower = lower[1:]
    return list(reversed(upper)) + lower


def read_text(filepath):
    """The text of a data file: UTF-8 (with or without a byte order mark),
    UTF-16 (with a byte order mark, or told by its zero bytes), or else
    Windows-1252, so no character of a name is dropped."""
    with open(filepath, 'rb') as f:
        data = f.read()
    if data.startswith((b'\xff\xfe', b'\xfe\xff')):
        return data.decode('utf-16', errors='replace')
    head = data[:200]
    if b'\x00' in head:
        big_endian = head[0::2].count(0) > head[1::2].count(0)
        return data.decode('utf-16-be' if big_endian else 'utf-16-le', errors='replace')
    try:
        return data.decode('utf-8-sig')
    except UnicodeDecodeError:
        return data.decode('cp1252', errors='replace')


_FORTRAN_EXPONENT = re.compile(r'(?<=[0-9.])[dD](?=[+-]?[0-9])')
# A sign right after a digit starts the next number of fixed-width columns
# ("0.95000-0.01200"); an exponent's sign follows its letter instead
_GLUED_SIGN = re.compile(r'(?<=[0-9.])(?=[+-][0-9.])')


def _number_fields(line):
    """The fields of a coordinate line, as strings Python reads as floats.

    Fields are separated by whitespace, commas or semicolons, or by nothing
    before a sign (fixed-width columns). A line without a dot whose fields
    hold commas uses the decimal comma (``0,5 0,01``). Fortran ``D``
    exponents become ``e``; parentheses and quotes are dropped.
    """
    line = _GLUED_SIGN.sub(' ', line.replace(';', ' '))
    if ',' in line:
        if len(line.split()) >= 2 and '.' not in line:
            line = line.replace(',', '.')
        else:
            line = line.replace(',', ' ')
    return [_FORTRAN_EXPONENT.sub('e', field.strip('()"\'')) for field in line.split()]


# A name that starts with "#", which readers would skip as a comment, is
# written with a backslash before it
_ESCAPED_NAME = re.compile(r'^\\+#')


def _name_line(name):
    """The header line of a name: one that starts with ``#`` (or with
    backslashes before one) gets a backslash more, so it is not a comment."""
    name = str(name).strip()
    return '\\' + name if re.match(r'^\\*#', name) else name


def _name_of_line(line):
    """The name a header line holds: undoes :func:`_name_line`."""
    return line[1:] if _ESCAPED_NAME.match(line) else line


def parse_airfoil_data(filepath):
    """
    Parses an airfoil data file, automatically detecting the format.
    Handles Selig and Lednicer formats, comments, and 0-100 scaling.
    Returns the airfoil name from filename, name from content, coordinates, and max precision.
    """
    text = read_text(filepath)
    return parse_airfoil_text(text, os.path.splitext(os.path.basename(filepath))[0])


def parse_airfoil_text(text, filename_name=None):
    """
    Parses airfoil coordinates from text, by the rules of parse_airfoil_data
    (for example coordinates copied from a web page). ``filename_name`` is
    returned as the name from the filename, and is the name from content
    when the text has none.

    The precision is the decimal places of the coordinates once normalized
    to unit chord, so a 0-100 file written with 4 decimals has 6.
    """
    lines = []
    for line in text.lstrip('﻿').splitlines():
        # Remove comments and strip whitespace; the whole line is kept for
        # a name, which may hold a # ("NACA 0012 #2")
        line_content = line.split('#', 1)[0].strip()
        if line_content:
            lines.append((line_content, line.strip()))

    if not lines:
        return None, None, None, 0

    content_name = filename_name # Default to filename if no name is in the file
    name_line = lines[0][1]

    # Every line holding a coordinate pair, as floats and as written (for precision)
    rows = []
    first_is_row = False
    for i, (line, _) in enumerate(lines):
        parts = _number_fields(line)
        try:
            x_str, y_str = parts[0], parts[1]
            rows.append((float(x_str), float(y_str), x_str, y_str))
            first_is_row = first_is_row or i == 0
        except (ValueError, IndexError):
            if i == 0:
                content_name = _name_of_line(name_line)  # First line is a name
            # Otherwise skip malformed coordinate lines

    # A first line that reads as numbers ("0012 64") is a name when the
    # rows after it are a Lednicer file, or it lies far from all of them
    if first_is_row and not _is_lednicer(rows) and (
            _is_lednicer(rows[1:]) or _is_stray_first_row(rows)):
        content_name = name_line
        rows = rows[1:]

    # Lednicer: the first row holds the upper and lower point counts
    if _is_lednicer(rows):
        n_upper, n_lower = int(rows[0][0]), int(rows[0][1])
        rows = rows[1:]
        coords = _lednicer_to_selig([(x, y) for x, y, _, _ in rows], n_upper, n_lower)
    else:
        coords = [(x, y) for x, y, _, _ in rows]

    if not coords:
        return None, None, None, 0

    precision = _file_precision([v for _, _, x_str, y_str in rows for v in (x_str, y_str)])
    coords = filter_coordinate_outliers(coords)
    coords, scale = _normalize(coords)
    return filename_name, content_name, coords, _scaled_precision(precision, scale)

def export_airfoil_data(filepath, coords, name=None, precision=None):
    """
    Exports airfoil coordinates to a .dat file in Selig format.
    - filepath: destination path (e.g., 'airfoil.dat').
    - coords: iterable of (x, y) tuples in Selig ordering
              (upper TE -> LE, then lower LE -> TE), with LE not duplicated.
    - name: optional header line; if None, no header line is written.
    - precision: number of decimal places; defaults to 8 if None.
    """
    if not coords:
        raise ValueError("No coordinates to export.")

    # Clamp precision to a sensible range
    if precision is None:
        precision = 8
    precision = max(0, min(int(precision), 16))
    fmt = f"{{:.{precision}f}} {{:.{precision}f}}"

    # Ensure directory exists
    out_dir = os.path.dirname(os.path.abspath(filepath))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        if name:
            f.write(_name_line(name) + "\n")
        for x, y in coords:
            f.write(fmt.format(float(x), float(y)) + "\n")

SPLINE_FORMAT_TAG = "spline-fitter B-spline v2"


def export_spline_data(filepath, degree, section_control_points, name=None, precision=None,
                       section_knots=None, continuity=None):
    """
    Exports spline control points in Selig-like .dat format.

    The XY lines hold the control points of all sections, with each junction
    point written once. Degree, per-section control-point counts, knot
    vectors and junction continuity are stored in ``#`` comment lines, which
    plain .dat readers skip, so the file still loads as a point list.

    - filepath: destination path (e.g., 'airfoil.deg5.spline.dat').
    - degree: integer, degree of the B-spline.
    - section_control_points: list of control point lists, one per section.
      Each section's control points follow the curve direction.
      For n=2: [upper_ctrl_pts (TE->LE), lower_ctrl_pts (LE->TE)]
      For n=1: [single_ctrl_pts (TE->TE around airfoil)]
    - name: optional header line.
    - precision: number of decimal places; defaults to 8 if None.
    - section_knots: optional knot vector per section (default: open uniform).
    - continuity: optional junction continuity labels, e.g. ["G2"].
    """
    if not section_control_points or not any(section_control_points):
        raise ValueError("No control points to export.")

    # Clamp precision to a sensible range
    if precision is None:
        precision = 8
    precision = max(0, min(int(precision), 16))
    fmt = f"{{:.{precision}f}} {{:.{precision}f}}"

    if section_knots is None:
        from .spline.spline import create_open_uniform_knots
        section_knots = [create_open_uniform_knots(degree, len(c)) for c in section_control_points]
    if len(section_knots) != len(section_control_points):
        raise ValueError("Need one knot vector per section.")
    for knots, ctrl in zip(section_knots, section_control_points, strict=False):
        if len(knots) != len(ctrl) + degree + 1:
            raise ValueError("Knot vector length must be control points + degree + 1.")

    # Ensure directory exists
    out_dir = os.path.dirname(os.path.abspath(filepath))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        if name:
            f.write(_name_line(name) + "\n")
        f.write(f"# {SPLINE_FORMAT_TAG}\n")
        f.write(f"# degree {int(degree)}\n")
        f.write(f"# sections {len(section_control_points)}\n")
        if continuity:
            f.write("# continuity " + " ".join(continuity) + "\n")
        for section_idx, (ctrl_pts, knots) in enumerate(zip(section_control_points, section_knots, strict=False), 1):
            f.write(f"# section {section_idx} control_points {len(ctrl_pts)}\n")
            f.write(f"# knots {section_idx} " + " ".join(repr(float(k)) for k in knots) + "\n")
        # Write control points in order: section 0 fully, then section 1+ skipping first point (junction)
        for section_idx, ctrl_pts in enumerate(section_control_points):
            start_idx = 1 if section_idx > 0 else 0  # skip junction point for subsequent sections
            for x, y in ctrl_pts[start_idx:]:
                f.write(fmt.format(float(x), float(y)) + "\n")


def parse_spline_data(filepath):
    """
    Reads a file written by :func:`export_spline_data`.

    Returns:
        dict with ``name``, ``degree``, ``section_control_points`` (list of
        (x, y) lists), ``section_knots`` (list of lists, or None for files
        without knots) and ``continuity`` (list of labels or None).

    Files from before knots were stored hold only the XY lines; they are
    returned as one section with the degree taken from a ``.deg{N}.`` part
    of the filename, if present.
    """
    import re

    name = ""
    degree = None
    counts = {}
    knots = {}
    continuity = None
    points = []
    first_line = None
    for line in read_text(filepath).splitlines():
        text = line.strip()
        if not text:
            continue
        is_first = first_line is None
        if is_first:
            first_line = text
        if text.startswith("#"):
            words = text[1:].split()
            if len(words) >= 2 and words[0] == "degree":
                degree = int(words[1])
            elif len(words) >= 4 and words[0] == "section" and words[2] == "control_points":
                counts[int(words[1])] = int(words[3])
            elif len(words) >= 3 and words[0] == "knots":
                knots[int(words[1])] = [float(v) for v in words[2:]]
            elif len(words) >= 2 and words[0] == "continuity":
                continuity = words[1:]
            continue
        parts = text.split()
        try:
            points.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            if is_first:
                name = _name_of_line(text)
    if degree is None:
        match = re.search(r"\.deg(\d+)\.", os.path.basename(filepath))
        degree = int(match.group(1)) if match else None
    if not counts:
        return {"name": name, "degree": degree, "section_control_points": [points],
                "section_knots": None, "continuity": continuity}
    expected = sum(counts.values()) - (len(counts) - 1)
    if not name and len(points) == expected + 1 and not first_line.startswith("#"):
        # A name that reads as a point, like "0012 64": one point too many
        name = first_line
        points = points[1:]
    sections = []
    offset = 0
    for s in sorted(counts):
        n = counts[s]
        if s == min(counts):
            sections.append(points[offset:offset + n])
            offset += n
        else:
            sections.append([sections[-1][-1]] + points[offset:offset + n - 1])
            offset += n - 1
    if offset != len(points):
        raise ValueError(f"{filepath}: expected {offset} control points, found {len(points)}.")
    section_knots = [knots[s] for s in sorted(counts)] if len(knots) == len(counts) else None
    return {"name": name, "degree": degree, "section_control_points": sections,
            "section_knots": section_knots, "continuity": continuity}

