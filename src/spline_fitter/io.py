"""
Airfoil File I/O
================

Functions for parsing and exporting airfoil coordinate data files.

This module handles multiple airfoil file formats:
    - Selig format: Name header followed by x y coordinate pairs
    - Lednicer format: Point counts header followed by upper/lower surfaces

The coordinates are automatically normalized to:
    - Leading edge at origin (0, 0)
    - Trailing edge at (1, 0)
    - Chord length of 1.0
"""
import math
import os


def get_precision(s_val):
    """Helper to find number of decimal places in a string."""
    if '.' in s_val:
        # Handle scientific notation like 1.23e-05 by ignoring the exponent part
        s_val = s_val.lower().split('e')[0]
        if '.' in s_val:
            return len(s_val.split('.')[1])
    return 0

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


def normalize_coords(coords):
    """
    Normalizes airfoil coordinates.
    1. Handles 0-100 scaling.
    2. Finds TE by averaging the first and last points.
    3. Finds LE as the point furthest from the TE center.
    4. Translates, rotates, and scales coordinates to place LE at (0,0) and TE at (1,0).
    """
    if not coords or len(coords) < 3:
        return coords

    # Preliminary check for 0-100 scaling
    large_coord_threshold = 2.0
    large_point_count = sum(1 for x, y in coords if abs(x) > large_coord_threshold or abs(y) > large_coord_threshold)
    if large_point_count > len(coords) / 2:
        coords = [(x / 100.0, y / 100.0) for x, y in coords]

    # 1. Find trailing edge (TE) center
    p_start = coords[0]
    p_end = coords[-1]
    te_center = ((p_start[0] + p_end[0]) / 2.0, (p_start[1] + p_end[1]) / 2.0)

    # 2. Find leading edge (LE)
    max_dist_sq = -1
    le_candidates = []
    for p in coords:
        dist_sq = (p[0] - te_center[0])**2 + (p[1] - te_center[1])**2
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            le_candidates = [p]
        elif dist_sq == max_dist_sq:
            le_candidates.append(p)

    # If multiple points are at max distance, average them for the LE
    if len(le_candidates) > 1:
        le_x = sum(p[0] for p in le_candidates) / len(le_candidates)
        le_y = sum(p[1] for p in le_candidates) / len(le_candidates)
        le = (le_x, le_y)
    else:
        le = le_candidates[0]

    # 3. Calculate chord length and angle for transformation
    chord_length = math.sqrt((te_center[0] - le[0])**2 + (te_center[1] - le[1])**2)
    if chord_length < 1e-6: # Avoid division by zero for degenerate cases
        return coords

    # Angle of the chord relative to the x-axis
    angle = math.atan2(te_center[1] - le[1], te_center[0] - le[0])
    cos_angle = math.cos(-angle)
    sin_angle = math.sin(-angle)

    # 4. Translate, rotate, and scale all points
    normalized_coords = []
    for x, y in coords:
        # Translate to move LE to origin
        tx, ty = x - le[0], y - le[1]
        # Rotate to align chord with x-axis
        rx = tx * cos_angle - ty * sin_angle
        ry = tx * sin_angle + ty * cos_angle
        # Scale by chord length
        normalized_coords.append((rx / chord_length, ry / chord_length))

    return normalized_coords

def _is_point_counts(x, y):
    """Whether a coordinate pair is a Lednicer header: two whole point counts.

    A Selig file's first point is the trailing edge, near (1, 0) or (100, 0).
    """
    return x >= 2 and y >= 2 and x == int(x) and y == int(y)


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


def parse_airfoil_data(filepath):
    """
    Parses an airfoil data file, automatically detecting the format.
    Handles Selig and Lednicer formats, comments, and 0-100 scaling.
    Returns the airfoil name from filename, name from content, coordinates, and max precision.
    """
    with open(filepath, encoding='utf-8', errors='ignore') as f:
        text = f.read()
    return parse_airfoil_text(text, os.path.splitext(os.path.basename(filepath))[0])


def parse_airfoil_text(text, filename_name=None):
    """
    Parses airfoil coordinates from text, by the rules of parse_airfoil_data
    (for example coordinates copied from a web page). ``filename_name`` is
    returned as the name from the filename, and is the name from content
    when the text has none.
    """
    lines = []
    for line in text.splitlines():
        # Remove comments and strip whitespace
        line_content = line.split('#', 1)[0].strip()
        if line_content:
            lines.append(line_content)

    if not lines:
        return None, None, None, 0

    content_name = filename_name # Default to filename if no name is in the file

    # Every line holding a coordinate pair, as floats and as written (for precision)
    rows = []
    for i, line in enumerate(lines):
        parts = line.split()
        try:
            # Clean strings by removing parentheses before float conversion
            x_str, y_str = parts[0].strip('()'), parts[1].strip('()')
            rows.append((float(x_str), float(y_str), x_str, y_str))
        except (ValueError, IndexError):
            if i == 0:
                content_name = line  # First line is a name
            # Otherwise skip malformed coordinate lines

    # Lednicer: the first row holds the upper and lower point counts
    if len(rows) > 4 and _is_point_counts(rows[0][0], rows[0][1]):
        n_upper, n_lower = int(rows[0][0]), int(rows[0][1])
        rows = rows[1:]
        coords = _lednicer_to_selig([(x, y) for x, y, _, _ in rows], n_upper, n_lower)
    else:
        coords = [(x, y) for x, y, _, _ in rows]

    if not coords:
        return None, None, None, 0

    max_precision = max(max(get_precision(x_str), get_precision(y_str)) for _, _, x_str, y_str in rows)
    coords = filter_coordinate_outliers(coords)
    coords = normalize_coords(coords)
    return filename_name, content_name, coords, max_precision

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
            f.write(str(name).strip() + "\n")
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
            f.write(str(name).strip() + "\n")
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
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue
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
            if i == 0:
                name = text
    if degree is None:
        match = re.search(r"\.deg(\d+)\.", os.path.basename(filepath))
        degree = int(match.group(1)) if match else None
    if not counts:
        return {"name": name, "degree": degree, "section_control_points": [points],
                "section_knots": None, "continuity": continuity}
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

