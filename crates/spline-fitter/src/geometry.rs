//! Airfoil data and its split into sections.

use std::collections::BTreeMap;

use bspline_varpro::centripetal_parameters;
use bspline_varpro::corners::{clear_corners, split_at};
pub use bspline_varpro::corners::{corners, Corner};

use crate::{bail, Result, Vec2};

/// Index of the point farthest from the trailing-edge midpoint (the first
/// on ties).
pub fn find_leading_edge_index(points: &[Vec2]) -> Result<usize> {
    if points.is_empty() {
        bail!("Cannot find leading edge in empty coordinate list");
    }
    let te = 0.5 * (points[0] + points[points.len() - 1]);
    let distances: Vec<f64> = points.iter().map(|p| (p - te).norm_squared()).collect();
    Ok((0..points.len()).fold(0, |best, i| if distances[i] > distances[best] { i } else { best }))
}

/// Chord-length parameters in `[0, 1]` of a polyline.
pub fn arc_length_parameters(points: &[Vec2]) -> Vec<f64> {
    centripetal_parameters(points, 1.0)
}

/// Indices splitting a polyline with parameters `u` into `count` pieces of
/// about equal parameter length, each at least one step long.
fn junctions(u: &[f64], count: usize) -> Vec<usize> {
    let mut j: Vec<usize> = (1..count)
        .map(|i| {
            let target = i as f64 / count as f64;
            (0..u.len()).fold(0, |best, k| if (u[k] - target).abs() < (u[best] - target).abs() { k } else { best })
        })
        .collect();
    for k in 0..j.len() {
        j[k] = j[k].max(if k == 0 { 1 } else { j[k - 1] + 1 });
    }
    for k in (0..j.len()).rev() {
        j[k] = j[k].min(if k + 1 == j.len() { u.len() - 2 } else { j[k + 1] - 1 });
    }
    j
}

/// Splits Selig-ordered coordinates into `count` sections sharing their end
/// points; returns the sections and the junction indices.
///
/// The layout of a section count, whatever the data: two sections split at
/// the leading edge; an odd count splits evenly by arc length; an even
/// count splits each side of the leading edge evenly. [`cut_into_sections`]
/// adds the corners of the data.
pub fn split_into_sections(coords: &[Vec2], count: usize) -> Result<(Vec<Vec<Vec2>>, Vec<usize>)> {
    if count < 1 {
        bail!("num_sections must be >= 1, got {count}");
    }
    if coords.len() < count + 1 {
        bail!("Need at least {} points for {count} sections, got {}", count + 1, coords.len());
    }
    let indices = match count {
        1 => vec![],
        2 => vec![find_leading_edge_index(coords)?.clamp(1, coords.len() - 2)],
        _ if count % 2 == 1 => junctions(&arc_length_parameters(coords), count),
        _ => {
            let le = find_leading_edge_index(coords)?;
            let (upper, lower) = (&coords[..=le], &coords[le..]);
            let half = count / 2;
            if upper.len() < half + 1 || lower.len() < half + 1 {
                bail!(
                    "Need at least {} points on each side of the leading edge for {count} sections, got {} and {}",
                    half + 1,
                    upper.len(),
                    lower.len()
                );
            }
            let mut indices = junctions(&arc_length_parameters(upper), half);
            indices.push(le);
            indices.extend(junctions(&arc_length_parameters(lower), half).iter().map(|i| i + le));
            indices
        }
    };
    Ok((split_at(coords, &indices), indices))
}

/// A junction of the layout this many points or fewer from a corner lies
/// on a side the corner is read from (two points each way): it gives way to
/// the corner.
pub const CORNER_SIDE: usize = 2;

/// How an outline is cut into sections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Split {
    /// Sections of the layout ([`split_into_sections`]).
    pub sections: usize,
    /// Cut at the clear corners of the data too.
    pub corners: bool,
    /// Points a section between a layout junction and a corner must have
    /// (what the fit of one needs: one more than its fewest control
    /// points); 0 for any.
    pub least: usize,
}

impl Split {
    /// The layout of a section count and the corners of the data, with the
    /// points a section beside a corner must have.
    pub fn at_corners(sections: usize, least: usize) -> Self {
        Split { sections, corners: true, least }
    }

    /// Whether a layout junction this many points from a corner gives way
    /// to it: on a side the corner is read from, or too near to leave a
    /// section that can be fitted.
    pub fn gives_way(&self, distance: usize) -> bool {
        distance <= CORNER_SIDE || distance + 1 < self.least
    }
}

impl From<usize> for Split {
    /// The layout of a section count alone.
    fn from(sections: usize) -> Self {
        Split { sections, corners: false, least: 0 }
    }
}

/// What a junction between two sections is: one of the layout, a corner of
/// the data, or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seam {
    /// A corner of the data: the sections meet there with G0, whatever the
    /// continuity of the layout's junctions.
    pub corner: bool,
    /// The junction of the layout it is (counted from 0), if it is one: its
    /// continuity is that junction's, unless it is a corner.
    pub layout: Option<usize>,
}

/// Cuts Selig-ordered coordinates into sections sharing their end points;
/// returns the sections, the junction indices and what each junction is.
///
/// The junctions are those of the layout ([`split_into_sections`]) and,
/// with `split.corners`, every clear corner of the data ([`corners`]). A
/// layout junction gives way to the nearest corner (the first of two as
/// near) when it is within [`CORNER_SIDE`] points of it, or so near that
/// the section between them would have fewer than `split.least` points: a
/// corner at the nose is one junction, not a corner and a leading-edge
/// junction a few points apart with a sliver between them. So only
/// sections between two corners (or a corner and an end) can be too short
/// to fit by their own points, and there is one section more for every
/// corner away from the layout's junctions.
pub fn cut_into_sections(coords: &[Vec2], split: Split) -> Result<(Vec<Vec<Vec2>>, Vec<usize>, Vec<Seam>)> {
    let (_, layout) = split_into_sections(coords, split.sections)?;
    let found = if split.corners { clear_corners(coords) } else { vec![] };
    let mut seams: BTreeMap<usize, Seam> = found.iter().map(|&c| (c, Seam { corner: true, layout: None })).collect();
    for (k, &j) in layout.iter().enumerate() {
        match found.iter().copied().filter(|c| split.gives_way(c.abs_diff(j))).min_by_key(|c| c.abs_diff(j)) {
            // The corner stands for the first layout junction it replaces.
            Some(corner) => {
                let seam = seams.get_mut(&corner).expect("every corner is a seam");
                seam.layout.get_or_insert(k);
            }
            None => {
                seams.insert(j, Seam { corner: false, layout: Some(k) });
            }
        }
    }
    let indices: Vec<usize> = seams.keys().copied().collect();
    Ok((split_at(coords, &indices), indices, seams.into_values().collect()))
}

/// Airfoil coordinates with their split into sections.
#[derive(Debug, Clone, PartialEq)]
pub struct AirfoilData {
    pub coordinates: Vec<Vec2>,
    pub sections: Vec<Vec<Vec2>>,
    pub junction_indices: Vec<usize>,
    /// What each junction is: of the layout, a corner, or both.
    pub seams: Vec<Seam>,
    /// How the sections were cut.
    pub split: Split,
    /// File name without directory and suffix ("" for pasted or created data).
    pub filename: String,
    pub name: String,
    /// Decimal places of the coordinates.
    pub precision: i32,
}

impl AirfoilData {
    /// The coordinates cut as `split` says: a section count for its layout
    /// alone, or a [`Split`] (a configuration's: `SplineConfig::split`).
    pub fn new(coordinates: Vec<Vec2>, split: impl Into<Split>, filename: &str, name: &str, precision: i32) -> Result<Self> {
        let split = split.into();
        let (sections, junction_indices, seams) = cut_into_sections(&coordinates, split)?;
        Ok(AirfoilData { coordinates, sections, junction_indices, seams, split, filename: filename.into(), name: name.into(), precision })
    }

    /// Sections of the airfoil: those of the layout, and one more for every
    /// corner away from the layout's junctions.
    pub fn num_sections(&self) -> usize {
        self.sections.len()
    }

    /// The same airfoil cut another way.
    pub fn resplit(&self, split: impl Into<Split>) -> Result<Self> {
        Self::new(self.coordinates.clone(), split, &self.filename, &self.name, self.precision)
    }

    /// Whether each section has a corner at one of its ends.
    pub fn at_corner(&self) -> Vec<bool> {
        let corner = |j: Option<&Seam>| j.is_some_and(|s| s.corner);
        (0..self.sections.len()).map(|i| corner(i.checked_sub(1).and_then(|k| self.seams.get(k))) || corner(self.seams.get(i))).collect()
    }

    /// The file name, else the name ("" without either): what default
    /// file names are made from.
    pub fn file_base(&self) -> &str {
        if self.filename.is_empty() {
            &self.name
        } else {
            &self.filename
        }
    }

    /// The name, else the file name, else `Airfoil`.
    pub fn display_name(&self) -> &str {
        [&self.name, &self.filename].into_iter().find(|s| !s.is_empty()).map_or("Airfoil", |s| s.as_str())
    }

    /// Point counts per section in Python list form: `[81, 81]`.
    pub fn section_sizes(&self) -> String {
        let sizes: Vec<String> = self.sections.iter().map(|s| s.len().to_string()).collect();
        format!("[{}]", sizes.join(", "))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use bspline_varpro::airfoil::Naca4;

    /// A wedge outline: straight sides meeting at the nose, `step` apart in x,
    /// with a flap-like kink at x = `kink` on the lower side when given.
    pub fn wedge(step: f64, kink: Option<f64>) -> Vec<Vec2> {
        let xs: Vec<f64> = (0..=(1.0 / step).round() as usize).map(|i| 1.0 - i as f64 * step).collect();
        let half = |x: f64| 0.1 * x.min(1.0 - x) + 0.02 * x;
        let lower = |x: f64| -half(x) - kink.map_or(0.0, |k| 0.5 * (x - k).max(0.0));
        xs.iter().map(|&x| Vec2::new(x, half(x))).chain(xs.iter().rev().skip(1).map(|&x| Vec2::new(x, lower(x)))).collect()
    }

    /// Saratov: 29 points, a round nose and a ridge at x 0.4 on the lower
    /// side.
    pub fn saratov() -> Vec<Vec2> {
        let xs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.025, 0.0125, 0.0];
        let top = [0.0, 0.0285, 0.055, 0.0775, 0.097, 0.1135, 0.125, 0.1175, 0.097, 0.08275, 0.0675, 0.04625, 0.03063, 0.02031, 0.0];
        let bottom = [-0.01469, -0.01938, -0.02175, -0.0205, -0.00925, 0.005, 0.0355, 0.07, 0.0625, 0.05, 0.0375, 0.025, 0.0125, 0.0];
        let lower_xs = [0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        xs.iter().zip(top).map(|(&x, y)| Vec2::new(x, y)).chain(lower_xs.iter().zip(bottom).map(|(&x, y)| Vec2::new(x, y))).collect()
    }

    /// A nose cut off straight (as rc0864c): smooth surfaces ending in a
    /// short face of three points, a corner at each end of it.
    pub fn cut_nose() -> Vec<Vec2> {
        let xs: Vec<f64> = (0..=125).map(|i| 1.0 - i as f64 * 0.008).collect();
        let half = |x: f64| 0.004 * (1.0 - x) + 0.1 * x * (1.0 - x);
        let upper = xs.iter().map(|&x| Vec2::new(x, half(x)));
        let lower = xs.iter().rev().map(|&x| Vec2::new(x, -half(x)));
        upper.chain([Vec2::new(-0.0002, 0.0)]).chain(lower).collect()
    }

    #[test]
    fn splits() {
        let pts = Naca4::new("2412", true).unwrap().points(81);
        let (s, j) = split_into_sections(&pts, 2).unwrap();
        assert_eq!((s[0].len(), s[1].len(), j), (81, 81, vec![80]));
        let (s, j) = split_into_sections(&pts, 3).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s.iter().map(Vec::len).sum::<usize>(), 161 + 2);
        assert!(j[0] < j[1]);
        let (s, j) = split_into_sections(&pts, 4).unwrap();
        assert_eq!(s.len(), 4);
        assert_eq!(j[1], 80);
        assert_eq!(split_into_sections(&pts, 1).unwrap().0.len(), 1);
        assert_eq!(split_into_sections(&pts[..3], 3).unwrap_err().0, "Need at least 4 points for 3 sections, got 3");
    }

    /// With corners, every clear corner of the data is a junction; the
    /// layout's junctions stay, except within two points of a corner.
    #[test]
    fn corners_cut_sections() {
        let cut = |pts: &[Vec2], sections: usize| {
            let (s, j, seams) = cut_into_sections(pts, Split::at_corners(sections, 0)).unwrap();
            assert_eq!((s.len(), seams.len()), (j.len() + 1, j.len()));
            assert!(s.windows(2).all(|w| w[0].last() == w[1].first()));
            (j, seams)
        };
        let (corner, layout) = (|k| Seam { corner: true, layout: k }, |k| Seam { corner: false, layout: Some(k) });
        // A sharp nose: the leading-edge junction is the corner, also as
        // the only junction of one section.
        let sharp = wedge(0.005, None);
        let le = sharp.iter().position(|p| p.x == 0.0).unwrap();
        assert_eq!(cut(&sharp, 2), (vec![le], vec![corner(Some(0))]));
        assert_eq!(cut(&sharp, 1), (vec![le], vec![corner(None)]));
        // A round nose and a ridge behind it (Saratov): the leading-edge
        // junction stays and the ridge is one more. With four sections the
        // layout's junction on the lower side, a point from the ridge,
        // gives way to it.
        let saratov = saratov();
        let (nose, ridge) = (14, 22);
        assert_eq!(clear_corners(&saratov), [ridge]);
        assert_eq!(cut(&saratov, 2), (vec![nose, ridge], vec![layout(0), corner(None)]));
        assert_eq!(cut(&saratov, 1), (vec![ridge], vec![corner(None)]));
        let four = split_into_sections(&saratov, 4).unwrap().1;
        assert_eq!((four[1], four[2].abs_diff(ridge)), (nose, 1), "{four:?}");
        assert_eq!(cut(&saratov, 4), (vec![four[0], nose, ridge], vec![layout(0), layout(1), corner(Some(2))]));
        // A nose cut off straight: the leading-edge junction, between the
        // two corners of the cut, gives way to the first; the face is a
        // section of three points.
        let blunt = cut_nose();
        let le = find_leading_edge_index(&blunt).unwrap();
        assert_eq!((blunt[le].x, clear_corners(&blunt)), (-0.0002, vec![le - 1, le + 1]));
        assert_eq!(cut(&blunt, 2), (vec![le - 1, le + 1], vec![corner(Some(0)), corner(None)]));
        // A corner four points behind a round nose (GOE 777 refitted): a
        // junction of its own only if five points are enough for a section.
        // A degree 5 section needs seven, so there the leading-edge
        // junction gives way and the corner is the one junction.
        let xs: Vec<f64> = (0..=60).map(|i| 0.5 * (1.0 + (std::f64::consts::PI * i as f64 / 60.0).cos())).collect();
        let upper = xs.iter().map(|&x| Vec2::new(x, 0.004 * (1.0 - x) + 0.1 * x.sqrt() * (1.0 - x)));
        let face = [0.002, 0.0, -0.002].map(|y| Vec2::new(0.0, y));
        let lower = xs.iter().rev().map(|&x| Vec2::new(x, -0.0039 * (1.0 - x) - 0.1 * x * (1.0 - x)));
        let near: Vec<Vec2> = upper.chain(face).chain(lower).collect();
        let le = find_leading_edge_index(&near).unwrap();
        assert_eq!(clear_corners(&near), [le + 4]);
        let at = |pts: &[Vec2], least| cut_into_sections(pts, Split::at_corners(2, least)).unwrap();
        assert_eq!((at(&near, 5).1, at(&near, 5).2), (vec![le, le + 4], vec![layout(0), corner(None)]));
        assert_eq!((at(&near, 7).1, at(&near, 7).2), (vec![le + 4], vec![corner(Some(0))]));
        // Saratov's ridge leaves nine points to its nose: both stay while
        // nine are enough for a section.
        assert_eq!((at(&saratov, 9).1, at(&saratov, 10).1), (vec![nose, ridge], vec![ridge]));
        // A corner four points behind a sharp nose is a junction of its own.
        let stepped = wedge(0.005, Some(0.02));
        let le = stepped.iter().position(|p| p.x == 0.0).unwrap();
        assert_eq!(cut(&stepped, 2), (vec![le, le + 4], vec![corner(Some(0)), corner(None)]));
        // Any count: every corner is a junction, and every layout junction
        // is one or has a corner within two points.
        for pts in [&sharp, &saratov, &blunt, &stepped, &wedge(0.005, Some(0.75))] {
            let found = clear_corners(pts);
            for sections in 1..=6 {
                let plain = split_into_sections(pts, sections).unwrap().1;
                let (j, seams) = cut(pts, sections);
                assert!(found.iter().all(|c| j.contains(c)), "{sections}: {j:?} {found:?}");
                assert!(plain.iter().all(|p| j.contains(p) || found.iter().any(|c| c.abs_diff(*p) <= CORNER_SIDE)), "{sections}: {j:?} {plain:?}");
                assert!(j.iter().zip(&seams).all(|(i, s)| s.corner == found.contains(i) && (s.corner || s.layout.is_some())), "{seams:?}");
                let kept: Vec<usize> = seams.iter().filter_map(|s| s.layout).collect();
                assert!(kept.windows(2).all(|w| w[0] < w[1]) && kept.iter().all(|&k| k + 1 < sections), "{sections}: {kept:?}");
            }
        }
        // Without corners in the data, or without asking for them: the layout.
        let naca = Naca4::new("2412", true).unwrap().points(81);
        for sections in 1..=5 {
            let plain = split_into_sections(&naca, sections).unwrap();
            let seams: Vec<Seam> = (0..sections - 1).map(layout).collect();
            assert_eq!(cut_into_sections(&naca, Split::at_corners(sections, 7)).unwrap(), (plain.0.clone(), plain.1.clone(), seams.clone()));
            let off = cut_into_sections(&saratov, sections.into()).unwrap();
            assert_eq!((off.1, off.2), (split_into_sections(&saratov, sections).unwrap().1, seams));
        }
    }

    /// An airfoil remembers how it was cut and which sections end at a
    /// corner.
    #[test]
    fn airfoils_keep_their_cut() {
        let a = AirfoilData::new(saratov(), Split::at_corners(2, 7), "saratov", "Saratov", 5).unwrap();
        assert_eq!((a.num_sections(), a.split, &a.junction_indices), (3, Split::at_corners(2, 7), &vec![14, 22]));
        assert_eq!(a.at_corner(), [false, true, true]);
        assert_eq!(a.section_sizes(), "[15, 9, 7]");
        let plain = a.resplit(2).unwrap();
        assert_eq!((plain.num_sections(), plain.split, plain.at_corner()), (2, Split::from(2), vec![false, false]));
    }
}
