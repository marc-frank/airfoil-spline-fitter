//! Airfoil data and its split into sections.

use bspline_varpro::centripetal_parameters;

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

/// Sections joined at the given indices (each junction point in both).
fn split_at(points: &[Vec2], junctions: &[usize]) -> Vec<Vec<Vec2>> {
    let mut bounds = vec![0];
    bounds.extend_from_slice(junctions);
    bounds.push(points.len() - 1);
    bounds.windows(2).map(|w| points[w[0]..=w[1]].to_vec()).collect()
}

/// Splits Selig-ordered coordinates into `count` sections sharing their end
/// points; returns the sections and the junction indices.
///
/// Two sections split at the leading edge; an odd count splits evenly by
/// arc length; an even count splits each side of the leading edge evenly.
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

/// Airfoil coordinates with their split into sections.
#[derive(Debug, Clone, PartialEq)]
pub struct AirfoilData {
    pub coordinates: Vec<Vec2>,
    pub sections: Vec<Vec<Vec2>>,
    pub junction_indices: Vec<usize>,
    /// File name without directory and suffix ("" for pasted or created data).
    pub filename: String,
    pub name: String,
    /// Decimal places of the coordinates.
    pub precision: i32,
}

impl AirfoilData {
    pub fn new(coordinates: Vec<Vec2>, num_sections: usize, filename: &str, name: &str, precision: i32) -> Result<Self> {
        let (sections, junction_indices) = split_into_sections(&coordinates, num_sections)?;
        Ok(AirfoilData { coordinates, sections, junction_indices, filename: filename.into(), name: name.into(), precision })
    }

    pub fn num_sections(&self) -> usize {
        self.sections.len()
    }

    /// The same airfoil split into another number of sections.
    pub fn resplit(&self, num_sections: usize) -> Result<Self> {
        Self::new(self.coordinates.clone(), num_sections, &self.filename, &self.name, self.precision)
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
mod tests {
    use super::*;
    use bspline_varpro::airfoil::Naca4;

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
}
