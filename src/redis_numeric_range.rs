//! Redis NUMERIC query encoding for one canonical range leaf.
//!
//! Parameter conversion belongs to the adapter. Keeping both sides of a leaf in
//! one interval avoids intersecting wide open ranges and requires the same
//! numeric value to satisfy both bounds when a field contains multiple values.

/// Recorded with effective engine settings so diagnostics can reproduce queries.
pub const ENCODING: &str = "bounded_leaf_v1";

/// Recognize a successfully converted range bound in the Redis adapter's PARAMS
/// map. Its names end in `<field>_<counter>_<operator>`; match parameters end in
/// the counter, while geo parameters end in `lon`, `lat`, or `radius`.
pub fn is_bound_parameter(name: &str) -> bool {
    ["_lt", "_gt", "_lte", "_gte"]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

#[derive(Clone, Copy, Debug)]
pub enum Operator {
    Lt,
    Gt,
    Lte,
    Gte,
}

/// An already converted bound; `parameter` is its PARAMS name without `$`.
#[derive(Clone, Copy, Debug)]
pub struct Bound<'a> {
    pub parameter: &'a str,
    pub operator: Operator,
}

/// Render one range leaf, never bounds collected from separate leaves.
///
/// A normal two-sided leaf becomes one bounded interval. If a leaf supplies
/// multiple bounds on either side, retain every lower/upper pairing. One pairing
/// is necessarily the strongest interval, so the conjunction is equivalent to
/// requiring one numeric value to satisfy all bounds, including for multivalued
/// fields. This avoids comparing or rounding parameter values in the renderer.
/// An empty leaf has no expression; a missing side is unbounded.
pub fn render(field: &str, bounds: &[Bound<'_>]) -> Option<String> {
    if bounds.is_empty() {
        return None;
    }
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for bound in bounds {
        match bound.operator {
            Operator::Gt => lower.push(format!("(${}", bound.parameter)),
            Operator::Gte => lower.push(format!("${}", bound.parameter)),
            Operator::Lt => upper.push(format!("(${}", bound.parameter)),
            Operator::Lte => upper.push(format!("${}", bound.parameter)),
        }
    }
    if lower.is_empty() {
        lower.push("-inf".to_string());
    }
    if upper.is_empty() {
        upper.push("+inf".to_string());
    }
    let mut clauses = Vec::new();
    for lo in &lower {
        for hi in &upper {
            clauses.push(format!("@{field}:[{lo} {hi}]"));
        }
    }
    Some(clauses.join(" "))
}

#[cfg(test)]
mod tests {
    use super::{is_bound_parameter, render, Bound, Operator};

    #[test]
    fn bound_parameter_suffixes_distinguish_match_and_geo_parameters() {
        for name in ["n_0_lt", "ts_10_gte", "name_lt_9_gt", "n_8_lte"] {
            assert!(is_bound_parameter(name), "{name}");
        }
        for name in ["n_lt_0", "n_gte_1", "n_9", "loc_gt_1_lon", "loc_0_radius"] {
            assert!(!is_bound_parameter(name), "{name}");
        }
    }

    fn bound(parameter: &str, operator: Operator) -> Bound<'_> {
        Bound {
            parameter,
            operator,
        }
    }

    #[test]
    fn two_sided_ranges_are_one_interval_with_all_inclusivity_combinations() {
        for (lo, hi, expected) in [
            (Operator::Gt, Operator::Lt, "@n:[($lo ($hi]"),
            (Operator::Gt, Operator::Lte, "@n:[($lo $hi]"),
            (Operator::Gte, Operator::Lt, "@n:[$lo ($hi]"),
            (Operator::Gte, Operator::Lte, "@n:[$lo $hi]"),
        ] {
            // Adapter discovery order can put the upper bound first.
            assert_eq!(
                render("n", &[bound("hi", hi), bound("lo", lo)]).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn one_sided_and_empty_ranges() {
        assert_eq!(render("n", &[]), None);
        for (op, expected) in [
            (Operator::Gt, "@n:[($p +inf]"),
            (Operator::Gte, "@n:[$p +inf]"),
            (Operator::Lt, "@n:[-inf ($p]"),
            (Operator::Lte, "@n:[-inf $p]"),
        ] {
            assert_eq!(render("n", &[bound("p", op)]).unwrap(), expected);
        }
    }

    #[test]
    fn retains_every_same_side_constraint_without_comparing_parameters() {
        assert_eq!(
            render(
                "n",
                &[
                    bound("a", Operator::Lt),
                    bound("b", Operator::Gt),
                    bound("c", Operator::Lte),
                    bound("d", Operator::Gte),
                ]
            )
            .unwrap(),
            "@n:[($b ($a] @n:[($b $c] @n:[$d ($a] @n:[$d $c]"
        );
        assert_eq!(
            render("n", &[bound("a", Operator::Gt), bound("b", Operator::Gte)]).unwrap(),
            "@n:[($a +inf] @n:[$b +inf]"
        );
    }

    // Interpret the emitted Redis interval syntax independently of the renderer.
    // Each interval can match any one value, but its two endpoints must match
    // that same value. Separate interval clauses are then intersected by Redis.
    fn query_matches(query: &str, values: &[f64], params: &[(&str, f64)]) -> bool {
        query.split(']').filter(|s| !s.is_empty()).all(|clause| {
            let endpoints: Vec<_> = clause
                .trim()
                .strip_prefix("@n:[")
                .unwrap()
                .split_whitespace()
                .collect();
            assert_eq!(endpoints.len(), 2);
            values.iter().any(|&value| {
                endpoints.iter().enumerate().all(|(side, endpoint)| {
                    if *endpoint == "-inf" || *endpoint == "+inf" {
                        return true;
                    }
                    let exclusive = endpoint.starts_with('(');
                    let name = endpoint.trim_start_matches('(').strip_prefix('$').unwrap();
                    let target = params.iter().find(|(key, _)| *key == name).unwrap().1;
                    match (side, exclusive) {
                        (0, true) => value > target,
                        (0, false) => value >= target,
                        (_, true) => value < target,
                        (_, false) => value <= target,
                    }
                })
            })
        })
    }

    #[test]
    fn multi_value_range_requires_one_value_matching_every_bound() {
        let values = [vec![], vec![-10.0, 10.0], vec![0.0], vec![-1.0, 1.0]];
        // Exhaustively include equal/reversed bounds and ties with differing
        // inclusivity; the oracle directly applies canonical leaf predicates.
        for gt in [-1.0, 0.0, 1.0] {
            for gte in [-1.0, 0.0, 1.0] {
                for lt in [-1.0, 0.0, 1.0] {
                    for lte in [-1.0, 0.0, 1.0] {
                        let params = [("gt", gt), ("gte", gte), ("lt", lt), ("lte", lte)];
                        for mask in 1..16 {
                            let bounds: Vec<_> = [
                                bound("gt", Operator::Gt),
                                bound("gte", Operator::Gte),
                                bound("lt", Operator::Lt),
                                bound("lte", Operator::Lte),
                            ]
                            .into_iter()
                            .enumerate()
                            .filter_map(|(i, b)| (mask & (1 << i) != 0).then_some(b))
                            .collect();
                            let query = render("n", &bounds).unwrap();
                            for document in &values {
                                let expected = document.iter().any(|&v| {
                                    (mask & 1 == 0 || v > gt)
                                        && (mask & 2 == 0 || v >= gte)
                                        && (mask & 4 == 0 || v < lt)
                                        && (mask & 8 == 0 || v <= lte)
                                });
                                assert_eq!(
                                    query_matches(&query, document, &params),
                                    expected,
                                    "{query}, params={params:?}, document={document:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
