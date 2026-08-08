//! Exact great-circle radius filtering for engines with **no geo type** but with
//! **arithmetic in their filter language** (issue #223).
//!
//! # The problem
//!
//! `datasets/datasets.json` ships `random-geo-radius-{100,2048}-angular-filters`,
//! whose every query carries `{"<field>": {"geo": {"lon": .., "lat": ..,
//! "radius": ..}}}` — radius in **metres**, great-circle. Qdrant, RediSearch,
//! Elasticsearch, OpenSearch, Weaviate and pgvector each have a native operator
//! for that (`geo_radius`, `@f:[lon lat r m]`, `geo_distance`, `WithinGeoRange`,
//! `earth_distance(ll_to_earth(..))`). VectorSets and Milvus have none.
//!
//! # The encoding
//!
//! A geodesic radius test is a **linear** test in 3-space, so it needs no
//! trigonometry at query time and no square root:
//!
//! Map a point `(lat, lon)` to its unit vector on the sphere
//!
//! ```text
//! u = (cos φ · cos λ,  cos φ · sin λ,  sin φ)        φ = lat, λ = lon (radians)
//! ```
//!
//! For two such unit vectors `u`, `q` the dot product **is** the cosine of the
//! central angle between them, and the great-circle distance is `R · angle`.
//! Therefore, exactly:
//!
//! ```text
//! distance(u, q) <= r   ⟺   u · q >= cos(r / R)
//! ```
//!
//! The right-hand side is a constant the client computes per query; the
//! left-hand side is `x·qx + y·qy + z·qz` over three stored scalars. Both sides
//! are expressible with `*`, `+` and `>=` alone — which is exactly the subset
//! VectorSets' `FILTER` and Milvus' boolean expressions provide.
//!
//! This is **not** an approximation and **not** a bounding box. A bounding box
//! in lat/lon admits the corners (up to `√2 · r` away, and unboundedly more near
//! the poles) and would inflate the candidate set with points the ground truth
//! excludes; an equirectangular "flat earth" form is wrong away from the query
//! latitude. The cap above is the same set the reference engines' native
//! operators select, up to their differing earth radii (see [`EARTH_RADIUS_M`]).
//!
//! # Storage
//!
//! Engines using this encoding must store the three components as **top-level
//! numeric fields** next to the point, named by [`component_field`]. VectorSets'
//! selectors reach only top-level keys and a nested object makes the whole
//! expression evaluate to false, so the `{lon, lat}` object it already stores is
//! unusable as-is; Milvus has no struct field type at all.
//!
//! # Precision
//!
//! `cos(r/R)` is evaluated in `f64`. Near `r = 0` it sits within one ulp of 1.0,
//! so the absolute error in the *implied* radius grows as the radius shrinks.
//!
//! The shipped radii, over the WHOLE `random_geo_1m` `tests.jsonl` (10 000
//! queries, 16 666 geo leaves): **min 1 031 m, max 1 999 939 m, mean 996 075 m**.
//! (An earlier revision of this comment quoted 1 193 / 1 991 786 / 990 613 —
//! those are the min/max/mean of the first **500** queries only, a sample
//! presented as the population.)
//!
//! Error measured by crossover search: binary-search the distance at which the
//! REAL predicate flips — `dot(u_doc, u_centre) >= cos(r/R)` on `f64` unit
//! vectors, including the dot product's own rounding, not the ideal `cos` — over
//! six centres (equator, mid-latitude, 81 N, 89 N, the antimeridian) x five
//! bearings. Worst case **3.6 µm at r = 1 031 m**, 4.1 µm at 1 193 m, 0.30 µm at
//! the mean, 0.01 µm at the max. An independent measurement with different
//! sampling put the small-radius end at ~8 µm; take **under 10 µm** as the
//! bound, since the exact figure depends on which centres you sample. At a
//! hand-written `r = 1 m` it is 5.6 mm.
//!
//! Every one of those is 7-10 orders of magnitude below the margin any fixture
//! here keeps around the boundary, and the dominant error is not this but the
//! choice of earth radius.
//!
//! Server-side the arithmetic is `double` on both engines that use this
//! encoding (VectorSets' `expr.c` parses with `strtod` and evaluates in
//! `double`; Milvus' expression evaluator likewise). That matters: in `f32` the
//! same predicate would be worth ±2.4 km at `r = 1 000 m`.
//!
//! One place the encoding is **not** literally exact: at `r = 0`, a document
//! exactly at the query point gives `dot(u, u) = 0.999999999999999 9` for many
//! `u`, just under the threshold of 1.0, so it is excluded. Unreachable on
//! shipped data (minimum radius 1 031 m) and only reachable from a hand-written
//! `r = 0` config, where "within zero metres" is a degenerate question anyway.

/// Mean earth radius, metres.
///
/// Chosen to match `tests/common/mod.rs::haversine_m`, which brute-forces this
/// repo's geo fixtures' ground truth, and Milvus' `ST_DWITHIN` refine step,
/// which uses the same constant.
///
/// **It is NOT the radius the shipped datasets' ground truth was built on.**
/// Those come from `qdrant/ann-filtering-benchmark-datasets`, whose `check_geo`
/// uses `haversine` with `_AVG_EARTH_RADIUS_KM = 6371.0088` and a strict `<` —
/// i.e. **6 371 008.8 m**. Established empirically rather than from the source:
/// recomputing the filter mask at 6 371 008.8 with `<` reproduces the shipped
/// `closest_ids` for 500/500 sampled queries (12 500/12 500 neighbours), and no
/// other radius/comparator combination tried does.
///
/// Using the slightly smaller 6 371 000 here is deliberate and **safe in the
/// only direction that matters**: a smaller `R` makes `cos(r/R)` smaller, so the
/// cap is a strict *superset* of the ground truth's and no ground-truth
/// neighbour can ever be excluded by the difference. It only admits a few extra
/// documents — measured at 24 extra admitted across 500 shipped queries
/// (0.048/query) against a mean of **14 594.2** matching documents per query.
/// Not literally free: a superset CAN displace a true neighbour out of top-k.
/// The magnitude here is ~1e-4 of recall, so the conclusion holds, but "costs
/// nothing" (an earlier wording) was wrong.
///
/// Independently reproduced on the real `random_geo_1m`, counting ground-truth
/// neighbours a strict `<` cap would EXCLUDE: R = 6 371 000 excludes **0** of
/// 11 119; R = 6 372 797.56 (RediSearch) excludes 8; R = 6 378 137 (WGS-84)
/// excludes 27.
///
/// The reference engines disagree here anyway, by up to ~0.1 % (RediSearch's
/// geohash library uses 6 372 797.56 m — 0.028 % stricter than the ground
/// truth's; PostgreSQL's `earthdistance` uses 6 378 168 m; Elasticsearch and
/// MongoDB Search use the WGS-84 mean 6 371 008.8 m). That spread is worth
/// ~1e-4 of recall and predates this module.
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Unit vector on the sphere for a `(lat, lon)` pair in **degrees**.
///
/// Returned in the `[x, y, z]` order that [`component_field`] names.
pub fn unit_vector(lat_deg: f64, lon_deg: f64) -> [f64; 3] {
    let (phi, lambda) = (lat_deg.to_radians(), lon_deg.to_radians());
    [
        phi.cos() * lambda.cos(),
        phi.cos() * lambda.sin(),
        phi.sin(),
    ]
}

/// `cos(radius / R)` — the threshold the dot product must reach, for a
/// **non-negative** radius.
///
/// Clamped below at `-1`: a radius over half the earth's circumference covers
/// the whole sphere, and past that `cos` turns back up, which would wrongly
/// start *excluding* the antipode. Clamping keeps the predicate monotone in the
/// radius for every input. A negative radius is rejected upstream by
/// [`query_terms`] rather than encoded, so that no engine has to render an
/// out-of-range literal.
pub fn cos_central_angle(radius_m: f64) -> f64 {
    let theta = (radius_m / EARTH_RADIUS_M).max(0.0);
    if theta >= std::f64::consts::PI {
        return -1.0;
    }
    theta.cos()
}

/// The three component axis names, in the order [`unit_vector`] returns them.
pub const COMPONENTS: [&str; 3] = ["x", "y", "z"];

/// Name of the stored scalar holding component `axis` of `field`'s unit vector.
///
/// The `__geo_` infix is deliberately unlikely to collide with a real payload
/// key: a dataset field named `a` yields `a__geo_x`. Both the storage side and
/// the filter side call this, so they cannot drift.
pub fn component_field(field: &str, axis: &str) -> String {
    format!("{field}__geo_{axis}")
}

/// The query-side numbers for a `{"lon":.., "lat":.., "radius":..}` criteria
/// object: the query point's unit vector and the cosine threshold.
///
/// `None` when any of the three keys is missing, not a finite number, or the
/// radius is negative. That is
/// a **drop**, which `query_filter::resolve` turns into a hard error — the same
/// choice `redis::build_geo_filter` makes for a missing radius. Defaulting the
/// radius (Qdrant/Elasticsearch/Weaviate substitute 1000 m) would silently
/// invent a filter nobody asked for.
pub fn query_terms(criteria: &serde_json::Value) -> Option<([f64; 3], f64)> {
    let num = |k: &str| {
        criteria
            .get(k)
            .and_then(|v| v.as_f64())
            .filter(|v| v.is_finite())
    };
    let (lat, lon, radius) = (num("lat")?, num("lon")?, num("radius")?);
    if radius < 0.0 {
        return None;
    }
    Some((unit_vector(lat, lon), cos_central_angle(radius)))
}

/// Render an `f64` as a **plain decimal** that parses back to the identical
/// value — never scientific notation.
///
/// `{:?}` is Rust's shortest round-tripping form and was the obvious choice, but
/// it switches to exponent form below `1e-4`, and that does not survive the
/// engines' expression parsers. VectorSets' `expr.c::exprParseNumber` accepts
/// digits, `.`, `e`/`E` and a **leading** `-`, but not the `-` that follows an
/// exponent marker, so a coefficient of `1.9426574824796435e-5` produces
///
/// ```text
/// ERR syntax error in FILTER expression near: -5 + .loc__geo_z * 0.0) >= 0.9
/// ```
///
/// and the whole query fails. That is not hypothetical: **25 of the 10 000
/// queries** in the shipped `random_geo_1m` have a query-centre unit-vector
/// component under `1e-4` in magnitude — a centre within ~0.006° of the equator
/// (`z` tiny) or within ~0.006° of a meridian (`y` tiny). Two real rows:
/// `{"lon": -52.54…, "lat": -0.00111…}` → `z = -1.94e-5`, and
/// `{"lon": -10.42…, "lat": 89.97…}` → `y = -9.39e-5`.
///
/// The search is for the SHORTEST fixed-point form that round-trips, from one
/// decimal place upward, so ordinary values stay short (`0.5` stays `0.5`,
/// `326341.0` stays `326341.0`) and only the tiny ones pay for the extra
/// digits. 340 places covers the smallest subnormal `f64`, so the fallback is
/// unreachable for any finite input.
pub fn plain_decimal(v: f64) -> String {
    if v == 0.0 {
        // Covers -0.0 too, whose sign carries no information here.
        return "0.0".to_string();
    }
    for p in 1..=340 {
        let s = format!("{v:.p$}");
        if s.parse::<f64>() == Ok(v) {
            return s;
        }
    }
    format!("{v:.340}")
}

/// Render the cap predicate as `(<f__geo_x> * qx + <f__geo_y> * qy +
/// <f__geo_z> * qz) >= cos_theta`, with `sel` naming how the engine spells a
/// field reference (`.a__geo_x` for VectorSets, plain `a__geo_x` for Milvus).
///
/// Literals go out through [`plain_decimal`], which round-trips exactly without
/// ever emitting exponent form — see that function for why `{:?}` cannot be used
/// here.
pub fn cap_expression(
    field: &str,
    criteria: &serde_json::Value,
    sel: impl Fn(&str) -> String,
) -> Option<String> {
    let (q, cos_theta) = query_terms(criteria)?;
    let terms: Vec<String> = COMPONENTS
        .iter()
        .zip(q.iter())
        .map(|(axis, coeff)| {
            format!(
                "{} * {}",
                sel(&component_field(field, axis)),
                plain_decimal(*coeff)
            )
        })
        .collect();
    Some(format!(
        "({}) >= {}",
        terms.join(" + "),
        plain_decimal(cos_theta)
    ))
}

/// Whether a condition tree mentions a `geo` operator anywhere.
///
/// For an engine that cannot express geo AT ALL, refusing the leaf is not
/// enough. Every builder in this tree keeps the leaves it understands and skips
/// the rest, so `and(geo, keyword)` would otherwise emit just the keyword clause
/// — a filter that is real, passes the #219 guard, and constrains LESS than the
/// ground truth does. That is the partial-drop escape `query_filter.rs`
/// documents, and it is silent. An engine with no geo capability calls this at
/// the top of its builder and refuses the whole tree instead.
pub fn conditions_mention_geo(conditions: &serde_json::Value) -> bool {
    match conditions {
        // The leaf shape is `{"<field>": {"geo": {...}}}`, so what identifies a
        // geo leaf is the OPERATOR key one level down — never the field name. A
        // field literally called `geo` carrying an ordinary condition
        // (`{"geo": {"match": {...}}}`) is not a geo leaf, and keying on the
        // name would refuse it.
        serde_json::Value::Object(obj) => obj.iter().any(|(_, v)| {
            v.as_object().is_some_and(|spec| spec.contains_key("geo")) || conditions_mention_geo(v)
        }),
        serde_json::Value::Array(items) => items.iter().any(conditions_mention_geo),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Great-circle distance in metres on [`EARTH_RADIUS_M`], the same formula
    /// the fixtures' ground truth uses.
    fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
        let dphi = (lat2 - lat1).to_radians();
        let dlambda = (lon2 - lon1).to_radians();
        let a = (dphi / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dlambda / 2.0).sin().powi(2);
        2.0 * EARTH_RADIUS_M * a.sqrt().asin()
    }

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn unit_vectors_are_unit() {
        for (lat, lon) in [(0.0, 0.0), (40.0, -74.0), (-52.0, 116.0), (90.0, 12.0)] {
            let u = unit_vector(lat, lon);
            assert!((dot(u, u) - 1.0).abs() < 1e-12, "({lat},{lon}) -> {u:?}");
        }
    }

    /// The whole claim, tested against haversine rather than against itself: for
    /// a grid of point pairs and a ladder of radii, the dot-product predicate
    /// and the metre-distance predicate must agree. A bounding box, an
    /// equirectangular approximation, or a wrong earth radius all break this.
    #[test]
    fn the_cap_predicate_agrees_with_haversine_everywhere() {
        let pts = [
            (0.0, 0.0),
            (40.0, -74.0),
            (-52.0, 116.0),
            (51.5, -0.12),
            (-33.9, 151.2),
            (89.0, 179.0),
            (-89.0, -179.0),
            (0.0, 180.0),
        ];
        let mut agreed = 0usize;
        for (qlat, qlon) in pts {
            let q = unit_vector(qlat, qlon);
            for (lat, lon) in pts {
                let u = unit_vector(lat, lon);
                let d = haversine_m(lat, lon, qlat, qlon);
                for r in [0.0, 1.0, 1_000.0, 326_341.0, 5_000_000.0, 15_000_000.0] {
                    // Skip pairs within a millimetre of the boundary: both sides
                    // are exact there and the comparison is a coin flip.
                    if (d - r).abs() < 1e-3 {
                        continue;
                    }
                    assert_eq!(
                        dot(u, q) >= cos_central_angle(r),
                        d <= r,
                        "({lat},{lon}) vs ({qlat},{qlon}) at r={r}: haversine {d}"
                    );
                    agreed += 1;
                }
            }
        }
        assert!(agreed > 300, "expected a dense grid, compared {agreed}");
    }

    /// A lat/lon bounding box of half-width `r` admits the corners. The
    /// dot-product form must reject exactly those — this is the discrimination
    /// the whole encoding exists for.
    #[test]
    fn a_corner_of_the_bounding_box_is_outside_the_cap() {
        let (qlat, qlon) = (40.0, -74.0);
        let q = unit_vector(qlat, qlon);
        let r = 20_000.0;
        // ~1.35 r away on the NE diagonal: inside the box (each axis < r), well
        // outside the circle.
        let (lat, lon) = (qlat + 0.171, qlon + 0.223);
        let d = haversine_m(lat, lon, qlat, qlon);
        assert!(d > r && d < r * std::f64::consts::SQRT_2, "d={d}");
        assert!(dot(unit_vector(lat, lon), q) < cos_central_angle(r));
    }

    #[test]
    fn a_radius_past_the_antipode_matches_the_whole_sphere() {
        let q = unit_vector(10.0, 20.0);
        let anti = unit_vector(-10.0, -160.0);
        assert!((dot(q, anti) + 1.0).abs() < 1e-12, "antipodal");
        assert!(dot(q, anti) >= cos_central_angle(std::f64::consts::PI * EARTH_RADIUS_M));
        // and past it, where a naive cos() would turn back up and exclude it
        assert!(dot(q, anti) >= cos_central_angle(4.0 * EARTH_RADIUS_M));
    }

    /// A negative radius is a DROP, not an encoded predicate: rendering
    /// `cos()` of it would emit a literal outside `[-1, 1]` (or, clamped, a
    /// match-everything filter).
    #[test]
    fn a_negative_radius_is_refused_before_it_is_encoded() {
        let bad = json!({"lon": 10.0, "lat": 20.0, "radius": -1.0});
        assert!(query_terms(&bad).is_none());
        assert!(cap_expression("a", &bad, |f| format!(".{f}")).is_none());
    }

    #[test]
    fn geo_is_detected_anywhere_in_a_condition_tree() {
        let leaf = json!({"a": {"geo": {"lon": 1.0, "lat": 2.0, "radius": 3.0}}});
        assert!(conditions_mention_geo(&leaf));
        assert!(conditions_mention_geo(&json!({"and": [leaf.clone()]})));
        // Mixed with a sibling the engine COULD express — the case a per-leaf
        // refusal would let through under-constrained.
        assert!(conditions_mention_geo(&json!({"and": [
            leaf.clone(), {"c": {"match": {"value": "red"}}}
        ]})));
        // Nested two levels deep.
        assert!(conditions_mention_geo(&json!({"or": [
            {"and": [leaf]}, {"and": [{"c": {"match": {"value": "red"}}}]}
        ]})));
        // And no false positives on the shapes that carry no geo, including a
        // FIELD literally named `geo` carrying an ordinary condition.
        for clean in [
            json!({"and": [{"c": {"match": {"value": "red"}}}]}),
            json!({"and": [{"n": {"range": {"gte": 1, "lt": 9}}}]}),
            json!({"and": [{"geo": {"match": {"value": "red"}}}]}),
            json!({}),
            serde_json::Value::Null,
        ] {
            assert!(!conditions_mention_geo(&clean), "{clean}");
        }
    }

    #[test]
    fn component_fields_are_stable_and_distinct() {
        assert_eq!(component_field("a", "x"), "a__geo_x");
        let names: Vec<String> = COMPONENTS.iter().map(|c| component_field("a", c)).collect();
        assert_eq!(names, ["a__geo_x", "a__geo_y", "a__geo_z"]);
    }

    #[test]
    fn an_incomplete_criteria_object_is_a_drop_not_a_default() {
        for bad in [
            json!({"lon": 10.0, "lat": 20.0}),     // no radius
            json!({"lon": 10.0, "radius": 500.0}), // no lat
            json!({"lat": 20.0, "radius": 500.0}), // no lon
            json!({"lon": 10.0, "lat": 20.0, "radius": "500"}),
            json!({}),
        ] {
            assert!(query_terms(&bad).is_none(), "{bad}");
            assert!(
                cap_expression("a", &bad, |f| format!(".{f}")).is_none(),
                "{bad}"
            );
        }
    }

    #[test]
    fn the_rendered_expression_names_all_three_components_and_the_threshold() {
        let e = cap_expression(
            "a",
            &json!({"lon": 116.0, "lat": -52.0, "radius": 326_341.0}),
            |f| format!(".{f}"),
        )
        .unwrap();
        for c in ["a__geo_x", "a__geo_y", "a__geo_z"] {
            assert!(e.contains(&format!(".{c} * ")), "{e}");
        }
        assert!(e.contains(">= 0.99868"), "{e}");
    }

    /// The rendered literals must round-trip: a truncated coefficient is a
    /// silently displaced query centre.
    #[test]
    fn rendered_literals_round_trip_exactly() {
        for criteria in [
            json!({"lon": 116.0, "lat": -52.0, "radius": 326_341.0}),
            // A centre 0.001° off the equator: `z` is ~1.9e-5.
            json!({"lon": -52.545, "lat": -0.0011, "radius": 1_193.0}),
            // A centre 0.03° off the pole: `y` is ~-9.4e-5.
            json!({"lon": -10.42, "lat": 89.97027, "radius": 1_991_786.0}),
            // Exactly on a meridian and exactly on the equator: `y` / `z` are 0.
            json!({"lon": 0.0, "lat": 0.0, "radius": 50_000.0}),
        ] {
            let (q, cos_theta) = query_terms(&criteria).unwrap();
            let e = cap_expression("a", &criteria, |f| f.to_string()).unwrap();
            // Every literal in the rendering must parse back to the exact f64.
            for v in q.iter().chain(std::iter::once(&cos_theta)) {
                assert!(e.contains(&plain_decimal(*v)), "{v:?} not verbatim in {e}");
                assert_eq!(plain_decimal(*v).parse::<f64>(), Ok(*v), "{v:?}");
            }
        }
    }

    /// The blocker `plain_decimal` exists for: `{:?}` switches to exponent form
    /// below 1e-4, and VectorSets' `exprParseNumber` rejects the `-` after the
    /// `e`, failing the WHOLE query. 25 of the 10 000 shipped `random_geo_1m`
    /// queries have a centre component in that range.
    #[test]
    fn no_rendered_literal_ever_uses_exponent_notation() {
        // The two real shipped centres, plus a sweep down to 1e-300 and the
        // smallest NORMAL f64. It does not reach the smallest subnormal
        // (~4.94e-324); that 340 places would cover it too is an argument (324
        // decimal places suffice) rather than something tested here.
        let mut values: Vec<f64> = vec![
            -1.9426574824796435e-5,
            -9.390384479741515e-5,
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            f64::MIN_POSITIVE,
        ];
        let mut v = 1.0f64;
        for _ in 0..300 {
            v /= 10.0;
            values.push(v);
            values.push(-v);
        }
        for v in values {
            let s = plain_decimal(v);
            assert!(
                !s.contains('e') && !s.contains('E'),
                "{v:?} rendered as {s}"
            );
            assert_eq!(s.parse::<f64>(), Ok(v), "{v:?} rendered as {s}");
        }
    }

    /// Ordinary values must not get longer just because tiny ones can.
    #[test]
    fn plain_decimal_stays_short_for_ordinary_values() {
        assert_eq!(plain_decimal(0.5), "0.5");
        assert_eq!(plain_decimal(-1.0), "-1.0");
        assert_eq!(plain_decimal(0.0), "0.0");
        assert_eq!(plain_decimal(326_341.0), "326341.0");
    }
}
