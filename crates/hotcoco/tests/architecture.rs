//! Architecture conformance — the crate's "one obvious home" rule, as CI.
//!
//! hotcoco's product is auditability: someone checking our numbers should find
//! **one** place answering "how is similarity computed?", "how are detections
//! matched?", "how is AP accumulated?". Before 0.5 there were four bbox-IoU
//! implementations, three copies of the crowd/union formula, two greedy matchers,
//! and two disagreeing `MIN_PARALLEL_WORK` constants (1024 vs 1000, under a
//! comment claiming they matched).
//!
//! Those are consolidated into `primitives/` and `metrics/`. These tests exist so
//! the duplication cannot silently creep back. They are deliberately crude — grep
//! over source text, no parsing. A determined author can evade them; the point is
//! to catch the *accidental* re-introduction that code review misses, and to make
//! the rule explicit and enforced rather than folklore in a planning doc.
//!
//! The layering checks below are **allowlists**, not banlists, and that distinction
//! is load-bearing. Their first version banned the spelling `crate::detection` —
//! which the same release made meaningless by shipping `hotcoco::eval` and ~25
//! crate-root re-exports of the very same types. A banlist must enumerate every
//! path to a thing; an allowlist enumerates what a layer is for.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// The workspace root, derived from this crate's location.
fn workspace_root() -> &'static Path {
    static ROOT: LazyLock<PathBuf> = LazyLock::new(|| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("crate lives at <workspace>/crates/<name>")
            .to_path_buf()
    });
    &ROOT
}

/// Every `.rs` file under `crates/*/src/` (the shipped source; tests excluded),
/// paired with its workspace-relative path and its code lines.
///
/// Read once and shared across the checks below.
fn sources() -> &'static [SourceFile] {
    static SOURCES: LazyLock<Vec<SourceFile>> = LazyLock::new(|| {
        let mut paths = Vec::new();
        let crates = fs::read_dir(workspace_root().join("crates")).expect("crates/ dir");
        for entry in crates.flatten() {
            collect_rs(&entry.path().join("src"), &mut paths);
        }
        assert!(!paths.is_empty(), "found no source files to scan");
        paths.sort();
        paths.iter().map(|p| SourceFile::read(p)).collect()
    });
    &SOURCES
}

struct SourceFile {
    /// Workspace-relative path with forward slashes, e.g. `crates/hotcoco/src/mask.rs`.
    path: String,
    /// 1-based line numbers paired with their text, with whole-line `//` comments
    /// dropped — prose mentioning a banned name is not a violation.
    lines: Vec<(usize, String)>,
}

impl SourceFile {
    fn read(path: &Path) -> Self {
        // `expect`, not a silent default: a conformance test that cannot read a
        // file must fail loudly, never pass by scanning nothing.
        let text = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let rel = path
            .strip_prefix(workspace_root())
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        SourceFile {
            path: rel,
            lines: text
                .lines()
                .enumerate()
                .map(|(i, l)| (i + 1, l.to_string()))
                .filter(|(_, l)| !l.trim_start().starts_with("//"))
                .collect(),
        }
    }
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Every code line matching `hit`, in files not excluded by `skip`, formatted as
/// `path:line  text`. The shared body of the grep-style checks below.
fn scan(skip: impl Fn(&str) -> bool, hit: impl Fn(&str) -> bool) -> Vec<String> {
    sources()
        .iter()
        .filter(|f| !skip(&f.path))
        .flat_map(|f| {
            f.lines
                .iter()
                .filter(|(_, l)| hit(l))
                .map(move |(n, l)| format!("{}:{n}  {}", f.path, l.trim()))
        })
        .collect()
}

/// The IoU formula itself lives in `primitives::sim` — nowhere else.
///
/// Matching on the *arithmetic* rather than on function names is deliberate:
/// plenty of legitimate code is named `…iou…` (config accessors, PyO3 bindings,
/// the marshaling helpers that reshape annotations before calling a kernel).
/// What must not recur is a second site computing `union = a + b - intersection`
/// — that was copy-pasted three times (mask, bbox, OBB) before 0.5.
///
/// # Adding a sanctioned home
///
/// [`FORMULA_HOMES`] is a list, not a constant, because a *genuinely different*
/// formula may legitimately need its own home. The known case is panoptic
/// quality (1.1): panopticapi computes
/// `union = pred_area + gt_area - intersection - void_intersection`, whose
/// void-overlap term has no analogue here, over a label-map co-occurrence
/// histogram rather than a dense `[D][G]` kernel. When that lands, add
/// `primitives/pq.rs` (or wherever it lives) to the list with a comment — do not
/// weaken or `#[ignore]` this test.
#[test]
fn iou_formula_only_in_primitives_sim() {
    /// Files permitted to define a similarity denominator. One entry per
    /// *distinct* formula — never a place that should have called `sim`.
    const FORMULA_HOMES: &[&str] = &["crates/hotcoco/src/primitives/sim.rs"];

    // `union = <area> + <area> - <intersection>`: the IoU denominator. RLE
    // set-union (`merge(.., false)`) has no subtraction and is not caught.
    let violations = scan(
        |path| FORMULA_HOMES.contains(&path),
        |line| line.contains("union") && line.contains('+') && line.contains(" - "),
    );

    assert!(
        violations.is_empty(),
        "a similarity denominator may only be defined in: {}.\n\
         Found what looks like another copy at:\n  {}\n\n\
         Call `primitives::sim` instead — it exposes matrix kernels (`bbox_iou`, \
         `mask_iou`, `obb_iou`) and the scalar `bbox_iou_pair`. If this really is \
         a different formula (see the panoptic note on this test), add its file to \
         FORMULA_HOMES with a comment.",
        FORMULA_HOMES.join(", "),
        violations.join("\n  ")
    );
}

/// Whether a line contains the `x / (a + b - x)` shape — an IoU denominator
/// with the intersection subtracted back out, spotted structurally.
///
/// Whitespace is stripped first, then every `ident / ( … )` division is
/// checked: the parenthesized body must be exactly `term + term - numerator`,
/// with the subtracted term equal to the identifier being divided. Names do
/// not matter, which is the point — the sibling `union` scan requires the
/// conventional variable name, and the historic copy-paste sometimes inlined
/// the denominator without ever naming it.
fn iou_shape(line: &str) -> bool {
    // Identifier characters, dots included, so `self.inter / (…)` matches too.
    fn is_ident(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_' || c == '.'
    }

    let flat: Vec<char> = line.chars().filter(|c| !c.is_whitespace()).collect();
    let n = flat.len();
    for i in 0..n {
        if flat[i] != '/' || i + 1 >= n || flat[i + 1] != '(' {
            continue;
        }
        // The divided identifier, read backwards from the `/`.
        let num_start = (0..i).rev().take_while(|&j| is_ident(flat[j])).last();
        let Some(num_start) = num_start else { continue };
        let numerator: String = flat[num_start..i].iter().collect();

        // The paren-balanced denominator body.
        let mut depth = 0usize;
        let mut close = None;
        for (j, &c) in flat.iter().enumerate().skip(i + 1) {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(j);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close else { continue };
        let body: String = flat[i + 2..close].iter().collect();

        // Exactly `a + b - numerator`, with simple (operator-free) a and b.
        if let Some((sum, subtracted)) = body.rsplit_once('-') {
            if let Some((a, b)) = sum.split_once('+') {
                if subtracted == numerator
                    && !a.is_empty()
                    && !b.is_empty()
                    && a.chars().all(is_ident)
                    && b.chars().all(is_ident)
                {
                    return true;
                }
            }
        }
    }
    false
}

/// Structural companion to [`iou_formula_only_in_primitives_sim`]: the same
/// rule, matched on shape rather than on the identifier `union`.
///
/// The identifier scan misses `inter / (a + b - inter)` written without a
/// named union — which is how a hurried copy inlines it. This scan misses the
/// two-line spelling (`let u = a + b - i; i / u`), but naming that variable
/// anything like `union` walks into the sibling check; evading *both* takes a
/// deliberately obfuscated name, which is no longer the accidental
/// reintroduction this file guards against.
#[test]
fn iou_formula_shape_only_in_primitives_sim() {
    /// Keep in lockstep with [`iou_formula_only_in_primitives_sim`]'s list —
    /// the two checks cover one rule.
    const FORMULA_HOMES: &[&str] = &["crates/hotcoco/src/primitives/sim.rs"];

    let violations = scan(|path| FORMULA_HOMES.contains(&path), iou_shape);

    assert!(
        violations.is_empty(),
        "a similarity denominator may only be defined in: {}.\n\
         Found the `x / (a + b - x)` shape at:\n  {}\n\n\
         Call `primitives::sim` instead — it exposes matrix kernels (`bbox_iou`, \
         `mask_iou`, `obb_iou`) and the scalar `bbox_iou_pair`.",
        FORMULA_HOMES.join(", "),
        violations.join("\n  ")
    );
}

/// One parallelism threshold for every kernel.
///
/// Two of these drifted apart (1024 vs 1000) while a comment insisted they
/// matched — the exact silent divergence this test exists to prevent.
#[test]
fn exactly_one_min_parallel_work_constant() {
    let found = scan(|_| false, |line| line.contains("const MIN_PARALLEL_WORK"));

    assert_eq!(
        found.len(),
        1,
        "expected exactly one MIN_PARALLEL_WORK declaration, found {}:\n  {}\n\n\
         Per-kernel copies drift. Kernels share the one in `primitives::sim`.",
        found.len(),
        found.join("\n  ")
    );
}

/// The detection cap is derived in `Params::max_det()` — nowhere else.
///
/// Five sites derived it independently: four spelled `max_dets.last()` and one
/// spelled `.iter().max()`. Identical on the sorted default `[1, 10, 100]`,
/// divergent on unsorted input — `evaluate()` stamped eval_imgs with one value
/// while `image_diagnostics` filtered on the other, returning nothing. The
/// `MIN_PARALLEL_WORK` failure mode, again. Indexing `max_dets[m_idx]` to walk
/// the M axis is fine and not caught; *reducing* the list to a single cap is
/// what must go through the owner.
#[test]
fn detection_cap_only_derived_in_params() {
    /// The one sanctioned reduction of `max_dets` to a cap.
    const CAP_OWNER: &[&str] = &["crates/hotcoco/src/params.rs"];

    let violations = scan(
        |path| CAP_OWNER.contains(&path),
        |line| {
            line.contains("max_dets.last(")
                || (line.contains("max_dets") && line.contains(".max()"))
        },
    );

    assert!(
        violations.is_empty(),
        "the per-image detection cap may only be derived in: {}.\n\
         Found another derivation at:\n  {}\n\n\
         Call `params.max_det()` instead — positional and max-based spellings \
         agree on sorted input and silently diverge on unsorted input.",
        CAP_OWNER.join(", "),
        violations.join("\n  ")
    );
}

/// Matching lives in `primitives/` — no hand-rolled greedy loops elsewhere.
///
/// `best_iou`/`best_gi` is the signature of the pycocotools greedy scan; a second
/// copy is how match semantics (tie-breaking especially, which is public contract
/// via `evalImgs`) drift apart.
///
/// This is a *naming* heuristic, so it can catch an innocent bystander — panoptic
/// (1.1) has a unique-match scan where `best_iou` would be a natural variable
/// name, even though PQ needs no solver and is not re-implementing this matcher.
/// If that happens, rename the variable (`pq_overlap`, say) rather than deleting
/// the check: the tokens are cheap to avoid, and their absence is what makes the
/// guard trustworthy.
#[test]
fn greedy_matching_only_in_primitives() {
    let violations = scan(
        |path| path.contains("/primitives/"),
        |line| line.contains("best_iou") || line.contains("best_gi"),
    );

    assert!(
        violations.is_empty(),
        "greedy matching belongs to `primitives::greedy`.\n\
         Hand-rolled matcher found at:\n  {}",
        violations.join("\n  ")
    );
}

/// The whole-dataset similarity cache stays inside the detection driver.
///
/// `COCOeval::ious` maps every evaluated `(image, category)` cell to its
/// similarity matrix. That is fine for detection — but the 0.5 primitives
/// contract review identified it as the single most likely route by which
/// *retention* leaks into a shared contract, and retention is the one thing the
/// primitives layer forbids: `primitives/mod.rs` states that no contract there may
/// require whole-sequence similarity retention, so HOTA's second pass (1.2) stays
/// free to recompute rather than hold hundreds of megabytes per sequence per
/// thread at MOT20 scale.
///
/// The analysis layer only ever needs one cell, so it goes through
/// `COCOeval::cell_ious(img_id, cat_id)`. The accessor's *shape* is the
/// enforcement — it cannot hand out the map — and this test keeps callers from
/// bypassing it by touching the field directly.
///
/// # Adding a sanctioned home
///
/// [`CACHE_OWNERS`] is the driver, and only the driver: the module that owns the
/// field, the one that populates it, and the one that consumes it per cell.
/// Do not add an analysis or `metrics`/`primitives` file to this list; needing to
/// is the signal that the code should call `cell_ious` instead.
#[test]
fn similarity_cache_stays_driver_private() {
    /// Files permitted to touch the `ious` field directly.
    const CACHE_OWNERS: &[&str] = &[
        // owns the field and the `cell_ious` accessor
        "crates/hotcoco/src/detection/mod.rs",
        // builds the cache during `evaluate()`
        "crates/hotcoco/src/detection/evaluate.rs",
        // declares the read-only context that carries it into the fan-out, and
        // consumes exactly one cell per call inside that fan-out
        "crates/hotcoco/src/detection/matching.rs",
    ];

    let violations = scan(
        |path| CACHE_OWNERS.contains(&path),
        |line| line.contains(".ious") || line.contains("ious:"),
    );

    assert!(
        violations.is_empty(),
        "the whole-dataset similarity cache is driver-private; only {} may touch \
         it.\nFound a direct field access at:\n  {}\n\n\
         Call `COCOeval::cell_ious(img_id, cat_id)` instead — it hands out one \
         cell and cannot leak the map. Exposing the map as a shared \"similarity \
         cache\" type would foreclose the 1.2 recompute-instead-of-retain lever.",
        CACHE_OWNERS.join(", "),
        violations.join("\n  ")
    );
}

/// The path an import brings in, for every spelling of `use`.
///
/// Handles `use`, `pub use`, and the restricted forms `pub(crate) use`,
/// `pub(super) use`, `pub(in path) use`. Matching only the first two — which is
/// what the first allowlist did — makes `pub(crate) use crate::detection::EvalMode`
/// invisible rather than violating, and that is the *most* likely spelling for a
/// real leak: a layer smuggling in a driver type wants it crate-visible.
fn import_target(line: &str) -> Option<&str> {
    let mut rest = line.trim_start();
    if let Some(after_pub) = rest.strip_prefix("pub") {
        let after_pub = after_pub.trim_start();
        rest = match after_pub.strip_prefix('(') {
            Some(vis) => vis.split_once(')')?.1.trim_start(),
            None => after_pub,
        };
    }
    rest.strip_prefix("use ")
}

/// Every import in `dir` whose target is not on `allowed`.
///
/// **Allowlist, not banlist.** The first version banned the spelling
/// `crate::detection`, and it was worthless: the crate re-exports ~25 of the same
/// types at its root, so `use crate::COCOeval` put a family driver inside
/// `metrics/` with the suite fully green. A banlist has to enumerate every path to
/// a thing — and a re-export or rename silently adds one.
///
/// # Why `super::` needs structural handling, not an allowlist entry
///
/// The *second* version allowlisted `super::` as "my own layer" and was bypassable
/// four ways. `super::` is relative, so what it reaches depends on where it is
/// written:
///
/// | Written at | Resolves to | Safe? |
/// |---|---|---|
/// | top level of `metrics/counts.rs` | `crate::metrics` | yes |
/// | top level of `metrics/mod.rs` | **the crate root** — all ~25 re-exports | no |
/// | `super::super::` at top level | at or above the crate root | no |
/// | inside `mod tests { … }` (indented) | the enclosing module | yes |
///
/// So a bare `super::` is accepted only from a non-`mod.rs` file, `super::super::`
/// is never accepted at top level, and indented `use` statements — which are inside
/// a nested `mod`, where one `super::` is consumed by the nesting — are left alone.
fn foreign_imports(dir: &str, allowed: &[&str]) -> Vec<String> {
    sources()
        .iter()
        .filter(|f| f.path.contains(dir))
        .flat_map(|f| {
            // `super::` from a module root climbs out of the layer.
            let is_module_root = f.path.ends_with("/mod.rs");
            f.lines.iter().filter_map(move |(n, line)| {
                // Indented => inside a nested `mod`, where `super::` stays local.
                let nested = line.starts_with(char::is_whitespace);
                let target = import_target(line)?;

                let ok = if target.starts_with("super::super::") {
                    nested
                } else if target.starts_with("super::") {
                    nested || !is_module_root
                } else {
                    allowed.iter().any(|prefix| target.starts_with(prefix))
                };

                (!ok).then(|| format!("{}:{n}  {}", f.path, line.trim()))
            })
        })
        .collect()
}

/// Imports every layer may reach for: the standard library and the shared
/// third-party substrate. None of these can carry a family's semantics.
///
/// `super::`/`self::` are deliberately absent — [`foreign_imports`] handles them
/// structurally, because whether they escape the layer depends on the file.
const NEUTRAL: &[&str] = &["std::", "core::", "serde", "rand", "rayon", "self::"];

/// What `metrics/` may reach: sibling metric functions, the crate's Result
/// type, and the cross-family output contract. Shared by the `use`-statement
/// check and the inline-path check so the two can never disagree about what
/// the layer means.
fn metrics_allowed() -> Vec<&'static str> {
    [
        NEUTRAL,
        &[
            "crate::metrics", // sibling metric functions
            "crate::error",   // the crate's Result type
            "crate::report",  // the cross-family output contract
        ],
    ]
    .concat()
}

/// What `primitives/` may reach. The sibling kernels are listed one by one
/// rather than as `crate::primitives`, so that adding a fourth kernel is a
/// deliberate edit here. Nothing from `metrics` is permitted at all — the
/// dependency runs one way.
fn primitives_allowed() -> Vec<&'static str> {
    [
        NEUTRAL,
        &[
            "crate::primitives::sim",
            "crate::primitives::greedy",
            "crate::primitives::assign",
            "crate::geometry",
            "crate::mask",
            "crate::types",
        ],
    ]
    .concat()
}

/// Every maximal `crate::`-rooted path spelled inline in a line of code.
///
/// A path is `crate::` followed by identifier characters and `::` separators;
/// the scan requires a non-identifier character (or line start) before the
/// keyword so `my_crate::x` is not a match.
fn inline_crate_paths(line: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    while let Some(pos) = line[start..].find("crate::") {
        let begin = start + pos;
        let ok_prefix = line[..begin]
            .chars()
            .next_back()
            .is_none_or(|prev| !(prev.is_alphanumeric() || prev == '_'));
        let end = line[begin..]
            .char_indices()
            .find(|&(i, c)| i >= "crate::".len() && !(c.is_alphanumeric() || c == '_' || c == ':'))
            .map_or(line.len(), |(i, _)| begin + i);
        if ok_prefix {
            out.push(&line[begin..end]);
        }
        start = end.max(begin + "crate::".len());
    }
    out
}

/// Every inline `crate::…` path in `dir`'s shipped code whose prefix is not on
/// `allowed` — the qualified-path twin of [`foreign_imports`].
///
/// The `use`-statement check alone was bypassable by construction: writing
/// `crate::detection::EvalMode::Coco` (or `crate::COCOeval`, through the root
/// re-exports) directly in a function body imports nothing and passed clean.
/// Same allowlist discipline as the import check — a path is a violation
/// unless its prefix is sanctioned, so the ~25 crate-root re-exports are
/// violations by default rather than needing enumeration.
///
/// The scan stops at the first `#[cfg(test)]` line: unit-test modules sit at
/// the bottom of every file in this crate, are not shipped, and legitimately
/// reach across layers (e.g. `metrics` tests build fixtures on
/// `crate::params::default_rec_thrs`). What it deliberately does not cover:
/// relative spellings (`super::` chains written inline without a `use`) and
/// paths split across lines by rustfmt — both possible, neither the accidental
/// reintroduction this file exists to catch.
fn inline_foreign_paths(dir: &str, allowed: &[&str]) -> Vec<String> {
    sources()
        .iter()
        .filter(|f| f.path.contains(dir))
        .flat_map(|f| {
            f.lines
                .iter()
                .take_while(|(_, l)| !l.trim_start().starts_with("#[cfg(test)"))
                .filter(|(_, l)| import_target(l).is_none())
                .filter(|(_, line)| {
                    inline_crate_paths(line)
                        .iter()
                        .any(|path| !allowed.iter().any(|p| path.starts_with(p)))
                })
                .map(move |(n, line)| format!("{}:{n}  {}", f.path, line.trim()))
        })
        .collect()
}

/// `metrics/` computes numbers from matches; it may not know which family called.
///
/// The whole point of the functional layer is that
/// `metrics::calibration::calibration_curve(scores, matched, n_bins)` is callable
/// by detection, tracking, or a user with two arrays and no evaluator at all. One
/// import of a family driver — under *any* of its names — and that stops being
/// true: the function silently becomes detection machinery filed in the wrong
/// drawer, which is the state 1.0 exists to fix.
///
/// Doc comments are exempt (the scanner drops `//`-prefixed lines), so `metrics/`
/// is free to *point at* its detection adapters in rustdoc. Naming them in prose
/// is documentation; importing them is a dependency.
#[test]
fn metrics_never_depends_on_a_family_driver() {
    let allowed = metrics_allowed();
    let violations = foreign_imports("/metrics/", &allowed);

    assert!(
        violations.is_empty(),
        "`metrics/` may only import {allowed:?}.\nFound:\n  {}\n\n\
         A metric function takes flat arrays — `(scores, matched)`, label pairs, a \
         closure. If it needs something from a family driver, that something is the \
         adapter's job to extract and pass in. Move the family-specific part into \
         `detection/` and keep the math here.\n\n\
         Widening this allowlist is a decision about what the layer means. Make it \
         deliberately, not to make a build pass.",
        violations.join("\n  ")
    );
}

/// The inline-path twin of [`metrics_never_depends_on_a_family_driver`].
///
/// A fully qualified `crate::detection::…` (or `crate::COCOeval`, via the root
/// re-exports) written in a function body imports nothing, so the `use` check
/// above never sees it — and that spelling is exactly what rustfmt produces
/// when someone reaches across layers "just once" without adding an import.
/// Same allowlist, same meaning; only the spelling differs.
#[test]
fn metrics_never_names_a_family_driver_inline() {
    let allowed = metrics_allowed();
    let violations = inline_foreign_paths("/metrics/", &allowed);

    assert!(
        violations.is_empty(),
        "`metrics/` may only name {allowed:?} in qualified paths.\nFound:\n  {}\n\n\
         See `metrics_never_depends_on_a_family_driver` — the rule is the same; \
         writing the path inline instead of importing it does not change what \
         the layer now depends on.",
        violations.join("\n  ")
    );
}

/// `primitives/` produces matches; it may not depend on the layer that scores them.
///
/// The dependency runs one way — `metrics` may use `primitives`, never the
/// reverse. A kernel that reached into `metrics` would be scoring, not matching,
/// and the "what does it produce?" split that separates the two modules would stop
/// describing anything.
///
#[test]
fn primitives_never_depends_on_metrics() {
    let allowed = primitives_allowed();
    let violations = foreign_imports("/primitives/", &allowed);

    assert!(
        violations.is_empty(),
        "`primitives/` may only import {allowed:?}.\nFound:\n  {}\n\n\
         Kernels match; metrics score. If a primitive needs a metric, the layering \
         is inverted — the caller should compose the two instead.",
        violations.join("\n  ")
    );
}

/// The inline-path twin of [`primitives_never_depends_on_metrics`] — see
/// [`metrics_never_names_a_family_driver_inline`] for why the `use` check
/// alone is not enough.
#[test]
fn primitives_never_names_metrics_inline() {
    let allowed = primitives_allowed();
    let violations = inline_foreign_paths("/primitives/", &allowed);

    assert!(
        violations.is_empty(),
        "`primitives/` may only name {allowed:?} in qualified paths.\nFound:\n  {}\n\n\
         Kernels match; metrics score. Writing `crate::metrics::…` (or a crate-root \
         re-export) inline instead of importing it does not change the dependency.",
        violations.join("\n  ")
    );
}

/// The semver lint override in `Cargo.toml` exists only while the published
/// baseline is 1.0.1. Once the version bumps, the override would silently keep
/// downgrading every removed public field to a warning, so it must go in the
/// same commit.
#[test]
fn semver_override_expires_with_1_0_1() {
    let manifest = fs::read_to_string(workspace_root().join("crates/hotcoco/Cargo.toml"))
        .expect("crate manifest");
    if manifest.contains("[package.metadata.cargo-semver-checks.lints]") {
        assert_eq!(
            env!("CARGO_PKG_VERSION"),
            "1.0.1",
            "1.1.0 shipped: delete the [package.metadata.cargo-semver-checks.lints] block \
             from crates/hotcoco/Cargo.toml"
        );
    }
}
