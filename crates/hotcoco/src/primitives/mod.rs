//! Matching kernels — the shared substrate that decides what pairs with what.
//!
//! Four kernels, and nothing else: [`sim`] computes similarity between two sets,
//! [`greedy`] resolves it into COCO's rank-ordered assignment, [`assign`] resolves
//! it optimally via rectangular LSAP, and [`panoptic`] counts pixel overlaps
//! between two label maps and matches segments on them. Family drivers
//! (detection, panoptic; tracking/concepts later) compose them. The one
//! non-kernel export is [`greedy::ThreshMatrix`], the container shape of the
//! greedy kernel's per-threshold output.
//!
//! Kernels here produce matches and similarities; the functions in
//! [`metrics`](crate::metrics) turn those into numbers. There is exactly one
//! implementation of each kernel — `tests/architecture.rs` fails the build if a
//! second appears.
//!
//! # Stability
//!
//! These APIs are **provisional** — not frozen until 1.4, with additive change
//! (new [`sim::SimKind`] variants) expected in the 1.x minors. The similarity
//! kernels re-exported onto Tier-1 paths are the exception, already frozen; see
//! [`sim`][sim#where-the-math-lives].
//!
//! # Kernels are stateless
//!
//! No primitive caches or retains input across calls, and no contract in this
//! module may come to require whole-sequence retention. Batched or per-sequence
//! helpers added later must iterate-and-consume — never return every timestep's
//! matrix at once.

pub mod assign;
pub mod greedy;
pub mod panoptic;
pub mod sim;
