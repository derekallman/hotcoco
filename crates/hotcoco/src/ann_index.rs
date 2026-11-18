//! The annotation indices behind [`COCO`](crate::COCO)'s queries: id to
//! position, which annotations an image holds, and which an (image,
//! category) pair holds.
//!
//! The groupings are ranges into flat id lists rather than one heap vector
//! per key. A 500k-detection results file has about 285k distinct pairs, and
//! a vector each meant 285k allocations to build and 285k to free, several
//! times the memory of the ids themselves. Here the per-image grouping is
//! built in two counting passes, the per-pair grouping by sorting each
//! image's slice by category in parallel, and a lookup is one hash probe on
//! the image plus a binary search over that image's categories.
//!
//! Order is contract. Within an image the ids keep the annotation array's
//! order, and within a pair likewise: pycocotools builds `_gts` by iterating
//! `dataset['annotations']` once, so array order is what feeds the matcher,
//! and the greedy tie-break (`>=`, later GT wins on equal IoU) makes that
//! order observable through `evalImgs`. Official COCO files are id-ordered
//! anyway; converted or merged files are where the two orders differ.
//!
//! Positions and counts are `u32`: `build` refuses more than `u32::MAX`
//! annotations, which is 32 GB of ids before anything else.

use std::ops::Range;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::primitives::sim::worth_parallel;
use crate::types::Annotation;

/// Annotation ids grouped by image and by (image, category).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AnnIndex {
    /// Image ids in first-seen order; an image's slot is its position here.
    img_ids: Vec<u64>,
    img_slot: FxHashMap<u64, u32>,
    /// Ids grouped by image, array order within an image; image `slot` owns
    /// `by_img[img_offsets[slot]..img_offsets[slot + 1]]`.
    by_img: Vec<u64>,
    img_offsets: Vec<u32>,
    /// The same partition with each image's ids stably sorted by category,
    /// and image `slot`'s categories as `groups[group_offsets[slot]..]`,
    /// each `(cat, start, len)` into `by_pair`, ascending by category.
    by_pair: Vec<u64>,
    groups: Vec<(u64, u32, u32)>,
    group_offsets: Vec<u32>,
}

/// The index of no annotations: every offset table is the single `0` a CSR
/// table of zero rows holds.
impl Default for AnnIndex {
    fn default() -> Self {
        AnnIndex::build(&[])
    }
}

impl AnnIndex {
    pub(crate) fn build(anns: &[Annotation]) -> Self {
        // The one read of `anns`: each annotation's `(image, category, id)`,
        // so the passes after it read 24 bytes per annotation rather than a
        // whole record. It fans out past the shared threshold: at a million
        // detections the records are hundreds of megabytes and the keys a
        // tenth of that.
        let key = |ann: &Annotation| (ann.image_id, ann.category_id, ann.id);
        let keys: Vec<(u64, u64, u64)> = if worth_parallel(anns.len()) {
            anns.par_iter().map(key).collect()
        } else {
            anns.iter().map(key).collect()
        };
        Self::from_keys(keys.len(), |i| keys[i])
    }

    /// The index of `n` annotations, annotation `i` read as
    /// `(image_id, category_id, id)` through `key`, in array order: what
    /// [`build`](Self::build) reads off the records, for a caller that holds
    /// the three fields without the records — the detections
    /// [`crate::DetectionColumns`] keeps. One builder, so the two cannot group
    /// or order differently.
    pub(crate) fn from_keys(n: usize, key: impl Fn(usize) -> (u64, u64, u64) + Sync) -> Self {
        assert!(
            u32::try_from(n).is_ok(),
            "annotation index positions are u32"
        );
        let parallel = worth_parallel(n);
        // One slot per distinct image, in first-seen order, which is
        // contract, and every image's annotations by array position, in
        // array order within it.
        //
        // When every image's annotations form one run, as a detector's
        // output listed image by image does, the runs are the slots and the
        // array order is already by image: the runs' starts are found in
        // parallel, and slots are assigned over the runs alone, not key by
        // key. A run whose image was seen before means the runs are not
        // distinct, and the serial walk below does the general case.
        let starts: Vec<u32> = {
            let starts_run = |&i: &usize| i == 0 || key(i).0 != key(i - 1).0;
            if parallel {
                (0..n)
                    .into_par_iter()
                    .filter(starts_run)
                    .map(|i| i as u32)
                    .collect()
            } else {
                (0..n).filter(starts_run).map(|i| i as u32).collect()
            }
        };
        let mut img_ids = Vec::with_capacity(starts.len());
        let mut img_slot = FxHashMap::default();
        img_slot.reserve(starts.len());
        let grouped = starts.iter().all(|&start| {
            let image_id = key(start as usize).0;
            let fresh = img_slot.insert(image_id, img_ids.len() as u32).is_none();
            img_ids.push(image_id);
            fresh
        });
        let (img_offsets, positions): (Vec<u32>, Vec<u32>) = if grouped {
            let mut offsets = starts;
            offsets.push(n as u32);
            let positions = if parallel {
                (0..n as u32).into_par_iter().collect()
            } else {
                (0..n as u32).collect()
            };
            (offsets, positions)
        } else {
            drop(starts);
            img_ids.clear();
            img_slot.clear();
            // Slots key by key; consecutive annotations usually share an
            // image, so the slot is looked up again only when it changes.
            let mut img_offsets: Vec<u32> = vec![0];
            let mut last: Option<(u64, u32)> = None;
            let slots: Vec<u32> = (0..n)
                .map(|i| {
                    let image_id = key(i).0;
                    let slot = match last {
                        Some((id, slot)) if id == image_id => slot,
                        _ => {
                            let slot = *img_slot.entry(image_id).or_insert_with(|| {
                                img_ids.push(image_id);
                                img_offsets.push(0);
                                img_ids.len() as u32 - 1
                            });
                            last = Some((image_id, slot));
                            slot
                        }
                    };
                    img_offsets[slot as usize + 1] += 1;
                    slot
                })
                .collect();
            for i in 1..img_offsets.len() {
                img_offsets[i] += img_offsets[i - 1];
            }
            // Every (category, id) at its image's next position.
            let mut cursor = img_offsets.clone();
            let mut positions = vec![0u32; n];
            for (i, &slot) in slots.iter().enumerate() {
                let at = &mut cursor[slot as usize];
                positions[*at as usize] = i as u32;
                *at += 1;
            }
            (img_offsets, positions)
        };
        let id_at = |p: u32| key(p as usize).2;
        let by_img: Vec<u64> = if parallel {
            positions.par_iter().map(|&p| id_at(p)).collect()
        } else {
            positions.iter().map(|&p| id_at(p)).collect()
        };
        // Per image, in parallel past the shared threshold: order its
        // annotations by category, keeping array order within one, and
        // run-length encode the categories. Positions are distinct and
        // ascend in array order, so an unstable sort on (category, position)
        // is exactly the stable sort by category, without the buffer a
        // stable sort allocates for every image.
        let mut by_pair = vec![0u64; n];
        let mut windows: Vec<(&mut [u64], &[u32], u32)> = Vec::with_capacity(img_ids.len());
        let mut rest = by_pair.as_mut_slice();
        for w in img_offsets.windows(2) {
            let (head, tail) = rest.split_at_mut((w[1] - w[0]) as usize);
            windows.push((head, &positions[w[0] as usize..w[1] as usize], w[0]));
            rest = tail;
        }
        //
        // A category id that fits 32 bits — every one in practice — packs
        // with the position into one `u64` whose order is the pair's, and a
        // sort of plain integers is the cheaper one.
        let group = |(packed, order): &mut (Vec<u64>, Vec<(u64, u32)>),
                     (out, pos, start): (&mut [u64], &[u32], u32)| {
            let mut groups: Vec<(u64, u32, u32)> = Vec::new();
            let mut push = |i: usize, cat: u64, p: u32, slot: &mut u64| {
                *slot = id_at(p);
                match groups.last_mut() {
                    Some((last, _, n)) if *last == cat => *n += 1,
                    _ => groups.push((cat, start + i as u32, 1)),
                }
            };
            packed.clear();
            let fits = pos.iter().all(|&p| {
                let cat = key(p as usize).1;
                packed.push(cat << 32 | u64::from(p));
                u32::try_from(cat).is_ok()
            });
            if fits {
                packed.sort_unstable();
                for (i, (&k, slot)) in packed.iter().zip(out.iter_mut()).enumerate() {
                    push(i, k >> 32, k as u32, slot);
                }
            } else {
                order.clear();
                order.extend(pos.iter().map(|&p| (key(p as usize).1, p)));
                order.sort_unstable();
                for (i, (&(cat, p), slot)) in order.iter().zip(out.iter_mut()).enumerate() {
                    push(i, cat, p, slot);
                }
            }
            groups
        };
        let per_img: Vec<Vec<(u64, u32, u32)>> = if parallel {
            windows
                .into_par_iter()
                .map_init(<(Vec<u64>, Vec<(u64, u32)>)>::default, group)
                .collect()
        } else {
            let mut scratch = <(Vec<u64>, Vec<(u64, u32)>)>::default();
            windows
                .into_iter()
                .map(|w| group(&mut scratch, w))
                .collect()
        };
        let mut groups = Vec::with_capacity(per_img.iter().map(Vec::len).sum());
        let mut group_offsets = Vec::with_capacity(per_img.len() + 1);
        group_offsets.push(0);
        for img_groups in per_img {
            groups.extend(img_groups);
            group_offsets.push(groups.len() as u32);
        }
        AnnIndex {
            img_ids,
            img_slot,
            by_img,
            img_offsets,
            by_pair,
            groups,
            group_offsets,
        }
    }

    /// The annotation ids of one image, in array order.
    pub(crate) fn for_img(&self, img_id: u64) -> &[u64] {
        match self.img_slot.get(&img_id) {
            Some(&slot) => &self.by_img[span(&self.img_offsets, slot)],
            None => &[],
        }
    }

    /// The annotation ids of one image grouped by category, ascending, and in
    /// array order within a category.
    pub(crate) fn for_img_by_cat(&self, img_id: u64) -> &[u64] {
        match self.img_slot.get(&img_id) {
            Some(&slot) => &self.by_pair[span(&self.img_offsets, slot)],
            None => &[],
        }
    }

    /// The annotation ids of one (image, category) pair, in array order.
    pub(crate) fn for_img_cat(&self, img_id: u64, cat_id: u64) -> &[u64] {
        self.ids_in(self.groups_of(img_id), cat_id)
    }

    /// [`for_img_cat`](Self::for_img_cat) for an image whose groups the
    /// caller already holds from [`groups_of`](Self::groups_of): the binary
    /// search without the image lookup, for a walk over one image's pairs.
    pub(crate) fn ids_in(&self, groups: &[(u64, u32, u32)], cat_id: u64) -> &[u64] {
        match groups.binary_search_by_key(&cat_id, |&(cat, _, _)| cat) {
            Ok(i) => {
                let (_, start, len) = groups[i];
                &self.by_pair[start as usize..(start + len) as usize]
            }
            Err(_) => &[],
        }
    }

    /// The categories one image has annotations in, ascending.
    pub(crate) fn cats_of(&self, img_id: u64) -> impl Iterator<Item = u64> + '_ {
        self.groups_of(img_id).iter().map(|&(cat, _, _)| cat)
    }

    /// One image's `(category, start, len)` groups, ascending by category.
    pub(crate) fn groups_of(&self, img_id: u64) -> &[(u64, u32, u32)] {
        match self.img_slot.get(&img_id) {
            Some(&slot) => &self.groups[span(&self.group_offsets, slot)],
            None => &[],
        }
    }

    /// How many (image, category) pairs hold an annotation.
    pub(crate) fn pair_count(&self) -> usize {
        self.groups.len()
    }

    /// Every image with at least one annotation, in first-seen order.
    pub(crate) fn img_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.img_ids.iter().copied()
    }

    /// Every (image, category) pair with at least one annotation: images in
    /// first-seen order, categories ascending within an image.
    pub(crate) fn pairs(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.img_ids.iter().enumerate().flat_map(|(slot, &img_id)| {
            self.groups[span(&self.group_offsets, slot as u32)]
                .iter()
                .map(move |&(cat, _, _)| (img_id, cat))
        })
    }
}

/// Slot `slot`'s range in a CSR offset table.
fn span(offsets: &[u32], slot: u32) -> Range<usize> {
    offsets[slot as usize] as usize..offsets[slot as usize + 1] as usize
}

/// Annotation id to position in the annotation array.
///
/// Results files get ids `1..=n` in array order from `load_res`, and many
/// ground-truth files are written the same way; for those the position is
/// the id and no map is needed. Everything else takes a hash map, with the
/// last annotation winning a repeated id as in pycocotools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum IdIndex {
    Dense { len: usize },
    Sparse(FxHashMap<u64, usize>),
}

impl Default for IdIndex {
    fn default() -> Self {
        IdIndex::Dense { len: 0 }
    }
}

/// Repeated annotation ids seen while building an [`IdIndex`]: how many,
/// and the first one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Duplicates {
    pub(crate) count: usize,
    pub(crate) first: u64,
}

impl IdIndex {
    pub(crate) fn build(anns: &[Annotation]) -> (Self, Option<Duplicates>) {
        // `load_res` numbers its results `1..=N`, so a results set is
        // usually dense; the check reads every record once, in parallel
        // past the shared threshold.
        let dense_at = |(i, ann): (usize, &Annotation)| ann.id == i as u64 + 1;
        let dense = if worth_parallel(anns.len()) {
            anns.par_iter().enumerate().all(dense_at)
        } else {
            anns.iter().enumerate().all(dense_at)
        };
        if dense {
            return (IdIndex::Dense { len: anns.len() }, None);
        }
        let mut map = FxHashMap::with_capacity_and_hasher(anns.len(), rustc_hash::FxBuildHasher);
        let mut dups: Option<Duplicates> = None;
        for (i, ann) in anns.iter().enumerate() {
            if map.insert(ann.id, i).is_some() {
                dups.get_or_insert(Duplicates {
                    count: 0,
                    first: ann.id,
                })
                .count += 1;
            }
        }
        (IdIndex::Sparse(map), dups)
    }

    pub(crate) fn position(&self, id: u64) -> Option<usize> {
        match self {
            IdIndex::Dense { len } => (1..=*len as u64).contains(&id).then(|| id as usize - 1),
            IdIndex::Sparse(map) => map.get(&id).copied(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rand::{Rng, SeedableRng};

    use super::*;

    fn ann(id: u64, image_id: u64, category_id: u64) -> Annotation {
        Annotation {
            id,
            image_id,
            category_id,
            ..Default::default()
        }
    }

    #[test]
    fn groupings_keep_array_order_and_find_every_pair() {
        // Image 7 first, then 3, then 7 again; categories interleaved.
        let anns = [
            ann(10, 7, 2),
            ann(11, 3, 1),
            ann(12, 7, 1),
            ann(13, 7, 2),
            ann(14, 3, 1),
            ann(15, 9, 5),
        ];
        let ix = AnnIndex::build(&anns);
        assert_eq!(ix.for_img(7), &[10, 12, 13]);
        assert_eq!(ix.for_img(3), &[11, 14]);
        assert_eq!(ix.for_img(9), &[15]);
        assert_eq!(ix.for_img(8), &[] as &[u64]);
        assert_eq!(ix.for_img_cat(7, 2), &[10, 13]);
        assert_eq!(ix.for_img_cat(7, 1), &[12]);
        assert_eq!(ix.for_img_cat(3, 1), &[11, 14]);
        assert_eq!(ix.for_img_cat(3, 2), &[] as &[u64]);
        assert_eq!(ix.for_img_cat(8, 1), &[] as &[u64]);
        assert_eq!(ix.img_ids().collect::<Vec<_>>(), vec![7, 3, 9]);
        assert_eq!(
            ix.pairs().collect::<Vec<_>>(),
            vec![(7, 1), (7, 2), (3, 1), (9, 5)]
        );
        let empty = AnnIndex::default();
        assert_eq!(
            (empty.for_img(7), empty.pairs().count()),
            (&[] as &[u64], 0)
        );
    }

    /// Many images and categories, so the parallel per-image sort, the
    /// offset tables past the first few slots, and a binary search over more
    /// than two categories all run — checked against naive maps, for keys in
    /// every order the builder takes a different path for: images
    /// interleaved, each image one run (no key moves), and one run each with
    /// ids that do not ascend in array order, which a sort on ids instead of
    /// positions would get wrong. Sizes on both sides of the parallel
    /// threshold.
    #[test]
    fn index_agrees_with_naive_maps_at_scale() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x9E37_79B9);
        for n in [300usize, 3000] {
            let interleaved: Vec<Annotation> = (0..n as u64)
                .map(|i| ann(i * 7 + 1, rng.random_range(0..41), rng.random_range(0..11)))
                .collect();
            let mut grouped = interleaved.clone();
            grouped.sort_by_key(|a| a.image_id);
            let mut shuffled_ids = grouped.clone();
            for i in (1..n).rev() {
                let j = rng.random_range(0..=i);
                let id = shuffled_ids[i].id;
                shuffled_ids[i].id = shuffled_ids[j].id;
                shuffled_ids[j].id = id;
            }
            // One run per image but the first image's, which comes back at
            // the end: the run check fails after it has filled its tables.
            // Category ids past 32 bits, which do not pack with a position.
            let mut wide_cats = interleaved.clone();
            for a in wide_cats.iter_mut().step_by(3) {
                a.category_id += 1 << 40;
            }
            let mut recurring = grouped.clone();
            let mut back = grouped[0].clone();
            back.id = 10 * n as u64 + 1;
            recurring.push(back);
            for (label, anns) in [
                ("interleaved", &interleaved),
                ("grouped", &grouped),
                ("grouped, ids shuffled", &shuffled_ids),
                ("grouped, first image recurring", &recurring),
                ("category ids past 32 bits", &wide_cats),
            ] {
                assert_matches_naive_maps(&format!("{label}, n={n}"), anns);
            }
        }
    }

    fn assert_matches_naive_maps(label: &str, anns: &[Annotation]) {
        let mut first_seen: Vec<u64> = Vec::new();
        let mut by_img: HashMap<u64, Vec<u64>> = HashMap::new();
        let mut by_pair: HashMap<(u64, u64), Vec<u64>> = HashMap::new();
        for a in anns {
            if !by_img.contains_key(&a.image_id) {
                first_seen.push(a.image_id);
            }
            by_img.entry(a.image_id).or_default().push(a.id);
            by_pair
                .entry((a.image_id, a.category_id))
                .or_default()
                .push(a.id);
        }
        let ix = AnnIndex::build(anns);
        assert_eq!(
            ix.img_ids().collect::<Vec<_>>(),
            first_seen,
            "{label}: image order"
        );
        for (img, ids) in &by_img {
            assert_eq!(ix.for_img(*img), ids.as_slice(), "{label}: image {img}");
            let mut cats: Vec<u64> = by_pair
                .keys()
                .filter(|k| k.0 == *img)
                .map(|k| k.1)
                .collect();
            cats.sort_unstable();
            assert_eq!(
                ix.cats_of(*img).collect::<Vec<_>>(),
                cats,
                "{label}: categories of {img}"
            );
            let by_cat: Vec<u64> = cats
                .iter()
                .flat_map(|&c| by_pair[&(*img, c)].clone())
                .collect();
            assert_eq!(
                ix.for_img_by_cat(*img),
                by_cat.as_slice(),
                "{label}: image {img} by category"
            );
        }
        for (&(img, cat), ids) in &by_pair {
            assert_eq!(
                ix.for_img_cat(img, cat),
                ids.as_slice(),
                "{label}: pair {img}/{cat}"
            );
        }
        assert_eq!(ix.for_img_cat(0, 99), &[] as &[u64]);
        let mut pairs: Vec<_> = ix.pairs().collect();
        pairs.sort_unstable();
        let mut expected: Vec<_> = by_pair.keys().copied().collect();
        expected.sort_unstable();
        assert_eq!(pairs, expected, "{label}: pairs");
        assert_eq!(ix.pair_count(), by_pair.len(), "{label}: pair count");
    }

    #[test]
    fn ids_one_to_n_are_dense_and_anything_else_is_a_map() {
        let dense = [ann(1, 1, 1), ann(2, 1, 1), ann(3, 2, 1)];
        let (ix, dups) = IdIndex::build(&dense);
        assert_eq!(ix, IdIndex::Dense { len: 3 });
        assert_eq!(dups, None);
        assert_eq!(ix.position(1), Some(0));
        assert_eq!(ix.position(3), Some(2));
        assert_eq!(ix.position(0), None);
        assert_eq!(ix.position(4), None);

        let sparse = [ann(10, 1, 1), ann(5, 1, 1), ann(10, 2, 1), ann(5, 2, 1)];
        let (ix, dups) = IdIndex::build(&sparse);
        assert!(matches!(ix, IdIndex::Sparse(_)));
        assert_eq!(
            dups,
            Some(Duplicates {
                count: 2,
                first: 10
            })
        );
        // Last occurrence wins, as in pycocotools.
        assert_eq!(ix.position(10), Some(2));
        assert_eq!(ix.position(5), Some(3));
        assert_eq!(ix.position(6), None);
    }
}
