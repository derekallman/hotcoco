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
        assert!(
            u32::try_from(anns.len()).is_ok(),
            "annotation index positions are u32"
        );
        // Pass 1: one slot per distinct image, in first-seen order, and how
        // many annotations each holds.
        let mut img_ids = Vec::new();
        let mut img_slot = FxHashMap::default();
        let mut img_offsets: Vec<u32> = vec![0];
        let slots: Vec<u32> = anns
            .iter()
            .map(|ann| {
                let slot = *img_slot.entry(ann.image_id).or_insert_with(|| {
                    img_ids.push(ann.image_id);
                    img_offsets.push(0);
                    img_ids.len() as u32 - 1
                });
                img_offsets[slot as usize + 1] += 1;
                slot
            })
            .collect();
        for i in 1..img_offsets.len() {
            img_offsets[i] += img_offsets[i - 1];
        }
        // Pass 2: place every (category, id) at its image's next position.
        let mut cursor = img_offsets.clone();
        let mut keyed = vec![(0u64, 0u64); anns.len()];
        for (ann, &slot) in anns.iter().zip(&slots) {
            let at = &mut cursor[slot as usize];
            keyed[*at as usize] = (ann.category_id, ann.id);
            *at += 1;
        }
        let by_img: Vec<u64> = keyed.iter().map(|&(_, id)| id).collect();
        // Per image, in parallel: stably sort its slice by category and
        // run-length encode the categories that sort produces.
        let mut slices: Vec<&mut [(u64, u64)]> = Vec::with_capacity(img_ids.len());
        let mut rest = keyed.as_mut_slice();
        for w in img_offsets.windows(2) {
            let (head, tail) = rest.split_at_mut((w[1] - w[0]) as usize);
            slices.push(head);
            rest = tail;
        }
        let per_img: Vec<Vec<(u64, u32, u32)>> = slices
            .into_par_iter()
            .zip(&img_offsets)
            .map(|(slice, &start)| {
                slice.sort_by_key(|&(cat, _)| cat);
                let mut groups: Vec<(u64, u32, u32)> = Vec::new();
                for (i, &(cat, _)) in slice.iter().enumerate() {
                    match groups.last_mut() {
                        Some((last, _, n)) if *last == cat => *n += 1,
                        _ => groups.push((cat, start + i as u32, 1)),
                    }
                }
                groups
            })
            .collect();
        let by_pair = keyed.into_iter().map(|(_, id)| id).collect();
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

    /// The annotation ids of one (image, category) pair, in array order.
    pub(crate) fn for_img_cat(&self, img_id: u64, cat_id: u64) -> &[u64] {
        let Some(&slot) = self.img_slot.get(&img_id) else {
            return &[];
        };
        let groups = &self.groups[span(&self.group_offsets, slot)];
        match groups.binary_search_by_key(&cat_id, |&(cat, _, _)| cat) {
            Ok(i) => {
                let (_, start, len) = groups[i];
                &self.by_pair[start as usize..(start + len) as usize]
            }
            Err(_) => &[],
        }
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
        if anns
            .iter()
            .enumerate()
            .all(|(i, ann)| ann.id == i as u64 + 1)
        {
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
    /// than two categories all run — checked against naive maps.
    #[test]
    fn index_agrees_with_naive_maps_at_scale() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x9E37_79B9);
        let anns: Vec<Annotation> = (0..3000)
            .map(|i| ann(i * 7 + 1, rng.random_range(0..41), rng.random_range(0..11)))
            .collect();
        let mut by_img: HashMap<u64, Vec<u64>> = HashMap::new();
        let mut by_pair: HashMap<(u64, u64), Vec<u64>> = HashMap::new();
        for a in &anns {
            by_img.entry(a.image_id).or_default().push(a.id);
            by_pair
                .entry((a.image_id, a.category_id))
                .or_default()
                .push(a.id);
        }
        let ix = AnnIndex::build(&anns);
        for (img, ids) in &by_img {
            assert_eq!(ix.for_img(*img), ids.as_slice(), "image {img}");
        }
        for (&(img, cat), ids) in &by_pair {
            assert_eq!(ix.for_img_cat(img, cat), ids.as_slice(), "pair {img}/{cat}");
        }
        assert_eq!(ix.for_img_cat(0, 99), &[] as &[u64]);
        let mut pairs: Vec<_> = ix.pairs().collect();
        pairs.sort_unstable();
        let mut expected: Vec<_> = by_pair.keys().copied().collect();
        expected.sort_unstable();
        assert_eq!(pairs, expected);
        assert_eq!(ix.img_ids().count(), by_img.len());
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
