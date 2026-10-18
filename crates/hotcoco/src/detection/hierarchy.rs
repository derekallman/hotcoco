//! Category hierarchy for Open Images detection evaluation.
//!
//! Supports three construction methods:
//! - [`Hierarchy::from_parent_map`] — from an explicit child→parent mapping.
//! - [`Hierarchy::from_categories`] — derives hierarchy from `Category.supercategory` fields.
//! - [`Hierarchy::from_oid_json`] — parses the Open Images hierarchy JSON format.

use std::collections::HashMap;

use serde::Deserialize;

use crate::Category;

/// Resolve a name to its category id, minting a virtual one on first miss.
///
/// Shared by [`Hierarchy::from_categories`] (over supercategory names) and
/// [`Hierarchy::from_oid_json`] (over OID labels) — both need "real id if
/// known, else a stable id reused across repeat references, else a fresh one
/// counting down from `u64::MAX - 1`." The caller does its own real-id lookup
/// (its map's key type varies — `&str` vs `String`) and passes the result in,
/// so this function needs no generic over the map's key.
fn resolve_id(
    name: &str,
    real_id: Option<u64>,
    virtual_ids: &mut HashMap<String, u64>,
    next_virtual_id: &mut u64,
) -> u64 {
    if let Some(id) = real_id {
        id
    } else if let Some(&id) = virtual_ids.get(name) {
        id
    } else {
        let id = *next_virtual_id;
        virtual_ids.insert(name.to_string(), id);
        *next_virtual_id = next_virtual_id.wrapping_sub(1);
        id
    }
}

/// A category hierarchy (tree or forest) used for Open Images evaluation.
///
/// Each category can have at most one parent. The hierarchy supports ancestor
/// lookups (self + all ancestors up to the root) for GT/DT expansion.
#[derive(Debug, Clone)]
pub struct Hierarchy {
    /// cat_id → list of direct child cat_ids
    children_map: HashMap<u64, Vec<u64>>,
    /// cat_id → parent cat_id
    parent_map: HashMap<u64, u64>,
    /// cat_id → [self, parent, grandparent, ...] (self included, root last)
    ancestors_map: HashMap<u64, Vec<u64>>,
    /// Virtual node IDs → human-readable name (supercategory string or OID label).
    /// Only populated for nodes that don't exist in the original dataset.
    pub virtual_names: HashMap<u64, String>,
}

impl Hierarchy {
    /// Build a hierarchy from an explicit child→parent mapping.
    ///
    /// Constructs `children_map` and precomputes `ancestors_map` for all nodes.
    pub fn from_parent_map(parent_map: HashMap<u64, u64>) -> Self {
        // Build children_map
        let mut children_map: HashMap<u64, Vec<u64>> = HashMap::new();
        for (&child, &parent) in &parent_map {
            children_map.entry(parent).or_default().push(child);
        }

        // Collect all node IDs (parents and children)
        let mut all_ids: Vec<u64> = parent_map.keys().copied().collect();
        for &parent in parent_map.values() {
            if !parent_map.contains_key(&parent) {
                all_ids.push(parent);
            }
        }
        all_ids.sort_unstable();
        all_ids.dedup();

        // Precompute ancestors for every node
        let mut ancestors_map: HashMap<u64, Vec<u64>> = HashMap::new();
        for &id in &all_ids {
            let mut ancestors = vec![id];
            let mut current = id;
            while let Some(&parent) = parent_map.get(&current) {
                // A cyclic parent map (possible via mutually-referencing
                // `supercategory` names or a hand-built map) would loop
                // forever; stop at the first repeat, so cycle members list
                // each other once and the walk terminates. Chains are short,
                // so the linear scan beats a per-walk set allocation.
                if ancestors.contains(&parent) {
                    break;
                }
                ancestors.push(parent);
                current = parent;
            }
            ancestors_map.insert(id, ancestors);
        }

        Hierarchy {
            children_map,
            parent_map,
            ancestors_map,
            virtual_names: HashMap::new(),
        }
    }

    /// Derive a hierarchy from `Category.supercategory` fields.
    ///
    /// For each category with a `supercategory` value, looks for a matching category
    /// by name. If no matching category exists, creates a virtual node (using IDs
    /// counting down from `u64::MAX - 1`) to represent the supercategory.
    pub fn from_categories(categories: &[Category]) -> Self {
        let name_to_id = crate::types::cat_name_to_id(categories);

        let mut parent_map: HashMap<u64, u64> = HashMap::new();
        let mut virtual_id = u64::MAX - 1;
        let mut virtual_names: HashMap<String, u64> = HashMap::new();

        for cat in categories {
            if let Some(ref supercat) = cat.supercategory {
                if supercat == &cat.name {
                    // Skip self-referencing supercategories
                    continue;
                }
                let real_id = name_to_id.get(supercat.as_str()).copied();
                let parent_id = resolve_id(supercat, real_id, &mut virtual_names, &mut virtual_id);
                parent_map.insert(cat.id, parent_id);
            }
        }

        let mut h = Self::from_parent_map(parent_map);
        // Invert virtual_names (name → id) back to (id → name) for expand.rs.
        h.virtual_names = virtual_names.into_iter().map(|(k, v)| (v, k)).collect();
        h
    }

    /// Parse an Open Images hierarchy JSON string.
    ///
    /// The JSON has a nested structure with `LabelName` and `Subcategory` fields.
    /// `label_to_id` maps OID label strings such as "/m/0jbk" to category IDs.
    /// Labels not found in `label_to_id` get virtual nodes.
    ///
    /// # Errors
    ///
    /// [`Error::Json`](crate::error::Error::Json) if `json` is not the OID
    /// hierarchy shape.
    pub fn from_oid_json(
        json: &str,
        label_to_id: &HashMap<String, u64>,
    ) -> crate::error::Result<Self> {
        let root: OidNode = serde_json::from_str(json)?;

        let mut parent_map: HashMap<u64, u64> = HashMap::new();
        let mut virtual_id = u64::MAX - 1;
        let mut virtual_labels: HashMap<String, u64> = HashMap::new();

        fn walk(
            node: &OidNode,
            parent_map: &mut HashMap<u64, u64>,
            label_to_id: &HashMap<String, u64>,
            virtual_labels: &mut HashMap<String, u64>,
            virtual_id: &mut u64,
        ) {
            let parent_real = label_to_id.get(&node.label_name).copied();
            let parent_id = resolve_id(&node.label_name, parent_real, virtual_labels, virtual_id);
            for child in &node.subcategory {
                let child_real = label_to_id.get(&child.label_name).copied();
                let child_id =
                    resolve_id(&child.label_name, child_real, virtual_labels, virtual_id);
                parent_map.insert(child_id, parent_id);
                walk(child, parent_map, label_to_id, virtual_labels, virtual_id);
            }
        }

        walk(
            &root,
            &mut parent_map,
            label_to_id,
            &mut virtual_labels,
            &mut virtual_id,
        );

        let mut h = Self::from_parent_map(parent_map);
        h.virtual_names = virtual_labels.into_iter().map(|(k, v)| (v, k)).collect();
        Ok(h)
    }

    /// Returns ancestors of `cat_id`: `[self, parent, grandparent, ...]`.
    ///
    /// Returns a single-element slice `[cat_id]` if `cat_id` has no parent,
    /// or an empty slice if `cat_id` is entirely unknown to the hierarchy.
    pub fn ancestors(&self, cat_id: u64) -> &[u64] {
        self.ancestors_map
            .get(&cat_id)
            .map_or(&[], std::vec::Vec::as_slice)
    }

    /// Returns direct children of `cat_id`, or an empty slice if none.
    pub fn children(&self, cat_id: u64) -> &[u64] {
        self.children_map
            .get(&cat_id)
            .map_or(&[], std::vec::Vec::as_slice)
    }

    /// Returns the parent of `cat_id`, or `None` if it is a root.
    pub fn parent(&self, cat_id: u64) -> Option<u64> {
        self.parent_map.get(&cat_id).copied()
    }

    /// Returns the human-readable name for a virtual node, if known.
    pub fn name_of(&self, id: u64) -> Option<&str> {
        self.virtual_names.get(&id).map(std::string::String::as_str)
    }

    /// Returns all node IDs known to this hierarchy (sorted).
    pub fn all_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.ancestors_map.keys().copied().collect();
        ids.sort_unstable();
        ids
    }
}

/// A node in the Open Images hierarchy JSON format.
#[derive(Deserialize)]
struct OidNode {
    #[serde(rename = "LabelName")]
    label_name: String,
    #[serde(rename = "Subcategory", default)]
    subcategory: Vec<OidNode>,
}
