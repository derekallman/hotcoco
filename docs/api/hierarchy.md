# Hierarchy

A category hierarchy for Open Images evaluation. Used by `COCOeval` to expand GT (and optionally DT) annotations up the hierarchy so that a "Dog" detection also counts as an "Animal" detection.

```python
from hotcoco import COCO, COCOeval, Hierarchy

label_to_id = {cat["name"]: cat["id"] for cat in coco_gt.dataset["categories"]}
h = Hierarchy.from_file("bbox_labels_600_hierarchy.json", label_to_id=label_to_id)

ev = COCOeval(coco_gt, coco_dt, "bbox", oid_style=True, hierarchy=h)
ev.run()
```

See the [Open Images evaluation](../guide/evaluation.md#open-images-evaluation) guide for a full walkthrough.

---

## Constructors

### `from_file`

```python
Hierarchy.from_file(path: str, label_to_id: dict[str, int] | None = None) -> Hierarchy
```

Parse an Open Images hierarchy JSON file (`bbox_labels_600_hierarchy.json`).

The JSON uses nested `LabelName` / `Subcategory` fields. `label_to_id` maps OID label strings (e.g. `"/m/0jbk"`) to category IDs. Labels not present in `label_to_id` get virtual node IDs that won't appear in your dataset's category list.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Path to the OID hierarchy JSON file |
| `label_to_id` | <code>dict &#124; None</code> | `None` | Maps OID label strings to category IDs; `None` assigns virtual IDs to all labels |

```python
label_to_id = {cat["name"]: cat["id"] for cat in coco_gt.dataset["categories"]}
h = Hierarchy.from_file("bbox_labels_600_hierarchy.json", label_to_id=label_to_id)
```

---

### `from_dict`

```python
Hierarchy.from_dict(
    tree_dict: dict,
    label_to_id: dict[str, int] | None = None,
) -> Hierarchy
```

Build a hierarchy from a Python dict in the OID JSON format (`LabelName` / `Subcategory` keys). Useful when you already have the hierarchy loaded as a dict.

```python
import json

with open("bbox_labels_600_hierarchy.json") as f:
    tree = json.load(f)

h = Hierarchy.from_dict(tree, label_to_id=label_to_id)
```

---

### `from_parent_map`

```python
Hierarchy.from_parent_map(parent_map: dict[int, int]) -> Hierarchy
```

Build a hierarchy from an explicit `{child_id: parent_id}` mapping. Useful when constructing a hierarchy programmatically or from a database.

```python
h = Hierarchy.from_parent_map({
    3: 1,   # cat 3's parent is cat 1
    4: 1,   # cat 4's parent is cat 1
    5: 2,   # cat 5's parent is cat 2
})
```

---

### `from_categories`

```python
Hierarchy.from_categories(categories: list[dict]) -> Hierarchy
```

Derive a hierarchy from `supercategory` fields in a category list. For each category with a `supercategory` value, looks up the matching category by name. If no matching category is found, a virtual node is created.

Handy for COCO-format datasets that already encode the hierarchy via `supercategory`:

```python
h = Hierarchy.from_categories(coco_gt.dataset["categories"])
```

Self-referencing supercategories (where `name == supercategory`) are silently skipped.

---

## Methods

### `ancestors`

```python
ancestors(cat_id: int) -> list[int]
```

Return the ancestor chain for a category, from self up to the root: `[cat_id, parent_id, grandparent_id, ...]`.

Returns `[cat_id]` if the category has no parent, or `[]` if the category is unknown to the hierarchy.

```python
h = Hierarchy.from_parent_map({3: 1, 1: 0})
h.ancestors(3)   # [3, 1, 0]
h.ancestors(1)   # [1, 0]
h.ancestors(0)   # [0]
h.ancestors(99)  # []  — unknown category
```

---

### `children`

```python
children(cat_id: int) -> list[int]
```

Return the direct children of a category, or `[]` if the category has no children.

```python
h.children(1)   # [3, 4]
h.children(3)   # []
```

---

### `parent`

```python
parent(cat_id: int) -> int | None
```

Return the parent of a category, or `None` if it is a root node.

```python
h.parent(3)   # 1
h.parent(0)   # None
```
