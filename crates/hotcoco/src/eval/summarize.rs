use std::collections::HashMap;

use crate::params::{IouType, Params};

use super::types::AccumulatedEval;
use super::COCOeval;

/// Definition of a single summary metric (one row in the COCO output table).
pub(super) struct MetricDef {
    /// true = Average Precision, false = Average Recall.
    pub ap: bool,
    /// Specific IoU threshold, or None to average over all thresholds.
    pub iou_thr: Option<f64>,
    /// Area range label to filter by (e.g. "all", "small", "medium", "large").
    pub area_lbl: &'static str,
    /// Maximum detections per image for this metric.
    pub max_det: usize,
}

pub(super) fn metrics_bbox_segm(max_d: usize, max_d_s: usize, max_d_m: usize) -> Vec<MetricDef> {
    vec![
        MetricDef {
            ap: true,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d,
        },
        MetricDef {
            ap: true,
            iou_thr: Some(0.5),
            area_lbl: "all",
            max_det: max_d,
        },
        MetricDef {
            ap: true,
            iou_thr: Some(0.75),
            area_lbl: "all",
            max_det: max_d,
        },
        MetricDef {
            ap: true,
            iou_thr: None,
            area_lbl: "small",
            max_det: max_d,
        },
        MetricDef {
            ap: true,
            iou_thr: None,
            area_lbl: "medium",
            max_det: max_d,
        },
        MetricDef {
            ap: true,
            iou_thr: None,
            area_lbl: "large",
            max_det: max_d,
        },
        MetricDef {
            ap: false,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d_s,
        },
        MetricDef {
            ap: false,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d_m,
        },
        MetricDef {
            ap: false,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d,
        },
        MetricDef {
            ap: false,
            iou_thr: None,
            area_lbl: "small",
            max_det: max_d,
        },
        MetricDef {
            ap: false,
            iou_thr: None,
            area_lbl: "medium",
            max_det: max_d,
        },
        MetricDef {
            ap: false,
            iou_thr: None,
            area_lbl: "large",
            max_det: max_d,
        },
    ]
}

pub(super) fn metrics_kp(max_d: usize) -> Vec<MetricDef> {
    vec![
        MetricDef {
            ap: true,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d,
        },
        MetricDef {
            ap: true,
            iou_thr: Some(0.5),
            area_lbl: "all",
            max_det: max_d,
        },
        MetricDef {
            ap: true,
            iou_thr: Some(0.75),
            area_lbl: "all",
            max_det: max_d,
        },
        MetricDef {
            ap: true,
            iou_thr: None,
            area_lbl: "medium",
            max_det: max_d,
        },
        MetricDef {
            ap: true,
            iou_thr: None,
            area_lbl: "large",
            max_det: max_d,
        },
        MetricDef {
            ap: false,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d,
        },
        MetricDef {
            ap: false,
            iou_thr: Some(0.5),
            area_lbl: "all",
            max_det: max_d,
        },
        MetricDef {
            ap: false,
            iou_thr: Some(0.75),
            area_lbl: "all",
            max_det: max_d,
        },
        MetricDef {
            ap: false,
            iou_thr: None,
            area_lbl: "medium",
            max_det: max_d,
        },
        MetricDef {
            ap: false,
            iou_thr: None,
            area_lbl: "large",
            max_det: max_d,
        },
    ]
}

impl COCOeval {
    /// Print the standard 12-line COCO evaluation summary.
    pub fn summarize(&mut self) {
        let eval = match &self.eval {
            Some(e) => e,
            None => {
                eprintln!("Please run evaluate() and accumulate() first.");
                return;
            }
        };

        let is_kp = self.params.iou_type == IouType::Keypoints;

        // Warn if parameters differ from what the hardcoded summary display expects.
        let defaults = Params::new(self.params.iou_type);
        let mut warnings = Vec::new();

        let default_iou: Vec<f64> = (0..10).map(|i| 0.5 + 0.05 * i as f64).collect();
        if self.params.iou_thrs != default_iou {
            warnings.push(
                "iou_thrs differ from default (0.50:0.05:0.95). AP50/AP75 lines may show -1.000."
                    .to_string(),
            );
        }
        let expected_max_dets = if self.is_lvis {
            vec![300usize]
        } else {
            defaults.max_dets.clone()
        };
        if self.params.max_dets != expected_max_dets {
            warnings.push(format!(
                "max_dets differ from expected ({:?}). AR lines may use unexpected max_dets values.",
                expected_max_dets
            ));
        }
        if self.params.area_rng_lbl != defaults.area_rng_lbl {
            warnings.push(format!(
                "area_rng_lbl differ from default ({:?}). Per-size metrics may fall back to index 0.",
                defaults.area_rng_lbl
            ));
        }

        for w in &warnings {
            eprintln!("Warning: {}", w);
        }

        // Compute a single summary statistic by averaging over the relevant slice
        // of the precision or recall array. Returns -1.0 if no valid data exists.
        let summarize_stat =
            |ap: bool, iou_thr: Option<f64>, area_lbl: &str, max_det: usize| -> f64 {
                let a_idx = self
                    .params
                    .area_rng_lbl
                    .iter()
                    .position(|l| l == area_lbl)
                    .unwrap_or(0);
                let m_idx = self
                    .params
                    .max_dets
                    .iter()
                    .position(|&d| d == max_det)
                    .unwrap_or(0);

                let t_indices: Vec<usize> = if let Some(thr) = iou_thr {
                    self.params
                        .iou_thrs
                        .iter()
                        .enumerate()
                        .filter(|(_, &t)| (t - thr).abs() < 1e-9)
                        .map(|(i, _)| i)
                        .collect()
                } else {
                    (0..eval.shape.t).collect()
                };

                let mut vals = Vec::new();
                for &t_idx in &t_indices {
                    for k_idx in 0..eval.shape.k {
                        if ap {
                            for r_idx in 0..eval.shape.r {
                                let idx = eval.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                                let v = eval.precision[idx];
                                if v >= 0.0 {
                                    vals.push(v);
                                }
                            }
                        } else {
                            let idx = eval.recall_idx(t_idx, k_idx, a_idx, m_idx);
                            let v = eval.recall[idx];
                            if v >= 0.0 {
                                vals.push(v);
                            }
                        }
                    }
                }

                if vals.is_empty() {
                    -1.0
                } else {
                    vals.iter().sum::<f64>() / vals.len() as f64
                }
            };

        let max_det_default = *self.params.max_dets.last().unwrap_or(&100);
        let max_det_small = if self.params.max_dets.len() >= 3 {
            self.params.max_dets[0]
        } else {
            max_det_default
        };
        let max_det_med = if self.params.max_dets.len() >= 3 {
            self.params.max_dets[1]
        } else {
            max_det_default
        };

        if self.is_lvis {
            // LVIS summarize: 13 metrics with max_dets=300.
            // APr/APc/APf are computed as mean per-category AP within each freq group.
            let per_cat_ap = self.per_cat_ap(eval);

            let freq_group_ap = |indices: &[usize]| -> f64 {
                if indices.is_empty() {
                    return -1.0;
                }
                let valid: Vec<f64> = indices
                    .iter()
                    .filter_map(|&k| {
                        let v = per_cat_ap[k];
                        if v >= 0.0 {
                            Some(v)
                        } else {
                            None
                        }
                    })
                    .collect();
                if valid.is_empty() {
                    -1.0
                } else {
                    valid.iter().sum::<f64>() / valid.len() as f64
                }
            };

            let ap = summarize_stat(true, None, "all", max_det_default);
            let ap50 = summarize_stat(true, Some(0.5), "all", max_det_default);
            let ap75 = summarize_stat(true, Some(0.75), "all", max_det_default);
            let aps = summarize_stat(true, None, "small", max_det_default);
            let apm = summarize_stat(true, None, "medium", max_det_default);
            let apl = summarize_stat(true, None, "large", max_det_default);
            let ap_r = freq_group_ap(&self.freq_groups[0]);
            let ap_c = freq_group_ap(&self.freq_groups[1]);
            let ap_f = freq_group_ap(&self.freq_groups[2]);
            let ar = summarize_stat(false, None, "all", max_det_default);
            let ar_s = summarize_stat(false, None, "small", max_det_default);
            let ar_m = summarize_stat(false, None, "medium", max_det_default);
            let ar_l = summarize_stat(false, None, "large", max_det_default);

            let lvis_metrics: &[(&str, f64)] = &[
                ("AP", ap),
                ("AP50", ap50),
                ("AP75", ap75),
                ("APs", aps),
                ("APm", apm),
                ("APl", apl),
                ("APr", ap_r),
                ("APc", ap_c),
                ("APf", ap_f),
                ("AR@300", ar),
                ("ARs@300", ar_s),
                ("ARm@300", ar_m),
                ("ARl@300", ar_l),
            ];

            let mut stats = Vec::with_capacity(lvis_metrics.len());
            for (name, val) in lvis_metrics {
                stats.push(*val);
                let val_str = Self::format_metric(*val);
                println!(" {:>10} = {}", name, val_str);
            }
            self.stats = Some(stats);
            return;
        }

        let metrics = if is_kp {
            metrics_kp(max_det_default)
        } else {
            metrics_bbox_segm(max_det_default, max_det_small, max_det_med)
        };

        let iou_type_str = self.params.iou_type.to_string();

        let mut stats = Vec::with_capacity(metrics.len());

        for m in &metrics {
            let val = summarize_stat(m.ap, m.iou_thr, m.area_lbl, m.max_det);
            stats.push(val);

            let metric_name = if m.ap {
                "Average Precision"
            } else {
                "Average Recall"
            };
            let metric_short = if m.ap { "AP" } else { "AR" };

            let iou_str = match m.iou_thr {
                Some(thr) => format!("{:.2}", thr),
                None => "0.50:0.95".to_string(),
            };

            let area_str = m.area_lbl;
            let det_str = m.max_det;

            let val_str = Self::format_metric(val);

            println!(
                " {:<18} @[ IoU={:<9} | area={:>6} | maxDets={:>3} ] = {}",
                format!("{} ({})", metric_name, metric_short),
                iou_str,
                area_str,
                det_str,
                val_str
            );
        }

        println!("Eval type: {}", iou_type_str);
        self.stats = Some(stats);
    }

    /// Format a metric value: -1.0 sentinel stays as "-1.000", positive values use 3 decimal places.
    pub(super) fn format_metric(val: f64) -> String {
        if val < 0.0 {
            format!("{:0.3}", -1.0f64)
        } else {
            format!("{:0.3}", val)
        }
    }

    /// Index of the "all" area range label, or 0 if not found.
    fn area_all_idx(&self) -> usize {
        self.params
            .area_rng_lbl
            .iter()
            .position(|l| l == "all")
            .unwrap_or(0)
    }

    /// Metric key names for the current evaluation mode.
    fn metric_keys(&self) -> &[&str] {
        if self.is_lvis {
            &[
                "AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf", "AR@300",
                "ARs@300", "ARm@300", "ARl@300",
            ]
        } else if self.params.iou_type == IouType::Keypoints {
            &[
                "AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl",
            ]
        } else {
            &[
                "AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm",
                "ARl",
            ]
        }
    }

    /// Per-category mean AP (averaged over all IoU thresholds and recall thresholds,
    /// at area="all" and the last max_dets setting). Returns one value per `params.cat_ids`
    /// entry; -1.0 for categories with no valid precision data.
    pub(super) fn per_cat_ap(&self, eval: &AccumulatedEval) -> Vec<f64> {
        let a_idx = self.area_all_idx();
        let m_idx = eval.shape.m - 1;
        (0..eval.shape.k)
            .map(|k_idx| {
                let mut sum = 0.0;
                let mut count = 0_usize;
                for t_idx in 0..eval.shape.t {
                    for r_idx in 0..eval.shape.r {
                        let idx = eval.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                        let v = eval.precision[idx];
                        if v >= 0.0 {
                            sum += v;
                            count += 1;
                        }
                    }
                }
                if count == 0 {
                    -1.0
                } else {
                    sum / count as f64
                }
            })
            .collect()
    }

    /// Run the full evaluation pipeline in one call: `evaluate` → `accumulate` → `summarize`.
    ///
    /// Equivalent to calling the three methods in sequence. Primarily used with LVIS
    /// pipelines (e.g. Detectron2 / MMDetection) that expect a single `run()` entry point.
    pub fn run(&mut self) {
        self.evaluate();
        self.accumulate();
        self.summarize();
    }

    /// Return summary metrics as a `HashMap<metric_name, value>`.
    ///
    /// Must be called after [`summarize`](COCOeval::summarize). Returns an empty map
    /// if `summarize` has not been run.
    ///
    /// # Arguments
    ///
    /// * `prefix` — When `Some("val/bbox")`, keys become `"val/bbox/AP"` etc.
    ///   When `None`, keys are bare metric names (`"AP"`, `"AR100"`, …).
    /// * `per_class` — When `true` and [`accumulate`](COCOeval::accumulate) has been
    ///   run, adds per-category AP entries keyed as `"AP/{cat_name}"` (or
    ///   `"{prefix}/AP/{cat_name}"` with a prefix). Categories where all precision
    ///   values are −1 are skipped.
    ///
    /// # Metric keys
    ///
    /// For LVIS mode: `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`, `APr`, `APc`, `APf`,
    /// `AR@300`, `ARs@300`, `ARm@300`, `ARl@300`.
    ///
    /// For standard COCO bbox/segm: `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`,
    /// `AR1`, `AR10`, `AR100`, `ARs`, `ARm`, `ARl`.
    ///
    /// For keypoints: `AP`, `AP50`, `AP75`, `APm`, `APl`,
    /// `AR`, `AR50`, `AR75`, `ARm`, `ARl`.
    pub fn get_results(&self, prefix: Option<&str>, per_class: bool) -> HashMap<String, f64> {
        let stats = match &self.stats {
            Some(s) => s,
            None => return HashMap::new(),
        };

        let keys = self.metric_keys();

        let make_key = |metric: &str| -> String {
            match prefix {
                Some(p) => format!("{p}/{metric}"),
                None => metric.to_string(),
            }
        };

        let mut results: HashMap<String, f64> = keys
            .iter()
            .zip(stats.iter())
            .map(|(&k, &v)| (make_key(k), v))
            .collect();

        if per_class {
            if let Some(eval) = &self.eval {
                let per_cat = self.per_cat_ap(eval);
                for (ap, cat_id) in per_cat.iter().zip(self.params.cat_ids.iter()) {
                    if *ap >= 0.0 {
                        let cats = self.coco_gt.load_cats(&[*cat_id]);
                        if let Some(cat) = cats.first() {
                            results.insert(make_key(&format!("AP/{}", cat.name)), *ap);
                        }
                    }
                }
            }
        }

        results
    }

    /// Compute F-beta scores after `accumulate()`.
    ///
    /// Returns three metrics analogous to AP/AP50/AP75, but using max F-beta instead of
    /// mean precision. For each (IoU threshold, category), finds the recall operating point
    /// that maximises F-beta, then averages across categories.
    ///
    /// `beta` controls the precision/recall trade-off:
    /// - `beta = 1.0`  → F1 (harmonic mean, equal weight)
    /// - `beta < 1.0`  → weights precision more heavily
    /// - `beta > 1.0`  → weights recall more heavily
    ///
    /// Returns an empty map if `accumulate()` has not been run.
    pub fn f_scores(&self, beta: f64) -> HashMap<String, f64> {
        let eval = match &self.eval {
            Some(e) => e,
            None => return HashMap::new(),
        };

        let beta2 = beta * beta;
        let a_idx = self.area_all_idx();
        let m_idx = eval.shape.m - 1;

        // Identify which IoU threshold indices correspond to 0.5 and 0.75.
        let mut is_t50 = vec![false; eval.shape.t];
        let mut is_t75 = vec![false; eval.shape.t];
        for (i, &thr) in self.params.iou_thrs.iter().enumerate() {
            if (thr - 0.5).abs() < 1e-9 {
                is_t50[i] = true;
            }
            if (thr - 0.75).abs() < 1e-9 {
                is_t75[i] = true;
            }
        }

        // Single pass: compute max-F-beta per (t_idx, k_idx), accumulate into three buckets.
        let mut sum_all = 0.0_f64;
        let mut count_all = 0_usize;
        let mut sum_50 = 0.0_f64;
        let mut count_50 = 0_usize;
        let mut sum_75 = 0.0_f64;
        let mut count_75 = 0_usize;

        for t_idx in 0..eval.shape.t {
            for k_idx in 0..eval.shape.k {
                let max_f = self
                    .params
                    .rec_thrs
                    .iter()
                    .enumerate()
                    .filter_map(|(r_idx, &r)| {
                        let p_idx = eval.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                        let p = eval.precision[p_idx];
                        if p < 0.0 {
                            return None;
                        }
                        let denom = beta2 * p + r;
                        if denom < f64::EPSILON {
                            return Some(0.0);
                        }
                        Some((1.0 + beta2) * p * r / denom)
                    })
                    .fold(f64::NEG_INFINITY, f64::max);

                if max_f > f64::NEG_INFINITY {
                    sum_all += max_f;
                    count_all += 1;
                    if is_t50[t_idx] {
                        sum_50 += max_f;
                        count_50 += 1;
                    }
                    if is_t75[t_idx] {
                        sum_75 += max_f;
                        count_75 += 1;
                    }
                }
            }
        }

        let mean_or_neg1 = |sum: f64, count: usize| -> f64 {
            if count == 0 {
                -1.0
            } else {
                sum / count as f64
            }
        };

        let prefix = if (beta - 1.0).abs() < 1e-9 {
            "F1".to_string()
        } else {
            format!("F{:.1}", beta)
        };

        let mut out = HashMap::new();
        out.insert(prefix.clone(), mean_or_neg1(sum_all, count_all));
        out.insert(format!("{}50", prefix), mean_or_neg1(sum_50, count_50));
        out.insert(format!("{}75", prefix), mean_or_neg1(sum_75, count_75));
        out
    }

    /// Print results to stdout in a compact key=value format.
    ///
    /// Must be called after [`summarize`](COCOeval::summarize). Prints nothing if
    /// `summarize` has not been run (emits a warning to stderr instead).
    pub fn print_results(&self) {
        let results = self.get_results(None, false);
        if results.is_empty() {
            eprintln!("No results to print. Run evaluate(), accumulate(), and summarize() first.");
            return;
        }

        let keys = self.metric_keys();

        for key in keys {
            let val = results.get(*key).copied().unwrap_or(-1.0);
            let val_str = Self::format_metric(val);
            println!(" {:>10} = {}", key, val_str);
        }
    }
}
