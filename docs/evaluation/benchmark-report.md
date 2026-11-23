# F-06 / F-12 Measurement report

Generated 2026-11-23 with `scripts/benchmark.py`. Raw reports (JSON and
Markdown, with manifest hashes, code revision, thresholds and checkpoint
hashes) are in [`benchmarks/`](benchmarks/). "Before" = `main` @ `225d5e1`;
"after" = this branch, same inputs (manifest hashes match; the harness refuses
to compare otherwise).

## Verdict per change

| Component | Before vs after | Evidence |
|---|---|---|
| Detector + post-processing | **No effect** on any metric | `detector_before.json` / `detector_after.json` |
| Cleanliness classifier path | **No effect** on any metric | `cleanliness_before.json` / `cleanliness_after.json` |
| Damage classifier path | **Not measured** (see below) | — |
| Per-egg classifier inputs on real photos | **Identical** (36/36 eggs: boxes, statuses, confidences) | `baseline.md` |
| Decision layer | Changed by design (policy), not an accuracy change | `baseline.md`, `tests/test_grading_policy.py` |
| End-to-end decision accuracy | **Not measured**: no labeled full-image set exists | — |
| Sizing | **Not measured**: no physical ground truth exists | — |

No model-accuracy improvement is claimed. Latency differences between runs
are within run-to-run noise on this CPU and are not interpreted.

## Detector (19 images, research repo `detection_finetune` test labels)

Images: `data/detection/images/test`; labels:
`data/detection_finetune/labels/test`; confidence 0.75; match IoU >= 0.5.

| Metric | Before | After |
|---|---|---|
| Ground-truth eggs | 57 | 57 |
| TP / FP / FN | 57 / 3 / 0 | 57 / 3 / 0 |
| Precision / Recall / F1 | 0.950 / 1.000 / 0.974 | identical |
| Exact per-image count accuracy | 0.895 | 0.895 |

The exact-count accuracy (0.895) and false positives per image (3/19 = 0.158)
equal the research repo's `reports/detector_finetune_test_report.json`, so
the app's detector path reproduces the research evaluation. Both false
positives occur in two images (`brown2…`, `white2…`). This test split comes
from the same source collection as training; it is not an independent field
set.

## Cleanliness classifier (research repo cleanliness test crops)

Manifest `data/cleanliness_candidates/manifests/test.csv`: 91 rows, **9
image files missing from disk** (they are staged for deletion in the research
repo's working tree), so 82 crops were evaluated.

| View | Precision (Stained) | Recall (Stained) | Specificity | Balanced acc. | Review rate |
|---|---|---|---|---|---|
| App policy (decision 0.60, review 0.75) | 0.974 | 0.927 | 0.975 | 0.951 | 1.2% (1/82) |
| Raw P(Stained) >= 0.04 (research threshold) | 0.976 | 0.976 | 0.975 | 0.976 | — |

At 0.04 the errors are exactly 1 false positive and 1 false negative, the
same as the research report (`cleanliness_crop_resnet18_v1_eval.json`, 91
crops: tn 41, fp 1, fn 1, tp 48); the 9 missing crops account for the
difference in totals. This supports that the app's crop preprocessing matches
training. These are web/stock crops from the same sourcing as training, so
they say nothing about field performance, and the model remains **not
release-approved**.

## Damage classifier: why there is no number

The research reports evaluate the damage model on detector crops
(`data/damage_crops/test/manifest.csv`), which are not present on this
machine. `data/damage/manifests_v2/test.csv` lists whole source images, which
are not valid input for a crop classifier, so results on it were discarded
rather than reported. To measure it, restore `data/damage_crops/` and run:

```bash
python scripts/benchmark.py classifier --task damage \
  --manifest <research>/data/damage_crops/test/manifest.csv --data-root <research> \
  --reference-threshold 0.18 --out docs/evaluation/benchmarks/damage_after.json
```

## End-to-end and sizing: protocol (not yet run)

Neither can be measured without new data; nothing here is estimated.

1. Freeze a set of full images that were never used for training, captured
   with the intended camera, lighting and ArUco marker (with a white margin of
   at least one marker cell).
2. For every egg, record the box, `damage_label`, `cleanliness_label`, and
   (for sizing) caliper `true_length_mm` / `true_breadth_mm`; add `group_*`
   columns (lighting, overlap, pigmentation, source, visibility). Template:
   [`end_to_end_manifest_template.csv`](end_to_end_manifest_template.csv).
3. Run `python scripts/benchmark.py end-to-end --manifest <csv> --out <json>`.
   It reports the decision confusion matrix, review rate, Reject recall
   (counting review as a miss), Acceptable precision, detection misses and
   false positives, latency, per-group breakdowns, invalid-calibration rate,
   length/breadth MAE, RMSE and % error, and the size confusion matrix.
4. Keep `ARUCO_GRADES_APPROVED=false` and weight disabled until that report
   exists and its sizing error is accepted. Never mix operational uploads into
   this set.
