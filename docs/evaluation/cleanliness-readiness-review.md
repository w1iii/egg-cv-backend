# Cleanliness model: evidence and readiness review

## Verdict

The model is suitable for a **supervised thesis demonstration** as an
experimental prediction, with a conspicuous beta label and human verification.
The current interface already displays it this way. It is **not yet supported
for automatic acceptance or rejection**, operational sorting, or an accuracy
claim about the intended physical setup. Keep
`CLEANLINESS_RELEASE_APPROVED=false` until independently validated and signed
off. Display is not proof of safety or correctness; users must not treat the
displayed percentage as a calibrated probability of being correct.

## Model and implementation

- `models/egg_grader_cleanliness_v1.pth`: ResNet-18, 224 x 224 input,
  classes `Clean` and `Stained`. Current SHA-256
  `b00884a1f6b03b87e2f8b3378081cbadbbdab9e68218de082d3c624e3c866a6f`
  matches the checkpoint in the research report and earlier benchmark.
- The pipeline uses detector bounding boxes, zero extra cleanliness padding,
  OpenCV BGR-to-RGB conversion, square resize, and ImageNet normalization.
  Crops below 2,500 pixels in area or invalid crops are refused. Input-quality
  warnings can independently send the egg to review.
- In the running application the explicit stained threshold is **0.60** and
  clean confidence review threshold is **0.75**. Therefore `P(stained) >=
  0.60` returns Stained; `0.25 < P(stained) < 0.60` returns Needs Review;
  otherwise Clean. A Stained result is not downgraded to review for confidence.
  This policy is implemented in `app/ml/cleanliness_inference.py` and supplied
  by `app/ml/model_registry.py`. The 0.60 value overrides any threshold stored
  in the checkpoint.
- `confidence` is the selected class score, not demonstrated field calibration.
  It is displayed as a percentage. It must not be interpreted as a guarantee.
- With the gate off, cleanliness cannot by itself accept or reject an egg;
  undamaged eggs generally require review. A reliable damage result can still
  independently reject an egg. See `app/services/grading_service.py`.

## Evidence assessed

The research freeze report lists 425 training crops, 92 validation crops and
91 test crops, with 68, 40 and 40 source images respectively. It reports no
cross-split overlap among checked filename, source-group, source-image and hash
columns, but explicitly says physical-source independence is not established.
The freeze's `release_ready` describes the manifest checks, not approval for
deployment. Source audit includes stock imagery, including vector-like images;
there is no evidence this collection represents the intended camera station.

The research evaluation selected a **0.04** stained threshold on the validation
set using a stated 98% stained-recall target. The application instead uses
0.60 to reduce artifact-driven false positives. These are different operating
points; the research threshold metrics do not describe application behavior.
The application threshold has not been validated on independent station data.

## Re-run on available test crops

Ran `scripts/benchmark.py classifier` using the actual application classifier,
configured thresholds and current checkpoint against the research test
manifest. Raw reproducibility record:
[`benchmarks/cleanliness_local_review.json`](benchmarks/cleanliness_local_review.json).
The checkout was dirty and its commit hash therefore does not identify all code
changes; the classifier code itself was reviewed. The full manifest now has
91 readable crops (49 stained, 42 clean). Its SHA-256 is recorded in the JSON.
An earlier archived benchmark used a different manifest hash and 82 readable
crops; these runs must not be treated as a paired before/after comparison.

| Actual label | Predicted stained | Predicted clean | Needs review |
|---|---:|---:|---:|
| Stained | 44 | 3 | 2 |
| Clean | 1 | 41 | 0 |

At the application's 0.60/0.75 policy, 44/49 stained crops were directly
flagged (89.8% including reviews in the denominator); 3/49 were called clean
and 2/49 were sent to review. Among decided stained cases, recall is 44/47
(93.6%); 41/42 clean crops were called clean (97.6% specificity). Overall
accuracy on decided crops is 85/89 (95.5%); 2/91 crops were reviewed (2.2%).
The two review cases are not included in decided-only metrics. Multiple eggs
can come from one source image, so these observations are not necessarily
independent. The small test set and shared source domain limit inference.

For reference only, raw threshold 0.04 gives 48 true positives, 1 false
negative, 41 true negatives and 1 false positive on these same 91 crops. It
does not include the application's review band. Do not select or tune a
threshold on this test set; doing so would make the test optimistic.

## Important limitations and risks

1. No independent labeled, full-image test from the intended camera, egg
   population, lighting and physical setup was found. Crop-only metrics do not
   measure detector misses, incorrect boxes, overlap or downstream decisions.
2. Web/stock imagery and grouped crops can create domain shift or correlations;
   recorded split checks do not prove physical independence. Nine previously
   unavailable images are now present, but no new physical validation follows.
3. The label `Stained` is a visual category, not a microbiological safety test.
   One image cannot establish cleanliness on hidden shell surfaces.
4. Application review rules and sample distribution affect reported metrics.
   Confidence values and subgroup performance have not been validated in the
   target setting. Poor lighting, glare, shell color and stain severity may
   affect errors. Presenting uncalibrated confidence can encourage overtrust.
5. Existing examples without expert labels cannot demonstrate accuracy. Prior
   code accepted stain outputs as final rejections; the current gate prevents
   that unsupported claim.

## Required work before approval

1. Define the intended use, visible-shell labeling rules, responsibility for
   inspection and unacceptable false-clean/false-reject rates before testing.
2. Capture independent full images with the actual camera, lighting, working
   distance, backgrounds and marker; include clean/stained eggs, stain severity,
   shell colors, glare, overlap and representative operating variation. Track
   physical egg, batch, session and capture source to prevent leakage.
3. Obtain independent expert labels for visible shell and bounding boxes;
   record disagreements and adjudicate without seeing model outputs. Avoid
   claiming anything about unseen surfaces.
4. Separate development, threshold-selection and final held-out test groups by
   egg and capture session. Freeze the final manifest and image hashes.
5. Run the full pipeline on untouched images. Report detector misses and false
   positives, stain confusion matrix, false-clean rate, review rate, performance
   by relevant group and uncertainty intervals respecting grouped samples.
   Compare results against the predeclared acceptance criteria. Retest after
   changes to checkpoints, thresholds, camera or environment.
6. Document human review workflow and obtain explicit sign-off. Only then set
   the release gate; do not treat the beta label alone as a safety control.

Sources: `benchmark-report.md`, `baseline.md`, research repository
`reports/cleanliness_manifest_freeze.json`, `reports/cleanliness_source_audit.json`,
`reports/cleanliness_crop_resnet18_v1_eval.json`, current application code and
the new raw benchmark. No new physical data was collected for this review.
