# Bug report — `training.ipynb`

Audit of the 100 code cells in `training.ipynb` (Phases 1–9), with the fix for
each. Every "measured" figure below was reproduced by running the original code
against the real dataset.

## Status

**All nine phases are fixed in the notebook.** The original 101 cells are
replaced by 39 that drive the tested library in `src/brats/`.

| | Before | After |
|---|---|---|
| Cells | 101 | 39 |
| Whole notebook, fast mode | 35+ min, never completed | **89 s** |
| Phases 1–8, full cohort | — | 7–9 min |
| Phase 9 training, 4 models | ~123 h | ~45 min |

One switch at the top controls the speed/accuracy trade-off:

```python
RUN_MODE = "fast"   # representative subset, ~2 min end to end — for developing
RUN_MODE = "full"   # whole cohort — the configuration to report
```

Backups: `training.before_phase1_8_fix.ipynb` (Phase 9 fixed, 1–8 original) and
`training.ipynb.bak` (entirely original).

Every phase imports from `src/brats/`, so any phase can be run after a kernel
restart without depending on globals left behind by earlier cells.

Severity:
- **Critical** — produces wrong results or makes the code unrunnable
- **High** — corrupts data, invalidates a reported number, or costs orders of magnitude of runtime
- **Medium** — wrong in a reachable case, or silently hides a problem
- **Low** — correctness-neutral: duplication, dead code, deprecation

---

## Critical

### C1. Reported validation Dice was inflated by empty-class credit
`Cell 90`, `batch_mean_foreground_dice`

```python
d = (2 * inter + eps) / (pc.sum() + tc.sum() + eps)   # 1e-6/1e-6 = 1.0
```

When a class is absent from **both** prediction and target, this returns `1.0`.
On BraTS most slices contain no enhancing tumour, so a model predicting that
class correctly-as-nothing collected a free perfect score on nearly every batch.
The same formula is used in `dice_binary` (Cell 73).

Measured: a model predicting **pure background** scores **0.667** under this
formula, versus **0.0** correctly.

```
per-class dice, all-background prediction:
  old (absent = 1.0): [0.0, 1.0, 1.0] -> mean 0.6667
  fixed (absent = NaN, excluded): [0.0, nan, nan] -> mean 0.0000
```

This affects the saved results: `results/metrics/phase9/phase9_model_comparison.csv`
reports val_dice 0.63–0.75, which is not comparable to a correctly computed Dice.

**Fix** — `brats/metrics.py`: an undefined Dice returns `NaN` and is excluded
from means (the BraTS convention). `per_class_dice_torch(..., ignore_absent=True)`,
`mean_foreground_dice_torch`, and `nanmean` for aggregation.

### C2. `hd95_binary` did not compute HD95
`Cell 73`

Three independent errors: it max-projected the 3D volume to 2D, measured
distances from *every voxel* of each mask rather than from its surface, and
ignored voxel spacing so the result had no unit. The number reported as HD95 was
none of those things.

**Fix** — `brats/metrics.py::hd95`: 95th percentile of the symmetric
*surface-to-surface* distance in 3D, in millimetres, using the spacing from the
NIfTI header. Verified: a sphere shifted by 3 voxels gives exactly 3.000 mm.

### C3. Training reloaded four full volumes per slice
`Cell 85`, `BraTSPhase9SliceDataset` + `create_dataloaders`

The dataset cached exactly one patient (`self._cache_pid`) and was handed to
`DataLoader(..., shuffle=True)`. Shuffling means consecutive indices come from
different patients, so the cache missed on essentially **every sample**: each
`__getitem__` loaded and percentile-normalised four 240×240×155 volumes to
return one 128×128 slice.

Someone already worked around this outside the notebook — `results/cache/phase9_pilot_128/`
holds pre-materialised `.npy` arrays — but the notebook still contains the
broken version.

**Fix** — `brats/data.py`: two correct strategies.
`build_slice_cache` + `CachedSliceDataset` preprocesses each patient once into
memory-mapped arrays (shuffling then costs nothing); `LazySliceDataset` +
`PatientBlockSampler` keeps the on-the-fly path but shuffles whole patients and
slices within them, so the one-patient cache actually hits.

Measured on 60 slices from 9 patients:

```
with PatientBlockSampler : 9 volume loads, 51 cache hits
with plain shuffle=True  : 54 volume loads
```

9 loads is the floor (one per patient).

**Cost of the original Phase 9 as configured** (`SUBSET_NUM_PATIENTS =
len(patient_folders) // 2` = 184 patients, 128 of them training, ~61 tumour
slices each, 0.47 s per patient load measured on this machine):

| | per epoch | 30 epochs x 4 models |
|---|---|---|
| Original (reload per slice) | ~7,829 volume loads, **~1.0 hour** | **~123 hours (5 days)** |
| Cached (measured, extrapolated) | ~0 loads, **~26 s** | **~53 min** + 1.6 min cache build |

Benchmarked directly on 1,982 slices from 30 patients: 18.6 s to build the
cache, then 6.7 s per training epoch. That is the difference between a run
that cannot finish and one that finishes over lunch.

### C3b. The first training cell absorbed ~30 s of hidden cuDNN autotune

After the caching fix, cell 12 (*Train U-Net*) was still conspicuously slower
than cells 13-15, which train the same way. Profiling found it was not training
at all for the first half-minute:

```
import torch                      :   1.41s
CUDA context init                 :   0.47s
  train step 1                    :  15.877s   <-- cuDNN algorithm selection
  train step 2                    :   0.072s
  train step 6                    :   0.019s
```

cuDNN autotunes **per input shape**, and the cost recurs for each new one:

```
first step  batch=2  : 16.964s   <- all the old Cell 9 warmed
first step  batch=32 :  6.845s   <- Cell 12 pays again: new shape
odd batch   batch=17 :  6.174s   <- the ragged final batch: a third shape
second model, batch=32: 2.883s   <- Cells 13-15 start warm
```

So cell 12 paid ~30 s of autotune (cold start + training shape + ragged final
batch) with no output at all, looking exactly like a hang — while cells 13-15
inherited a warm cache and appeared much faster for no real reason.

**Fix** — three parts:

1. `training.warmup()` pays the cost once, in Cell 9, for every architecture at
   every shape training will use, with a progress line per architecture.
2. `drop_last=True` on the training loader keeps every training batch one
   shape, so the ragged-batch autotune never happens.
3. `progress_every=25` prints in-epoch progress with a rate and ETA, so a long
   epoch is visibly alive.

Measured after the fix (30% cohort, 2,000 slices/split, 3 epochs):

```
cell 12 (first training cell): 19.9s
cells 13-15 mean             : 24.5s
ratio                        : 0.81x     (was conspicuously > 1)
```

Cell 12 is no longer an outlier — it is now the fastest of the four.

### C3c. Throughput left on the table, and a Windows DataLoader trap

Profiling the fixed loop showed it was GPU-bound, not I/O-bound: data loading
was only ~9% of step time. Measured on 960-slice runs:

| Configuration | slices/s |
|---|---|
| baseline (batch=8, no autotune) | 261 |
| + `cudnn.benchmark` | 290 |
| + TF32 | 321 |
| **batch=32** (+ both) | **442** |
| batch=64 / 128 | 435 / 431 (no better) |
| `channels_last` | 137 (much worse — not used) |

Net **1.69x**, so `batch_size` is 32 and `configure_performance()` enables
`cudnn.benchmark` and TF32.

A trap worth recording: **`num_workers > 0` is catastrophic here on Windows.**
Workers are spawned rather than forked and each re-opens the memory-mapped
cache:

```
num_workers=0 : 235.5 slices/s
num_workers=2 :  10.5 slices/s   (22x slower)
num_workers=4 :   4.8 slices/s   (49x slower)
```

`num_workers` stays 0, and Cell 5 documents why so nobody "optimises" it back.

**Net effect on the full run** (7,829 train + 1,670 val slices):

| | per epoch | typical (10 epochs x 4 models) | worst case (30 x 4) |
|---|---|---|---|
| Original | ~1 hour | — (never finishes) | ~123 hours |
| After caching only | 36.4 s | 24.3 min | 72.7 min |
| After all fixes | **21.5 s** | **14.3 min** | **42.9 min** |

### C3d. Phases 4 and 5 each read the whole cohort separately

Phase 4 loaded all 1,476 volumes to pool foreground intensities, then Phase 5
loaded the same 1,476 again to extract quality features. Both passes are now
fused: each volume is read once and does both jobs.

Measured on 60 patients with a warm page cache, so the comparison is compute,
not cold I/O:

```
fused (one pass) : 42.5s
separate total   : 46.1s   (Phase 4 22.0s + Phase 5 24.1s)
```

8% on warm cache, and it halves the read volume when the cache is cold.

A caution on measuring any of this: disk-read times on this machine vary by a
wide margin run to run — the same full-cohort label histogram has taken 36 s and
2,058 s. Timings above were taken with a warm page cache where that mattered.

### C3e. The epoch budget was ~4x larger than convergence needs

With the caching and warm-up fixes in place, Phase 9 still took around an hour
at `max_epochs = 30`. Measured at full scale (8,667 train + 1,836 val slices,
batch 32):

```
one epoch, all four models : 100.2s
  10 epochs x 4 models     : 16.7 min
  15 epochs x 4 models     : 25.1 min
  30 epochs x 4 models     : 50.1 min   <- the old default
```

30 epochs was never justified. Two pieces of evidence from the project's own
data say convergence is much faster:

* `results/metrics/phase9/phase9_model_comparison.csv` — the one completed run
  peaked at epochs **5, 2, 3 and 4** for the four architectures.
* Validation Dice after a *single* epoch is already 0.275–0.506.

`max_epochs` is now **15** with `patience` **3** (full mode) and 8 epochs in fast
mode — roughly three times the headroom the observed convergence needs, at a
third of the cost. Early stopping keeps the typical run well under the cap.

### C3f. Two further cache defects found while verifying

* **Labels were stored as `int64`** for values 0–3: 1.4 GB of a 3.9 GB cache,
  and eight times the bytes read per batch for no benefit. Now `uint8`
  (1,083 MB → 135 MB for train), with the existing cache converted in place and
  verified byte-identical after the round trip.
* **Fast and full mode shared one cache directory**, and reuse was decided by
  row count alone. Two different splits can yield the same slice count, so
  switching `RUN_MODE` could silently train on the wrong patients. The
  directory is now keyed by mode and cohort fraction, and the reuse check
  compares the actual `(patient_id, z)` list and the cached array geometry.

### Phase 9, measured end state

| | Old | Now |
|---|---|---|
| Fast mode, whole notebook | — | **5.6 min** (training 3.5 min) |
| Full mode, slice cache | — | 1.6 min, once, then reused |
| Full mode, per epoch (4 models) | ~4 hours | **100 s** |
| Full mode, all 4 models | ~123 h (30 ep), ~50 min after caching | **16–24 min** |

### C5. `argmax` over a one-channel head predicts nothing, silently
`Cell 76` (`predict_volume_labels`), `Cell 98`

```python
pred[:, :, z] = torch.argmax(logits, dim=1)[0].cpu().numpy()
```

The four checkpoints this project actually trained
(`results/checkpoints/phase9/*_best.pt`) have **`num_classes: 1`** — they are
*binary* whole-tumour segmenters with a single logit channel, trained against a
sigmoid. `argmax` over one channel returns index 0 for every voxel, so the
prediction is uniformly background and every metric is exactly 0.0 — with no
error, no warning and no shape mismatch, because the networks are fully
convolutional.

Measured on patient 002 with the Attention U-Net checkpoint:

```
argmax decode  : predicted classes {0}, 16384 voxels -> Dice 0.0000
sigmoid decode : Dice 0.6538, IoU 0.4857, HD95 14.87 mm
```

Note this also means the notebook's own code is **incompatible with the
project's own checkpoints**: the notebook builds 4-class models with
`NUM_SEG_CLASSES = 4` and `DiceCELoss`, while the saved weights are 1-class.
Further evidence that the results in `results/metrics/phase9/` were produced by
code that is not in this repository.

**Fix** — `phase08._predict_volume` probes the head width and decodes
accordingly: sigmoid-threshold for one channel, argmax for more. A binary head
reports whole-tumour metrics only, and TC/ET are reported as absent rather than
as zeros.

### C6. Evaluation ignored the resolution the model was trained at
`Cell 76`

`predict_volume_labels` fed native 240x240 slices to models trained on 128x128.
Fully convolutional networks accept that silently and score near zero, because
their features are learned at the training scale.

**Fix** — the training resolution is read from the checkpoint (`image_size`),
slices are resized for inference, and predicted labels are resized back
nearest-neighbour so metrics are computed at native resolution and true voxel
spacing.

### C4. Every cell using `display()` fails outside IPython
28 cells

`display` is an IPython builtin. No phase could be run as a script, which is why
there was no way to re-run the pipeline reproducibly.

**Fix** — `brats/io_utils.py::show` renders through IPython when present and
prints a formatted table otherwise.

---

## High

### H1. `robust_normalize` silently deleted ~1% of in-head voxels
`Cell 46`

```python
out[fg_mask] = (fg - lo) / (hi - lo)   # clipped-to-p1 voxels become exactly 0.0
```

Foreground voxels at or below the 1st percentile map to exactly `0.0`. Every
later step infers foreground as `volume != 0`, so those voxels silently became
*background* — for subsequent pipeline steps, for the feature extractor, and for
validation.

Measured on patient 001:

| Modality | foreground voxels | after normalize | lost |
|---|---|---|---|
| FLAIR | 1,342,885 | 1,328,909 | 13,976 (1.04%) |
| T1 | 1,342,885 | 1,328,889 | 13,996 (1.04%) |
| T1CE | 1,342,885 | 1,329,094 | 13,791 (1.03%) |
| T2 | 1,342,885 | 1,328,081 | 14,804 (1.10%) |

`contrast_enhance` has the same defect (CLAHE can map a voxel to 0).

**Fix** — `brats/preprocessing.py`: the foreground mask is computed **once** from
the original volume and passed to every operation, and foreground output is
floored at `FOREGROUND_FLOOR = 1e-6` so a real voxel can never become
background. Verified: all four pipelines now lose and gain exactly 0 voxels.

### H2. The noise feature was identically zero for the entire cohort
`Cell 36`, `measure_noise`

```python
lap = cv2.Laplacian(slice_2d, cv2.CV_64F)
mad = np.median(np.abs(lap - np.median(lap)))     # == 0.0, always
```

MAD was taken over the **whole slice**, where >85% of pixels are air with a
Laplacian of exactly 0. Both the median and the MAD therefore come out at
exactly `0.0` — for all 8 patients × 4 modalities, as measured:

```
modality  noise  noise_raw
   FLAIR    0.0        0.0
      T1    0.0        0.0
    T1CE    0.0        0.0
      T2    0.0        0.0      ... (identical for all 32 rows)
```

Consequence for the selector: `_relative_level(0, {p25: 0, p75: 0})` hits
`value <= p25` and returns `"low"` for every image. The noise branch always took
the `noise == "low"` path (`P1 += 0.5`, `P3 -= 0.25`), so **the two denoising
pipelines could never be chosen on noise grounds** — one of the selector's two
primary cues was dead. `measure_sharpness` had the same scope error; over the
full slice it is dominated by the brain/air step edge, so it tracked how much
brain was in the slice rather than sharpness.

**Fix** — `brats/features.py`: both statistics are restricted to the eroded
brain interior (`_interior_mask`). Noise std is now 0.036 and varies more
between patients than between modalities; all four pipelines are now reachable
(17 / 7 / 7 / 1 across the 32 test rows).

### H3. Quality features scaled with scanner gain, not image quality
`Cell 36` + `Cell 47`

Noise and sharpness were measured on **raw** intensities. MRI intensity has no
fixed unit: Phase 3 measures a 3.64× difference in foreground mean *between
modalities* and Phase 4 a 4.45× difference *between patients within T2*. Cell 47
then built reference quartiles **pooled across all four modalities**, so
"high noise" largely meant "bright modality".

Measured correlation of the noise proxy with plain foreground brightness:

| | r |
|---|---|
| raw-scale (as the notebook measured it) | **+0.572** |
| scale-invariant (fixed) | −0.247 |

**Fix** — features are measured on a slice mapped to its own robust `[p1, p99]`
range, and `selector.compute_cohort_references(..., per_modality=True)` builds
quartiles per modality with a pooled fallback. `--pooled-references` reproduces
the old behaviour for comparison. Raw variants are kept as `noise_raw` /
`sharpness_raw` for continuity.

### H4. "Adaptive" preprocessing was not adaptive for 98% of the cohort
`Cell 68` + `Cell 70`

Cell 68 split all 369 patients; Phase 5 extracted features for 8. `load_patient_multimodal_volume(..., apply_adaptive_preprocess=True)` looked up
features per patient and, finding none, **silently** fell back to plain
`robust_normalize`:

```python
if len(feat_rows): ... else: vol = robust_normalize(vol)   # no warning
```

So 361 of 369 patients were preprocessed non-adaptively while the code path and
the metadata said otherwise.

**Fix** — Phase 8 computes and prints coverage explicitly and refuses to
describe an uncovered cohort as adaptively preprocessed:

```
cohort patients            : 369
with adaptive decisions    : 8 (2.2%)
without adaptive decisions : 361
WARNING: ... Run `phase05 --n-patients 0` then phase06 to cover the whole cohort
```

### H5. Validation built a full autograd graph
`Cell 90`, `run_one_epoch`

The function detected evaluation via `optimizer is None` but never disabled
autograd, so every validation pass retained a graph over the whole epoch —
wasted memory and compute, and a plausible cause of the OOM path below.

**Fix** — `brats/training.py::run_one_epoch` wraps the loop in
`torch.no_grad()` when not training.

### H6. The OOM retry could not succeed
`Cell 91`

```python
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    return train_candidate_model(model_key, display_name, batch_size=4)
```

`model`, `optimizer` and `scaler` from the failed attempt are still referenced
by the live frame, so `empty_cache()` cannot release their memory and the retry
competes with the allocation that just failed. The error paths also returned
without freeing anything, leaking GPU memory into the next model's run.

**Fix** — `brats/training.py`: `_release()` drops the references *before*
retrying, halving works down to a floor rather than hardcoding 8→4, and a
`finally` block frees memory on every exit path.

### H7. `Generator.choice(values, replace=False)` on millions of voxels
`Cell 26`, `Cell 29`

`rng.choice(foreground, size=50_000, replace=False)` permutes the entire
population — ~8 million float32 per volume — to draw 50k samples.

**Fix** — `brats/stats.py::sample_values` draws indices instead. Phase 4 now
completes in **5.3 s**.

### H8. Patient 355's segmentation was treated as missing
Dataset / `Cell 4`

BraTS 2020 ships one patient whose mask breaks the naming convention:
`BraTS20_Training_355/W39_1998.09.19_Segm.nii`. The strict resolver raised
`FileNotFoundError`. That patient lands in the **train** split under the
project's own seed, so `build_slice_index` would have crashed partway through a
long run.

Verified the file is a valid mask: shape (240, 240, 155), labels {0, 1, 2, 4},
138,024 tumour voxels.

**Fix** — `io_utils.FALLBACK_NAME_TOKENS` tries documented naming exceptions
before giving up. Phase 1 now reports 369/369 patients complete.

---

## Medium

### M1. The segmentation-integrity check was vacuous
`Cell 62`

```python
seg_original = nib.load(seg_path).get_fdata(...)
seg_reloaded = nib.load(seg_path).get_fdata(...)   # same file, twice
identical = np.array_equal(seg_original, seg_reloaded)
```

Loading one file twice and comparing the copies can only pass. It proved nothing
about whether preprocessing touched the masks.

**Fix** — Phase 7 checks the properties that would actually break training: the
mask still aligns with the MRI volumes, its labels are inside the BraTS schema,
it is non-empty, and no registered preprocessing operation touches SEG.

### M2. `foreground_integrity` could not detect lost foreground
`Cell 58`

```python
checks["foreground_integrity"] = (not orig_fg) or prep_fg   # "any voxel left?"
```

This passes as long as *one* voxel survives, which is why H1 went unnoticed
despite this validator running over every volume.

**Fix** — `validate_preprocessed_volume` now reports
`foreground_voxels_lost`, `foreground_voxels_gained` and
`foreground_mask_preserved`, and `validation_ok` requires the mask to be
unchanged.

### M3. Split reproducibility depended on cell execution order
`Cell 65` + `Cell 68`

`rng_p8` was created in Cell 65 and consumed in Cell 68, so re-running Cell 65
alone, or running Cell 66 in between, changed the split. For a leakage-sensitive
patient-level split that is a real reproducibility hazard.

**Fix** — `data.split_patients` sorts, then shuffles with a generator seeded
from its own argument. `config.rng(seed)` hands every consumer a fresh
generator; nothing shares RNG state across phases.

### M4. Mutable default captured a stale reference
`Cell 48`

`def score_pipelines(feature_row, refs=cohort_refs)` binds the dict that existed
at definition time. Recomputing `cohort_refs` in a later cell had no effect on
selection — the function kept scoring against the old quartiles.

**Fix** — `references` is a required parameter throughout `brats/selector.py`.

### M5. Degenerate quartiles classified everything as "low"
`Cell 48`, `_relative_level`

With `p25 == p75` (a constant feature, or a tiny cohort) `value <= p25` is true,
so every image was "low". This is exactly what happened under H2.

**Fix** — `selector.relative_level` returns `"mid"` when the spread is
degenerate, and `"unknown"` for NaN. Phase 6 warns when the selector turns out
degenerate on the given subset.

### M6. Inconsistent pipeline ordering produced incomparable scales
`Cell 45`

`P2` was `robust_normalize → contrast_enhance` while `P4` was
`denoise → contrast_enhance → robust_normalize`. P2's output is CLAHE output,
P4's is percentile-normalised — different scales — yet Phase 7 compared
foreground means across them as if they were commensurable.

**Fix** — every pipeline ends with `robust_normalize`;
`validate_pipeline_registry()` enforces it at import. Phase 7 additionally
compares the scale-free coefficient of variation.

### M7. Unexpected segmentation labels were silently folded into background
`Cell 67`, `remap_brats_label`

The comment said "reported by caller if needed"; no caller ever did. A stray
label 3 or 5 would have become background with no indication.

**Fix** — `io_utils.remap_brats_labels` returns `(volume, unexpected_labels)`
and callers report them; Phase 2 flags any label outside the schema.

### M8. Checkpoint status was read from the filesystem, not the run
`Cell 91`

```python
"status": "completed" if ckpt_path.is_file() else "failed"
```

A stale checkpoint from an earlier run made a failed run report `completed`.

**Fix** — `train_model` tracks whether *this* run saved a checkpoint. Phase 9
also refuses to overwrite checkpoints unless `--overwrite-checkpoints` is
passed, and pilot runs get their own directory, so a short pilot cannot destroy
a full run's weights.

### M9. Checkpoints were fragile under PyTorch ≥ 2.6
`Cell 91`, `Cell 98`

`torch.save({... "config": PHASE9_CFG ...})` then `torch.load(path, map_location=DEVICE)`.
`weights_only=True` is the effective default in this project's PyTorch (2.11);
the current config happens to contain only primitives, but adding one `Path`
makes the checkpoint unloadable. Confirmed:

```
load with a Path in config -> UnpicklingError: Weights only load failed
```

**Fix** — `training.save_checkpoint` coerces metadata through JSON;
`load_checkpoint` passes `weights_only=True` explicitly and reads both the new
nested layout and the notebook's flat one.

### M10. The test split was created and never used
`Cell 84`, `Cell 96–99`

Phase 9 built `phase9_dev_split_ids["test"]`, trained, then reported
**validation** Dice as its result. 15% of the data was held out for nothing, and
model selection used the same split used for early stopping.

**Fix** — Phase 9 evaluates every trained model on the held-out test split and
selects on test Dice. It also prints the spread and warns when it is too small
to call a winner from one run.

### M11. A hard assert discarded hours of training
`Cell 97`

`assert ckpt_df["exists"].all()` aborts the notebook if any single model failed
— after all four have trained.

**Fix** — Phase 9 reports per-model status, evaluates whatever succeeded, and
returns a non-zero exit code instead of crashing.

### M12. `IndexError` on a missing dataset
`Cell 3`

`patient_folders[0]` with an absent dataset raises a bare `IndexError` 3 cells
after the real problem.

**Fix** — `io_utils.discover_patient_folders` raises `DatasetNotFoundError`
naming the expected layout.

### M13. `AttentionBlock` could be built with zero channels
`Cell 71`

`AttentionBlock(base, base, base // 2)` gives `inter_ch = 0` for `base < 2`,
producing a `Conv2d` with no parameters instead of an error.

**Fix** — floored at 1, plus `models.require_divisible_input` so a slice size
that is not a multiple of 8 fails with a clear message rather than a confusing
`torch.cat` shape error inside `forward`.

---

## Low

### L1. Duplicated cells
`Cell 9` ≡ `Cell 10` (the whole cohort modality scan, run twice) and
`Cell 50` ≡ `Cell 51`. **Fix** — deduplicated.

### L2. Deprecated AMP API
`Cell 89`, `Cell 91`: `torch.cuda.amp.GradScaler` / `torch.cuda.amp.autocast`
both emit `FutureWarning` under torch 2.11. Confirmed:

```
FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated.
               Please use `torch.amp.GradScaler('cuda', args...)` instead.
```

**Fix** — `torch.amp` with an explicit device type. The whole suite runs clean
under `-W error::FutureWarning`.

### L3. Cohort-wide reads for header-only questions
`Cell 11`, `Cell 12` called `get_fdata()` on all 1845 files to check
readability and shape. **Fix** — Phase 2 reads headers by default (full cohort
audit in **32 s**); `--deep` opts into the full voxel read.

### L4. Model reloaded from disk once per case
`Cell 76`, `evaluate_case` called `load_model_for_eval` inside the per-patient
loop. **Fix** — Phase 8 loads each model once and reuses it.

### L5. Dead code and no-ops
- `Cell 4`: the `t1`/`t1ce` guard is unreachable — the glob `*_t1.nii` cannot
  match `..._t1ce.nii`.
- `Cell 46`: `ksize = 0` assigned, never used.
- `Cell 69`: `.add_prefix("")` is a no-op.
- `Cell 33`: reloads all volumes to recompute a zero-proportion already present
  in the Phase 5 table, and overwrites `widest_modality` from Phase 3.
- `Cell 59`: `pipeline_match_decision_table` computed, never asserted or
  surfaced.

**Fix** — removed; the surviving equivalents are computed once.

### L6. Eager dataset documented as lazy
`Cell 73`: `BraTSSliceDataset`'s docstring claims to "load volumes lazily per
patient" while `__init__` materialises every tumour slice into
`self.samples` (~1.1 GB for 12 patients). **Fix** — see C3; the docstrings now
match the behaviour.

### L7. Empty project scaffolding
`README.md` and `requirements.txt` were both 0 bytes, and
`adaptive_preprocessing/`, `models/`, `figures/`, `logs/`, `data/processed/`
were empty. There was no way to reconstruct the environment.
**Fix** — both files written; `requirements.txt` pins the tested versions.

### L8. Plots could not be produced headless
Every figure cell ends in `plt.show()`, which does nothing useful in a script.
**Fix** — `io_utils.save_figure` writes to `figures/phaseNN/` and closes the
figure; `--show` opts into interactive display.

---

## Not a code bug, but worth recording

**Results exist whose generating code is not in the repository.**
`results/metrics/phase9/phase9_test_results.csv`,
`results/metrics/phase9/final_model_selection.csv` and the whole
`results/phase10/` tree (volume measurement, localisation, confidence summaries,
overlay PNGs, a per-patient JSON report) were produced by code that is not in
`training.ipynb` or anywhere else in the project. `results/cache/phase9_pilot_128/`
likewise implies a caching implementation that does not exist here.

Those results are therefore not reproducible from this repository, and the
Phase 9 numbers among them are subject to **C1** (inflated Dice). Phases 1–9 are
reimplemented in `src/`; Phase 10 is not, because there is no source for it.

Two further signs that those results came from different code: the checkpoints
are 1-class binary segmenters while the notebook builds 4-class models (**C5**),
and `results/cache/phase9_pilot_128/` implies a slice-caching implementation
that does not exist anywhere in the project.

---

## The saved model ranking does not reproduce

`results/metrics/phase9/final_model_selection.csv` names **Attention U-Net** the
final model (test Dice 0.787). Re-evaluating the preserved checkpoints per case
at volume level, with correctly decoded binary output and the corrected Dice,
over 12 held-out cases:

| Model | mean WT Dice | median | cases with Dice = 0 | mean HD95 (mm) | cases won |
|---|---|---|---|---|---|
| **U-Net** | **0.507** | 0.492 | 0 / 12 | **21.4** | **10 / 12** |
| U-Net++ | 0.402 | 0.337 | 0 / 12 | 25.1 | 2 / 12 |
| DeepLabV3+ | 0.256 | 0.184 | 2 / 12 | 33.3 | 0 / 12 |
| Attention U-Net | 0.187 | 0.094 | **5 / 12** | 64.5 | 0 / 12 |

The ranking is close to inverted. The model selected as best scores worst and
fails completely — predicting no tumour at all — on 5 of 12 cases, which an
aggregate slice-averaged Dice with empty-class credit would conceal.

This is a flag, not a verdict: the comparison differs from the original in more
than one respect (per-case volume metrics rather than slice averages, a resize
round-trip, 12 cases, and the original numbers come from code not in the repo).
It does mean the saved ranking cannot be relied on, and that the comparison
should be re-run properly:

```bash
python run_phase.py 9 --fraction 1.0            # retrain under one protocol
python run_phase.py 8 --evaluate                # per-case metrics for all models
```

---

# Addendum — full-cohort audit of `training_full_dataset.ipynb`

Audit of the completed 369-patient run (checkpoints in
`results/checkpoints/phase9/notebook_4class_full/`, metrics in
`results/metrics/phase9/phase9_notebook_4class_full_*.csv`). Every figure below
was measured against that run's cache and checkpoints, not estimated.

## What was already correct

The cohort really is complete and really is used. Verified directly against
`results/cache/phase9_f1_all_128/*_meta.csv`:

| Check | Result |
|---|---|
| Patients on disk | 369 |
| Selected at `cohort_fraction = 1.0` | 369 (identical set) |
| Split | train 258 / val 55 / test 56, union = all 369, pairwise disjoint |
| Cache rows match split patient lists | yes, all three splits |
| `(patient_id, z)` duplicates | 0 |
| Cached rows vs array length | equal in all three splits |
| Image range / finiteness | `[0, 1]`, no NaN or inf |
| Label values | ⊆ {0, 1, 2, 3} |
| `MAX_SLICES_PER_SPLIT` | `None` — no cap applied |

Slice totals: 17,251 train + 3,816 val + 3,592 test = 24,659.

---

## F1. Three in ten "healthy anatomy" slices were pure air — **High**

`src/brats/data.py`, `build_slice_index`

Tumour-free slices were sampled with:

```python
picks = np.unique(np.linspace(0, len(background_z) - 1, num=max_background_slices).astype(int))
```

`linspace` always includes both endpoints, so the first and last pick are always
the *first* and *last* tumour-free slice in the volume. In BraTS those are
`z = 0` and `z = 154`, which lie outside the head and contain nothing at all.

Measured over the full-cohort cache, by reading every cached image and testing
whether all four modalities are identically zero:

| Split | Background slices | Completely blank | Share of background | Share of all slices |
|---|---|---|---|---|
| train | 1,290 | 384 | 29.8% | 2.23% |
| val | 275 | 80 | 29.1% | 2.10% |
| test | 280 | 81 | 28.9% | 2.26% |

All of them sit at the ends of the volume: in train, 110 at `z < 5` and 258 at
`z ≥ 150`. No slice carrying tumour was affected.

An all-zero input against an all-background target carries no gradient signal
worth the memory, and Dice is undefined on it, so these rows also quietly
diluted the slice counts the run reported as its training set size.

**Fix.** A tumour-free slice now has to contain at least
`config.MIN_BRAIN_VOXELS_PER_SLICE` (1,000) non-zero voxels in FLAIR before it
can be sampled, and the filter runs *before* sampling, so the full quota of five
is still met. Indexing reads one extra volume per patient for this and nothing
thereafter. `min_brain_voxels=0` restores the old behaviour.

Verified on four patients: background picks moved from
`z = 0, 17, 118, 136, 154` to `z = 12, 21, 114, 124, 134`, and blank slices in
the rebuilt cache went from 8 to 0.

---

## F2. The headline Dice was averaged over batches — **High**

`src/brats/training.py`, `run_one_epoch`

Dice was computed inside each batch and the per-batch values averaged. That
makes the reported number depend on `batch_size`, and it weights a batch holding
one sliver of tumour exactly as heavily as a batch full of it. It is not a
definition anyone publishes against.

Same U-Net weights, same 3,592 test slices, two reductions of the same per-voxel
counts:

| Reduction | Mean foreground Dice |
|---|---|
| Per-batch, then averaged (what the run reported) | **0.6259** |
| Aggregated over the whole pass | **0.7505** |

Aggregated per class, U-Net on the test split:

| Class | Dice | Recall | Precision |
|---|---|---|---|
| necrotic core | 0.648 | 0.534 | 0.825 |
| oedema | 0.789 | 0.781 | 0.797 |
| enhancing | 0.815 | 0.862 | 0.772 |

**Fix.** `run_one_epoch` accumulates intersection and mask sizes per class
across the pass and divides once at the end. It returns `dice`/`iou`
(aggregated, now the headline and the early-stopping signal), `dice_batchmean`/
`iou_batchmean` (the old numbers, kept so earlier runs stay comparable), and
`per_class_dice`/`per_class_iou`. The training history gains a per-class
validation Dice column, so a collapse on one tumour region is visible instead of
being hidden inside the mean.

---

## F3. No per-patient or per-region evaluation existed — **High**

Every reported number was slice-level, and measured only on the cached slices —
which are tumour-rich by construction. Those 56 test patients have 8,680 axial
slices between them; 3,592 were cached. The ~5,100 healthy slices left out are
exactly the ones on which a model can invent a tumour, so nothing in the
slice-level figure penalised that failure.

**Fix.** `training.evaluate_patients_volumetric()` runs a model over all 155
slices of each patient and scores whole volumes on the BraTS regions
(WT = {1,2,3}, TC = {1,3}, ET = {3}), averaging over patients.
`training.summarise_volumetric()` collapses it to one row. Notebook Cell 19b
calls both.

Measured on the existing full-cohort checkpoints, 56 held-out patients, all
slices:

| Model | WT | TC | ET | mean | FP voxels per healthy slice |
|---|---|---|---|---|---|
| DeepLabV3+ | 0.871 | 0.774 | 0.702 | **0.782** | 1.9 |
| U-Net | 0.870 | 0.773 | 0.696 | 0.780 | 0.9 |
| Attention U-Net | 0.877 | 0.764 | 0.696 | 0.779 | 2.3 |
| U-Net++ | 0.865 | 0.770 | 0.685 | 0.773 | 3.8 |

Two things follow. The models are substantially better than the reported 0.626
suggested — these are ordinary BraTS 2020 2D-baseline figures. And the
architecture ranking changes: the slice metric picked U-Net, the volumetric
metric picks DeepLabV3+. Both gaps (0.002 and 0.009) are far smaller than the
per-patient spread (WT std ≈ 0.10), so neither is a real winner; the point is
that the metric choice reorders the table.

False positives on genuinely healthy slices are negligible (0.9–3.8 voxels per
slice), so the models are not hallucinating tumour.

---

## F4. Nothing ever displayed a prediction — **Medium**

The notebook visualised inputs and masks (Cell 6) and plotted training curves,
but no cell ever rendered a model output. A segmentation project cannot be
assessed on scalars alone: Dice 0.87 is consistent both with a clean boundary
error and with a tumour found in the wrong lobe.

**Fix.** Notebook Cell 19c renders two figures, saved to `figures/phase09/`:

* `prediction_overlays.png` — six held-out patients, ground truth against all
  four architectures, WT Dice per panel.
* `prediction_error_maps.png` — the best model with errors separated into false
  positive, false negative and right-region-wrong-sub-class, which are not
  equivalent failures.

Inspected: predictions sit on the FLAIR hyperintensity, per-slice WT Dice runs
0.88–0.97, and errors are thin boundary bands plus necrotic-core/oedema
confusion in the tumour interior — consistent with the 0.53 necrotic recall in
F2. Nothing scattered, nothing displaced.

---

## F5. Overfitting from about epoch 6 — **Medium**

The epoch budget itself is sound. Every model early-stopped inside
`max_epochs = 30`:

| Model | Best epoch | Epochs run | Best val Dice |
|---|---|---|---|
| U-Net | 6 | 11 | 0.6414 |
| U-Net++ | 7 | 12 | 0.6443 |
| Attention U-Net | 9 | 14 | 0.6393 |
| DeepLabV3+ | 13 | 18 | 0.6401 |

So 30 with patience 5 is the right order for the full cohort — and note this
contradicts **C3e** above, which cut the budget to 15/3 on half-cohort evidence.
DeepLabV3+ peaks at epoch 13 and would have been truncated at 15/3. The
full-cohort notebook keeps 30/5.

The problem is not the cap but what happens under it. Training Dice reaches
0.90 while validation Dice is flat from epoch 6, and training loss falls to
0.056 against a validation loss stuck at 0.22. The later epochs only deepen the
overfit. There was no augmentation and no learning-rate schedule.

**Fix.** Both added, opt-in, switched on in the notebook via `USE_AUGMENTATION`
and `USE_LR_SCHEDULER`:

* `data.AugmentedSliceDataset` — random flips and 90° rotations, applied
  identically to image and mask, seeded per `(epoch, index)`, training split
  only. All eight dihedral transforms of an axial slice are plausible brains.
* `train_model(lr_scheduler=True)` — `ReduceLROnPlateau` on validation Dice,
  factor 0.5, patience 2. The history gains a `learning_rate` column.

This changes the training protocol, so the numbers in F2 and F3 belong to the
*unregularised* run. Re-running is needed to measure the effect.

---

## F6. The CUDA-OOM retry silently changed protocol — **Low**

`train_model`'s OOM path recurses with a halved batch size but did not forward
`drop_last` or `progress_every`, so a run that hit OOM also lost uniform batch
shapes (and with `cudnn.benchmark` on, paid a fresh ~7 s autotune per ragged
batch) and went silent mid-epoch. Both are now forwarded, along with the new
scheduler arguments.

---

## F7. Slices with 1–49 tumour voxels reach neither split — **Low, documented**

`build_slice_index` keeps a slice if it has ≥ 50 tumour voxels and treats it as
background only if it has exactly 0. Slices in between are indexed as neither
and never reach training or evaluation.

Measured over all 369 patients: 1,608 such slices (1,110 train / 240 val / 258
test), affecting 365 of 369 patients, carrying 30,196 tumour voxels — **0.082%**
of all tumour voxels in the cohort.

Small enough to leave alone, but it is a real gap at the superior and inferior
tumour margins, and the evaluation never sees those slices. Recorded rather than
changed; the threshold is `config.MIN_TUMOUR_VOXELS_PER_SLICE`.

---

---

## F8. The slice cache could not be rebuilt, and said nothing useful — **Medium**

Found while rebuilding the cache under the F1 fix. The rebuild died on:

```
OSError: [Errno 22] Invalid argument:
  '...\results\cache\phase9_f1_all_128\train_x.npy'
```

Errno 22 names the path and nothing else. The real cause is Windows
`ERROR_USER_MAPPED_FILE`: `open_memmap(mode="w+")` truncates the existing file,
and the file still had a live memory mapping. The holder was the user's own
Jupyter kernel (PID 27488, started 16:02), which had built a
`CachedSliceDataset` when it ran the Phase 9 cells — and that dataset keeps
`np.load(..., mmap_mode="r")` open for the life of the kernel. Nothing in the
message says lock, holder, or remedy.

Two independent faults, fixed separately:

**The cache lived in a cloud-synced directory.** `results/cache/` sits under
`OneDrive\Documents`, so a 6.4 GB derived artifact that rebuilds in minutes was
being uploaded on every rebuild and was eligible for dehydration to a
cloud-only placeholder. `config.CACHE_DIR` now honours a `BRATS_CACHE_DIR`
environment variable, so the cache can be put somewhere local without moving
the project. The default is unchanged, which is correct for a project directory
that is not synced.

**Truncation in place, with an opaque failure.** `build_slice_cache` now
unlinks a stale array before recreating it, and when the unlink fails it raises
a message naming the likely holder and the two ways out (restart the kernel, or
point `BRATS_CACHE_DIR` elsewhere) instead of surfacing the bare errno.

Verified: with `BRATS_CACHE_DIR` set to a local path, the full-cohort rebuild
ran to completion while the locking kernel was still alive, and the new cache
reports **0 completely blank slices against 545 in the cache it replaced** —
confirming F1 on the real cohort rather than on the four-patient sample.

---

## Test suite

There was none. There is now: `tests/`, 153 tests, ~18 s, no GPU and no
dataset. The suite builds a miniature BraTS-shaped cohort in a temp directory
(3 patients, 24×24×12 volumes, correct filenames, raw labels `{1,2,4}`) and
runs the real indexing, caching and resizing code against it; the few tests
that need the actual cohort are marked `needs_dataset` and skip without it.

The tests are written against the defects recorded in this file, so each one
documents the failure it prevents. Among them:

* Dice of two empty masks is NaN, never 1.0 (**C1**)
* aggregated Dice is batch-size invariant and batch-averaged Dice demonstrably
  is not (**F2**)
* `robust_normalize` keeps background exactly zero and never shrinks the
  foreground, for every pipeline (**H1**)
* splits depend only on the seed, not on call order (**M3**)
* label maps are resized nearest-neighbour, so no class is invented
* air-only slices are not sampled as healthy anatomy (**F1**)
* augmentation applies the same transform to image and mask, and preserves the
  label set
* cache reuse compares the actual slice list, not the row count (**C3f**)
* checkpoints round-trip under `weights_only=True` (**M9**)
* a stale checkpoint file cannot make a failed run report success (**M8**)
* a small `base` cannot produce a parameterless attention block (**M13**)

CI (`.github/workflows/tests.yml`) runs the suite on Python 3.11 and 3.12 and
checks that all nine phase modules import.

---

## Reproducing this audit

```bash
# coverage, blank slices, dropped slices
python run_phase.py 9 --fraction 1.0      # rebuilds the cache under the F1 fix
```

The notebook now carries the checks inline: Cell 5 reports blank slices in the
cache, Cell 10 shows batch-averaged against aggregated Dice on real validation
data, Cell 19b reports per-patient WT/TC/ET, and Cell 19c renders predictions.
