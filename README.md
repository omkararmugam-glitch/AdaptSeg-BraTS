

Brain tumour segmentation on the BraTS 2020 training cohort (369 patients, four
MRI modalities each). The project asks whether preprocessing and architecture
choices should be made *per image* rather than fixed globally, and builds the
measurement apparatus needed to answer that honestly.

The work is organised as nine phases. Each phase is a standalone script: it
reads the artifacts earlier phases wrote, writes its own, and shares no
in-memory state with any other phase.

---

## Quick start

```bash
# 1. Install (PyTorch first, from the index matching your CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# 2. Check the data is where it should be
python run_phase.py 1

# 3. Run the analysis chain (phases 1-8, a few minutes)
python run_phase.py all

# 4. Train the four candidate architectures (opt-in; needs a GPU)
python run_phase.py 9 --fraction 0.5
```

`python run_phase.py list` prints every phase with its dependencies and
artifacts. `python run_phase.py <n> --help` shows one phase's options.

Expected data layout:

```
data/raw/BraTS2020/MICCAI_BraTS2020_TrainingData/
    BraTS20_Training_001/
        BraTS20_Training_001_flair.nii
        BraTS20_Training_001_t1.nii
        BraTS20_Training_001_t1ce.nii
        BraTS20_Training_001_t2.nii
        BraTS20_Training_001_seg.nii
    ...
```

---

## The phases

| # | Phase | Question it answers | Needs |
|---|---|---|---|
| 1 | Data loading | Is the dataset complete and readable? | — |
| 2 | Integrity audit | Do all 369 patients have five consistent, readable volumes? | — |
| 3 | Intensity analysis | How is intensity distributed in one patient? | — |
| 4 | Cohort distribution | How much does intensity vary between patients? | — |
| 5 | Quality features | What measurable properties distinguish these images? | — |
| 6 | Adaptive selector | Which preprocessing pipeline suits each image? | 5 |
| 7 | Execution + validation | Does the chosen preprocessing run correctly? | 6 |
| 8 | Splits + selection dataset | What would a learned model selector train on? | 5, 6 |
| 9 | Training benchmark | Which architecture wins on held-out data? | — |

Artifacts land in `artifacts/phaseNN/` as CSV, figures in `figures/phaseNN/`,
checkpoints in `results/checkpoints/`, slice caches in `results/cache/`.

### What each phase establishes

**Phase 1–2 — the data is sound.** All 369 patients carry five
240×240×155 volumes at 1 mm isotropic spacing. One patient
(`BraTS20_Training_355`) ships its mask under a non-conforming name,
`W39_1998.09.19_Segm.nii`; the resolver handles that known BraTS quirk.

**Phase 3–4 — intensity has no fixed scale.** Background occupies ~85% of every
volume, so full-volume statistics describe air. Within the foreground, the mean
differs by 3.6× *between modalities* and up to 4.4× *between patients in one
modality*. Tumour voxels are 1.1% of the cohort — an ~89:1 class imbalance.

That is the case for adaptive, foreground-aware, modality-aware preprocessing:
no single global transform can suit every image.

**Phase 5 — features that mean something.** Contrast, noise, sharpness, entropy,
intensity range and foreground ratio, each measured scale-invariantly. This
matters: measured on raw intensities, the "noise" proxy correlates r = +0.57
with plain brightness, i.e. it restates scanner gain rather than image quality.

**Phase 6 — four candidate pipelines, chosen by rule.**

| Pipeline | Operations | Chosen when |
|---|---|---|
| `P1_robust_norm` | normalise | baseline; good contrast, low noise |
| `P2_contrast_norm` | CLAHE → normalise | contrast low for its modality |
| `P3_denoise_norm` | blur → normalise | noise elevated |
| `P4_denoise_contrast_norm` | blur → CLAHE → normalise | both |

Reference quartiles are computed **per modality**, so "low contrast" means low
for a T1CE rather than low compared to a FLAIR. Every pipeline ends on the same
normalisation step so outputs are comparable.

This is a transparent prototype, **not** a trained or validated optimum. It
encodes the Phase 3–5 findings as readable rules; Phase 8 builds the dataset
needed to replace it with a learned selector.

**Phase 7 — preprocessing is verified, not assumed.** Each result is checked for
preserved shape, no NaN/Inf, untouched background, and — importantly — an
*unchanged foreground mask*. Preprocessing must not quietly reclassify in-head
voxels as air.

**Phase 8 — the foundation for a learned selector.** A patient-level
train/val/test split (70/15/15, asserted disjoint), a registry of the four
architectures with real checkpoint status, and one row per case joining
characteristics, preprocessing decision, and measured performance.

With no trained checkpoints the performance columns stay empty and the phase
says so. Nothing is invented. Pass `--evaluate` to measure per-case Dice, IoU,
sensitivity, specificity and HD95 for whichever models have checkpoints.

**Phase 9 — the architecture benchmark.** Trains U-Net, U-Net++, Attention U-Net
and DeepLabV3+ under one protocol: 128×128 slices, 4 input channels,
`0.5·CrossEntropy + 0.5·softDice`, Adam, AMP, early stopping on validation Dice.
Each model is then evaluated on the **held-out test split**, and selection is
made on test Dice.

```bash
python run_phase.py 9 --fraction 0.5                       # half the cohort
python run_phase.py 9 --models unet unetpp --epochs 10     # two models
python run_phase.py 9 --max-slices-per-split 2500          # fast pilot
python run_phase.py 9 --adaptive-preprocessing             # use Phase 6 decisions
python run_phase.py 9 --dry-run                            # prepare data only
```

Adaptive preprocessing is **off by default** here, so the architecture
comparison is controlled: one preprocessing, four models. Turning it on changes
two variables at once, which cannot attribute a difference to either.

Slice selection keeps every slice with at least
`config.MIN_TUMOUR_VOXELS_PER_SLICE` (50) tumour voxels, plus up to
`MAX_BACKGROUND_SLICES_PER_PATIENT` (5) tumour-free slices that contain at least
`MIN_BRAIN_VOXELS_PER_SLICE` (1,000) voxels of brain — the brain floor exists
because the even sampling otherwise always picked `z = 0` and `z = 154`, which
are outside the head, making ~30% of every split's healthy-anatomy sample
completely blank. Slices holding 1–49 tumour voxels fall into neither category
and are not used: 1,608 slices cohort-wide, carrying 0.082% of all tumour
voxels.

---

## Reading the results

Three cautions apply to any number this project produces.

**Dice excludes undefined cases.** Where a class appears in neither prediction
nor target, Dice is undefined and reported as NaN, not 1.0. The convenient
`(2·inter + ε)/(sum + ε)` form awards a perfect score for predicting nothing,
and most BraTS slices contain no enhancing tumour — a model predicting pure
background scores 0.667 that way versus 0.0 correctly. Figures computed with the
smoothed form are not comparable with these.

**HD95 is a real surface distance.** 95th percentile of the symmetric
surface-to-surface distance in millimetres, in 3D, using header spacing.

**One run does not establish a winner.** Phase 9 prints the test-Dice spread
across architectures and warns when it is too small to call. Repeat over several
seeds before claiming an architecture is better.

**Dice is aggregated over the pass, not averaged over batches.** Computing Dice
inside each batch and averaging the batches weights a batch holding one sliver
of tumour as heavily as a batch full of it, and shifts the result whenever
`batch_size` shifts. Measured on the full-cohort test split with the same U-Net
weights: 0.626 batch-averaged against 0.751 aggregated. `run_one_epoch` still
returns the old figure as `dice_batchmean` so earlier runs stay comparable.

**Slice-level and patient-level Dice are different numbers.** The cached splits
hold only tumour-rich slices plus at most five tumour-free samples per patient —
3,592 of the 8,680 slices the 56 test patients actually have. A model is
therefore never asked about the healthy anatomy on which it could invent a
tumour. Report the per-patient volumetric figures from
`training.evaluate_patients_volumetric()` (whole volumes, WT/TC/ET, averaged
over patients), which is what BraTS results mean; the slice figure is a
training diagnostic.

On the full cohort the two disagree in both level and ranking:

| | slice-level, aggregated | per-patient WT | TC | ET | mean |
|---|---|---|---|---|---|
| U-Net | 0.751 | 0.870 | 0.773 | 0.696 | 0.780 |
| DeepLabV3+ | 0.746 | 0.871 | 0.774 | 0.702 | **0.782** |
| Attention U-Net | 0.747 | 0.877 | 0.764 | 0.696 | 0.779 |
| U-Net++ | 0.742 | 0.865 | 0.770 | 0.685 | 0.773 |

The gaps between architectures are far smaller than the per-patient spread
(WT std ≈ 0.10), so this still does not name a winner — it only shows that the
choice of metric reorders the table.

### Pre-existing results in this repository

`results/metrics/phase9/` and `results/phase10/` contain outputs from earlier
work whose **generating code is not in this repository**. They are not
reproducible from here, and the Phase 9 metrics among them were computed with
the inflated Dice described above. Treat them as historical.

`results/checkpoints/phase9/*_best.pt` are genuine trained weights and are
preserved. Note they are **binary whole-tumour** segmenters (single logit
channel, `num_classes: 1`) trained at 128×128 — not the 4-class models the
notebook code builds. Phase 8 detects this and decodes them with a sigmoid;
they report whole-tumour metrics only.

Re-evaluating those checkpoints per case with the corrected metric does **not**
reproduce the saved ranking: U-Net wins 10 of 12 cases while Attention U-Net,
named the final model in `final_model_selection.csv`, scores worst and predicts
no tumour at all on 5 of 12. See the end of `BUGS.md`. Retrain and re-evaluate
before relying on any architecture comparison:

```bash
python run_phase.py 9 --fraction 1.0     # retrain under one protocol
python run_phase.py 8 --evaluate         # per-case metrics
```

---

## Layout

```
run_phase.py              # CLI: run one phase, or the chain
src/
  brats/                  # shared library
    config.py             # paths, constants, seeding
    io_utils.py           # dataset discovery, NIfTI resolution, label remapping, output
    stats.py              # intensity statistics
    features.py           # scale-invariant image-quality features
    preprocessing.py      # operations, pipelines, validation
    selector.py           # cohort references and the rule-based selector
    metrics.py            # Dice, IoU, sens/spec, 3D surface HD95
    models.py             # the four architectures + factory
    data.py               # splits, slice indexing, caching, datasets, samplers
    training.py           # loss, epoch loop, trainer, checkpointing
  phases/
    phase01_data_loading.py        ... phase09_training_benchmark.py
artifacts/phaseNN/        # per-phase CSV outputs
figures/phaseNN/          # per-phase figures
results/                  # checkpoints, caches, metrics
data/raw/BraTS2020/       # input data (not in version control)
training.ipynb            # original notebook, superseded -- see BUGS.md
BUGS.md                   # audit of the notebook: 30+ defects and their fixes
```

### `training.ipynb`

**All nine phases are fixed.** The original 101 cells are replaced by 39 that
drive the library in `src/brats/`, so the notebook and the CLI run identical,
tested code.

One switch at the top of the notebook controls everything:

```python
RUN_MODE = "fast"   # representative subset — ~2 min end to end, for developing
RUN_MODE = "full"   # whole cohort — the configuration to report
```

| | Before | After |
|---|---|---|
| Cells | 101 | 39 |
| Whole notebook, fast mode | 35+ min, never completed | **5.6 min** |
| Phases 1–8, full cohort | — | 7–9 min |
| Phase 9 slice cache (full) | — | 1.6 min, once, then reused |
| Phase 9 training, 4 models | ~123 h | **16–24 min** |

Phase 9's epoch budget is set from measurement, not habit: one epoch across all
four models costs ~100 s at full scale, and the project's own completed run
peaked at epochs 2–5. So `max_epochs` is 15 with patience 3 — about three times
the headroom convergence needs, instead of the 30 that cost an hour.

That budget was measured on the half cohort. **On the full 369-patient cohort
convergence is later, and 15/3 would truncate it:** best epochs were 6 (U-Net),
7 (U-Net++), 9 (Attention U-Net) and 13 (DeepLabV3+), with runs ending at 11,
12, 14 and 18. `training_full_dataset.ipynb` therefore keeps `max_epochs = 30`
with patience 5, and every model still stops well inside it.

Epochs are not the limiting factor there. Training Dice reaches 0.90 while
validation Dice is flat from about epoch 6 — memorisation, not a shortage of
epochs. `data.AugmentedSliceDataset` (flips and 90° rotations, training split
only) and `train_model(lr_scheduler=True)` (`ReduceLROnPlateau` on validation
Dice) address that; both are off by default in the library and on in the
full-cohort notebook.

Because each phase imports from `src/brats/` rather than relying on globals from
earlier cells, any phase can be run on its own after a kernel restart.

---

## The deployment run

`scripts/run_deployment_benchmark.py` trains all four architectures over the
full cohort under the protocol meant for a result you would report, rather
than the shorter one used while developing:

```bash
export BRATS_CACHE_DIR="$HOME/.cache/brats"      # keep 6.4 GB out of OneDrive
python scripts/run_deployment_benchmark.py
```

| Setting | Value | Why |
|---|---|---|
| `max_epochs` | 60 | Augmentation moves convergence later than the 6-13 epochs the unaugmented run needed |
| early-stopping patience | 10 | An augmented validation curve is noisier; 5 would stop on noise |
| LR schedule | `ReduceLROnPlateau`, factor 0.5, patience 4 | Sits inside the early-stopping patience, so the rate can drop two or three times before the run ends |
| augmentation | flips + 90° rotations, training split only | Augmenting validation would make selection measure something the test set lacks |
| selection | mean per-patient Dice over WT/TC/ET, whole volumes | The figure comparable with published BraTS results |

It writes its summary, history and metrics after **every** model, so an
interruption costs one model rather than the whole run, and re-running skips
what is already on disk. Checkpoints go to a separate run name, so an earlier
run's weights are never overwritten.

---

## Tests

```bash
pip install pytest
pytest
```

153 tests, about 18 seconds, no GPU and no dataset: the suite builds a
miniature BraTS-shaped cohort in a temp directory (3 patients, 24x24x12
volumes, correct filenames and raw labels) and exercises the real indexing,
caching and resizing code against it. The handful of tests that genuinely need
the 31 GB cohort are marked `needs_dataset` and skip when it is absent.

They are written against the defects this project actually hit, so a
regression is caught by a test that explains what went wrong:

| Area | What is pinned |
|---|---|
| `test_metrics.py` | Dice of two empty masks is NaN, never 1.0; IoU never exceeds Dice; HD95 scales with voxel spacing and is NaN when a mask is empty |
| `test_data.py` | splits are disjoint and depend only on the seed, never on call order; label maps are resized nearest-neighbour so no class is invented; air-only slices are not sampled as "healthy anatomy"; augmentation applies the *same* transform to image and mask and preserves the label set; cache reuse compares the actual slice list, not the row count |
| `test_preprocessing.py` | background stays exactly zero and the foreground never shrinks, for every pipeline; normalisation is scale-invariant |
| `test_training.py` | aggregated Dice is batch-size invariant and batch-averaged Dice is not; validation leaves no autograd graph; checkpoints round-trip under `weights_only=True`; a stale checkpoint file cannot make a failed run report success |
| `test_models.py` | every architecture returns the input resolution with the requested channel count; a small `base` cannot produce a parameterless attention block |
| `test_config.py` | seeding is reproducible; the cache directory is keyed by fraction, cap and size; raw label 4 maps to training label 3 and unknown labels are reported |

CI runs the suite on Python 3.11 and 3.12 and checks that all nine phase
modules import ([.github/workflows/tests.yml](.github/workflows/tests.yml)).

### Where the slice cache lives

The cache is a derived artifact — 6.4 GB at full cohort, rebuilt from the
dataset in minutes. By default it sits in `results/cache/`, which is fine until
the project directory is cloud-synced, as this one is. Then two things bite:

* the sync client uploads every rebuild, and can dehydrate the files to
  cloud-only placeholders;
* a notebook kernel that ran the Phase 9 cells holds a memory map on the cache
  for as long as it lives, and rebuilding then fails on Windows with a bare
  `OSError: [Errno 22] Invalid argument` that names only the path.

Set `BRATS_CACHE_DIR` to somewhere local and the problem disappears:

```bash
# Windows
set BRATS_CACHE_DIR=%LOCALAPPDATA%\brats_cache
# bash
export BRATS_CACHE_DIR="$HOME/.cache/brats"
```

The rebuild also now deletes a stale array before recreating it, and reports
the lock and what to do about it rather than the raw errno.

> **The editor runs its own in-memory copy of a notebook, not the file on
> disk.** If `training.ipynb` was open while it was being fixed, VS Code or
> Jupyter will keep showing and running the *old* cells no matter what the file
> says — and pressing Save in that tab overwrites the fixed version.
>
> Two ways around it, in order of reliability:
>
> 1. **Run from a terminal — immune to the problem:**
>    ```bash
>    python run_phase.py 9 --fraction 0.5
>    ```
>    or double-click `train_phase9.bat`.
> 2. **Open `training_fixed.ipynb`** — identical content under a name the editor
>    has never cached, so it must load fresh. Close the old `training.ipynb`
>    tab *without saving*.
>
> Phase 9's trainer also refuses to start if the kernel and the notebook
> disagree, so a stale kernel now fails in a second instead of after an hour.

Backups: `training.before_phase1_8_fix.ipynb` (Phase 9 fixed, 1–8 original) and
`training.ipynb.bak` (entirely original).

`BUGS.md` documents every defect with the measurement that demonstrates it.

---

## Notes on running

- **Phase 2's label histogram** is the only step that reads voxel data for the
  whole cohort (~1.4 GB). Warm cache ~35 s; on a cloud-synced folder a cold read
  can take tens of minutes. `--skip-labels` drops the rest of the audit to ~3 s.
- **Phase 5 defaults to 8 patients.** `--n-patients 0` covers the whole cohort.
  Phase 8 reports coverage and warns when most of the cohort has no adaptive
  decision — in that state the pipeline is *not* adaptive for those patients.
- **Phase 9 will not overwrite checkpoints.** Pilot runs (`--max-slices-per-split`)
  get their own directory; a conflicting run needs `--run-name` or
  `--overwrite-checkpoints`.
- **Figures are saved, not shown.** Pass `--show` for interactive display,
  `--no-figures` to skip them.
#   A d a p t S e g - B r a T S  
 