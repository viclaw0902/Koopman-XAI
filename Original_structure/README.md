# TEP final experiment: original shared temporal-AE structure

## Scope and architecture

All scripts and outputs in this directory are new. Historical project scripts, prepared arrays, checkpoints, and results are read-only. `../Changed_structure` is reserved and remains empty.

The historical `15faults_Ks/route3_shared_ae.py` uses structure B: the same input window independently enters CNN-GRU for classification and a temporal convolutional encoder for manifold modeling. It does not feed CNN-GRU hidden states to the manifold encoder. Some old aggregate scripts still contain structure A; those are not the source of the multi-K checkpoint.

The new experiment preserves structure B and the historical CNN-GRU and temporal-AE layer sizes. A learned lift maps h=16 to z=16/32/64/128. An independent full-input MLP maps ALL z coordinates to h=16. No prefix slicing, passthrough, or prefix-only inverse is permitted. The shared AE and classifier are frozen before dynamics training. The nonlinear lift, inverse, and 21 autonomous K matrices are trained. No XMV control input and no affine bias are added to the Koopman dynamics.

## Data and fixes verified before launch

- Input: XMEAS1–22 and XMV1–11, 33 channels; window20, stride1; all legal classification windows.
- Classes: Normal plus Fault1–20, 21 classes. Attribution metrics only Fault1–12,14,15.
- Existing split: 400 training,50 validation,50 test runs per class, with disjoint run IDs. No resplitting across seeds.
- Counts: 4,040,400 training windows;505,050 validation windows;988,050 test windows per seed.
- Training and validation fault runs have20 initial normal samples, testing fault runs160. These prefixes were checked against the same-run normal trajectory for every fault trajectory. Classification labels use Normal until the window endpoint crosses injection. Windows spanning injection remain in classification, labeled by their endpoint. Dynamics pairs spanning injection are excluded, not silently assigned to a mode.
- Historical float32 scaler accumulation caused normal-training XMEAS9 to have standardized mean~0.9966 and std~0.08017. The equivalent affine normalization is refit in float64 ONLY on normal training trajectories and applied in memory to all splits. Old prepared files remain untouched. New scaler is in Results/normal_scaler_corrected.npz. This corrects historical scaling up to the stored float32 precision; it is not a raw-RData regeneration.
- Only fault-free training trajectories fit normalization, Expected IG backgrounds and AFR normal representation centers. Manifold fitting includes all training modes. z normalization includes all training modes and is fixed after lift pretraining.

## Training and selection

Seeds42,43,44,45,46 run sequentially to respect the MX350 2GB GPU limit. Each seed independently trains the shared classifier and AE. Learning rate1e-3, batch128, accumulation2, AdamW, minimum15 and maximum50 epochs, early stopping patience7. Actual settings and histories are saved.

Each z dimension uses inverse pretraining, per-class ridge initialization selected by recursive10-step validation error, and end-to-end one/ten-step training of lift, inverse and K. Recursive prediction never receives future target states as inputs. Target encodings are detached in the dynamics loss. h inverse, original reconstruction, cycle and variance constraints prevent trivial low-error collapse. Spectral-radius cap1.05 prevents explosive operators without imposing strict contraction below1.

Choose the smallest noncollapsed dimension within2% of the minimum validation decoded10-step MSE + reconstruction MSE. Also export latent errors, h errors, raw persistence baseline, effective rank, per-class metrics and spectra. Test attribution metrics do not select dimensions or hyperparameters.

## Endpoints and paths

For each of the14 evaluation faults use all50 test trajectories. Normal medoid candidates are fully pre-injection windows. Fault medoid candidates are fully post-injection windows predicted as any Fault. Compute exact empirical medoids using Euclidean/Frobenius distance in the common standardized original33x20 space. The fault medoid prediction determines both target logit and fixed K. Wrong predictions are not replaced by ground truth. A missing fault candidate is explicitly reported as failed coverage, not given a fabricated endpoint.

Route1 uses the normal-medoid straight line. Route2 optimizes decoded path energy under the shared decoder pullback geometry. Koopman IG optimizes fixed-endpoint dynamics residuals plus decoded geometry and latent validity; initialization uses observed same-trajectory windows. K powers correspond to actual window time gaps, not IG sample counts. K is fixed throughout explanation. This is a numerically optimized, fixed-endpoint dynamics bridge, not a guaranteed exact free K rollout. Optimization convergence and residuals are exported rather than assumed.

Decode z->learned inverse->h->shared decoder->original window. Explicit straight endpoint connectors link real windows to reconstructed endpoints. These connectors are integrated and their attribution fraction is exported. All methods evaluate the same frozen CNN-GRU. Curve contributions use per-segment original-space increments and input gradients, not straight-line delta-times-average-gradient. Integration increases resolution up to1024 steps based on completeness tolerance; failures remain visible in diagnostics.

## Comparison and metrics

- Ablation: normal-medoid IG, manifold geodesic IG, Koopman IG, sharing endpoints and target logit.
- Comparison: zero-baseline Vanilla IG, zero-baseline Guided IG, zero-baseline IDGI, Saliency, Expected IG, normal-medoid DeepLIFT, AFR-baseline ABIGX, Koopman IG.
- Expected IG averages16 complete IG maps using normal training backgrounds. Guided IG uses bounded low-absolute-gradient coordinate updates. IDGI redistributes output increments proportional to squared gradients.
- DeepLIFT is a transparent explicit-GRU implementation: unary Rescale rules and symmetric reference-aware product rules for gates. It is not assumed identical to an off-the-shelf package's unsupported fused-GRU handler. Tests verify forward equivalence and contribution completeness; identify this implementation variant in publication.
- ABIGX adapts official PGD feature-center reconstruction to the shared64-dimensional GRU representation:1000 normalized-gradient steps, step0.02, L2 radius2000, official-style clip[-1000,1000]. Baseline normality success is reported. It is a windowed CNN-GRU adaptation, not a claim of reproducing the paper's original tabular classifier or numerical results. Failed AFR baselines are not silently replaced.
- Attribution target: predicted-fault logit. ADD/DEL target: same predicted-fault Softmax probability. Ground-truth mapping: ABIGX Table B.2, explicitly copied into core.py.
- Channel scores are sum of absolute time-point contributions, normalized to unit sum for SUM. Correctness-AUC is root-vs-nonroot ROC-AUC, not classifier ROC-AUC. SUM is normalized contribution on roots.
- ADD/DEL replace complete original20-point channels between the same real normal/fault endpoints,33 replacements plus initial state, trapezoidal integral over replaced fraction. No latent-space replacements or path confidence averages.
- Compute each trajectory metric, average within each fault, then macro-average14 faults, then mean/sample-std across independent seeds. Keep classification errors, report target-vs-true fault and explanation coverage. Paired comparison uses the identical successfully explained trajectory set across methods.

## Execution and outputs

On the new RTX5060 computer, use the repository-root `environment.yml` (Python3.11, PyTorch2.7.1 CUDA12.8) and activate `koopman_xai`. The old computer used `D:/App/Miniconda/myenvs/multitask_project/python.exe` with PyTorch2.5.1 CUDA12.4; that historical environment is not the RTX5060 recommendation. Read `../TECHNICAL_HANDOFF.md` for the complete experiment protocol and migration state. Run the following commands from this `Original_structure` directory, or use `python start_resume.py` from the repository root after restoring the transfer.

```powershell
python train.py --audit
python train.py --self-check
python explain.py --self-check
python run_experiments.py
```

The initial seed42 trainer was launched separately; the background queue adopts its PID with `run_experiments.py --adopt-pid PID`. Do not launch a second queue while `Results/pipeline.lock` exists. The queue stops and records errors; it never silently skips a failed training or explanation.

`Results/pipeline_status.json` records queue state. Initial logs are `Results/seed42_training.*.log`; subsequent subprocess logs are `Results/logs`. Each `Results/seed_*/status.json` tracks training progress. Checkpoints and epoch-end resume states are per seed. `z_*/` contains dimension candidates, K matrices, ridge selection and diagnostics.

`explanations/Fault_*/run_*/` contains endpoints, paths, feature contributions, ADD/DEL curves, diagnostics and per-method metrics. CSV files are UTF-8 with BOM for Origin. `Summary/` contains method comparison and ablation mean/std CSVs, seed-level and clearly labeled descriptive trajectory-level boxplots (PNG/SVG), classification confusion matrices and signed/absolute feature plots. Signed contributions are averaged directly; absolute contributions are averaged AFTER taking absolute values, avoiding cancellation. Partial summaries are marked incomplete.

Full data and five seeds are substantially more expensive than historical20-window subsampling. Do not report training as completed until the queue state is complete. Closing/suspending the computer interrupts progress; the runner can resume epoch-end training and completed explanation units. A seed finished training is not equivalent to the whole five-seed experiment being finished.
