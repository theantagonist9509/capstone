# Running training on Kaggle

The capstone training/traversal scripts (`runs/shapes/*.py`) are the source of
truth. They are annotated with `# %%` cell markers (jupytext's light format),
so `kaggle.py` can convert them into Kaggle notebooks with **jupytext**, push
them with the Kaggle API, and execute them on a GPU — with plots rendered
inline in the notebook.

```
annotated .py  ──kaggle.py (jupytext)──▶  .ipynb  ──push──▶  run
                                            ▲                     │
                           local source of truth          kernels output
                                                 (plots, checkpoints)
```

## One-time setup

- Python env: `source ~/dev/test/.venv/bin/activate` (ships `kaggle` CLI
  2.2.4 and `jupytext` 1.19 — `pip install jupytext` if missing).
- Auth: `~/.kaggle/kaggle.json` (already configured).
- The `muplus2sigma/btp-26` kernel on kaggle.com has:
  - **GPU** enabled,
  - **Internet** enabled (the bootstrap shallow-clones this repo at run time),
  - the **`smeschke/four-shapes` dataset** attached (`dataset_sources` —
    push also sets this; its class folders live under `shapes/`, which the
    bootstrap's walk finds and copies into `<clone>/dataset/shapes`),
  - a **package-manager attachment** providing `pytorch-msssim`,
  - **2× T4 GPUs** — the run scripts auto-wrap the model in `nn.DataParallel`
    and scale `BATCH_SIZE` by `torch.cuda.device_count()`; checkpoints are
    always saved with unwrapped (no `module.` prefix) weights, so single- and
    multi-GPU checkpoints are interchangeable.
- **Fresh kernels**: a push can create a missing kernel, but it can only set
  GPU/internet/dataset sources — the **package-manager attachment**
  (`pytorch-msssim`) must be added once in the kernel's web-UI settings, or
  the notebooks die on `import pytorch_msssim`.

**Do not `pip install` anything from within the scripts/notebooks** — the
kernel's package-manager attachment already provides every dependency the
scripts need (`torch`, `torchvision`, `scikit-learn`, `tqdm`, `matplotlib`,
`pytorch-msssim`, …). Installing again just wastes run time. If a genuinely
new dependency is ever added to the repo, attach it once in the kernel's
web-UI settings instead of pip-installing in code.

## Annotating a script

Cell markers follow the jupytext convention:

```python
# %% [markdown]
# ## Section title          ← rendered as markdown in the notebook

# %%
import torch                ← plain code cell
```

The scripts must keep working as plain `python path/to/script.py` runs —
`# %%` lines are comments, so this is free. Notebook-specific display is
handled by `utils.maybe_show(fig)`: plots are shown inline under
IPython/Jupyter and closed silently under a local Agg run.

## Converting to a notebook

`kaggle.py` splits the annotated script with **jupytext** (no hand-rolled
converter), then prepends the header + bootstrap cells (see below):

```bash
python kaggle.py runs/shapes/train_classifier.py            # full run
python kaggle.py runs/shapes/train_classifier.py --smoke    # smoke test
python kaggle.py runs/shapes/traverse.py -o /tmp/x.ipynb    # custom output
python kaggle.py runs/shapes/train_classifier.py --smoke --push-run     # upload + run
python kaggle.py runs/shapes/train_classifier.py --smoke --push-no-run  # upload, no run
```

`--push-run` / `--push-no-run` target `muplus2sigma/btp-26` by default
(`--kernel owner/slug` to override). `--push-no-run` uses Kaggle's
quick-save API (`KernelExecutionType.QUICK_SAVE`), which updates the kernel
without starting a session; `--push-run` starts a run like `kaggle kernels push`.
Both set GPU, internet, the `smeschke/four-shapes` dataset attachment, and a
title that matches the kernel slug.

The generated `.ipynb` (default: alongside the `.py`) contains:

1. a markdown header,
2. a **bootstrap cell** (Kaggle setup, see below),
3. the script's own `# %%` cells in order (parsed by jupytext).

`kaggle.py` bakes the script's CLI argv into the bootstrap cell (`--smoke`
etc.), since notebooks have no command line. A bare `jupytext --to ipynb`
would give you cells 1+3 but *not* the bootstrap — such a notebook cannot
run on Kaggle.

### What the bootstrap cell does

- `%matplotlib inline` so figures render in the notebook,
- **syncs the repo clone**: `/kaggle/working` persists between runs on a
  kernel, so an existing `/kaggle/working/capstone` is `fetch` +
  `reset --hard FETCH_HEAD`-ed to the latest `main` (a plain "skip clone if
  `.git` exists" check would keep running a stale version forever);
  untracked files like `checkpoints/` survive, and a non-git leftover dir is
  removed and re-cloned fresh. Prints the resulting commit SHA (the
  notebook's code cells are self-contained; the clone is needed for the
  repo's *modules* — `custom_datasets.py`, `models.py`, …),
- finds the attached dataset by walking `/kaggle/input` for its class folders
  (the mount path is nested as
  `/kaggle/input/datasets/<owner>/<slug>/…` and has changed before — the walk
  makes the bootstrap layout-agnostic) and copies it to
  `<clone>/dataset/<name>` (skipped when the destination already exists —
  reading through the `/kaggle/input` mount is slow; `/kaggle/working`
  persists between kernel runs),
- `sys.path.insert(0, ROOT)`, re-points `__file__` into the clone (so the
  scripts' `dirname(__file__)`-based repo-root math works unchanged), and
  sanitises `sys.argv` for `argparse`,
- prints the GPU name.

It does **not** pip-install anything (see setup note above).

## Pushing and running

> **Preferred path for notebooks: manual upload in the Kaggle GUI.** CLI-pushed
> *notebook*-type kernels (`kaggle kernels push` with `kernel_type: "notebook"`)
> wedged twice in `RUNNING` for 30–50+ min with zero execution/log output and
> had to be cancelled manually. Uploading the generated `.ipynb` through the
> website and running it there works normally. Generate the file, then use
> the GUI's upload/new-kernel flow (keep the kernel's GPU/internet/dataset/
> package-manager settings). CLI push of *script*-type kernels was reliable
> and remains fine when inline plots aren't needed. (`kaggle.py --push-no-run`
> — the quick-save API — has worked fine for staging notebook *sources*
> without starting a wedged run.)

A push folder holds exactly two files: `kernel-metadata.json` + the generated
`.ipynb`. Example metadata:

```json
{
  "id": "muplus2sigma/btp-26",
  "title": "btp-26",
  "code_file": "train_classifier.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_tpu": "false",
  "enable_internet": "true",
  "machine_shape": "",
  "dataset_sources": ["smeschke/four-shapes"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

Gotchas:

- **`title` must slugify to the slug in `id`** ("cli dep probe" silently
  created kernel `cli-dep-probe`, and status/logs against the wrong slug
  failed with a confusing `Permission 'kernels.get' was denied`).
- `kernel_type: "notebook"` — script kernels work too, but notebooks show
  plots/curves inline in the Kaggle GUI.
- The repo must be **pushed to GitHub before the run**: the bootstrap clones
  `main`, so uncommitted local changes are invisible to the kernel.

```bash
kaggle kernels push -p <push-folder>            # upload + start a run (new version)
kaggle kernels status muplus2sigma/btp-26       # QUEUED / RUNNING / COMPLETE / ERROR
kaggle kernels logs  muplus2sigma/btp-26        # full stdout+stderr (after run ends)
kaggle kernels files muplus2sigma/btp-26        # list /kaggle/working artifacts
kaggle kernels output muplus2sigma/btp-26 -p out/   # download artifacts + logs
```

## Observability — what to expect

- **`RUNNING` includes environment setup** (machine boot, dataset mount,
  package install): several minutes are normal before any of our code runs.
  A run that "ran in 38s" can show "running for 200s"+. Long `QUEUED`
  periods are GPU contention, not hangs. But `RUNNING` with **zero cell
  execution / logs for ~30+ min** (seen on CLI-pushed notebook kernels) is
  a wedge — cancel and use the GUI upload path above.
- **CLI logs appear only after the run ends** (`kaggle kernels logs`);
  the `logs -f` live-stream endpoint is flaky (500s). In the web GUI, logs
  may start appearing only several minutes in.
- Each push creates a new **version** of the same kernel; the website shows
  every version's executed notebook (with inline plots) under the kernel's
  Code tab — that's the "populated notebook". `kaggle kernels pull` fetches
  the *pushed source*, not the executed copy.
- Training writes per-epoch checkpoints under
  `/kaggle/working/capstone/checkpoints/...`, so partial progress survives
  in the output artifacts if a run is cancelled.

## Fetching results

```bash
kaggle kernels output muplus2sigma/btp-26 -p out/          # everything
kaggle kernels output muplus2sigma/btp-26 -p out/ \
    --file-pattern '.*\.png'                                 # just the plots
kaggle kernels output muplus2sigma/btp-26 -p out/ \
    --file-pattern '.*conv_ae.*epoch_.*\.pth'                # AE checkpoints
```

`kaggle kernels output` also drops a `<kernel>.log` copy of the run's logs
next to the downloaded files. Checkpoints land under
`out/capstone/checkpoints/...` (the clone dir inside `/kaggle/working` is
always named `capstone`, regardless of the kernel slug); copy them into the
local `checkpoints/` tree as needed for the traversal notebook.

## Smoke-first workflow

1. `python kaggle.py runs/shapes/<script>.py --smoke`
2. push; wait for `COMPLETE`; `kaggle kernels logs` — look for
   `[bootstrap] GPU: Tesla T4` and the `[SMOKE] …` marker lines.
3. If green, push the full (non-`--smoke`) notebook and let it train.

## Script → notebook gotchas (things already handled — don't redo)

- `sys.argv` inside a kernel is ipykernel's, not ours — argparse needs the
  bootstrap's `sys.argv` reset.
- `__file__` inside a kernel points at `/kaggle/src/script.py` (script
  kernels) or is meaningless (notebooks) — the bootstrap re-points it into
  the clone.
- Our dataset module is **`custom_datasets.py`** (renamed from `datasets.py`)
  so a plain `from custom_datasets import ...` can never collide with HF's
  installed `datasets` package — which otherwise shadows any repo module
  named `datasets`, regardless of `sys.path` order.
- The helper itself is named **`kaggle.py`** at the repo root, which shadows
  the installed `kaggle` package under `python kaggle.py` (sys.path[0] is the
  script's dir) — its push code imports the real package with the repo root
  temporarily removed from `sys.path`.
