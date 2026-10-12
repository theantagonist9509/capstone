# Running training on Kaggle

The capstone training/traversal scripts (`runs/shapes/*.py`) are the source of
truth. They are annotated with `# %%` cell markers so they can be converted
into Kaggle notebooks, pushed with the Kaggle CLI, and executed on a GPU —
with plots rendered inline in the notebook.

```
annotated .py  ──tools/py2nb.py──▶  .ipynb  ──kaggle kernels push──▶  run
                                      ▲                                   │
                        local source of truth              kernels output │
                                                        (plots, checkpoints)
```

## One-time setup

- Python env: `source ~/dev/test/.venv/bin/activate` (ships `kaggle` CLI 2.2.4).
- Auth: `~/.kaggle/kaggle.json` (already configured).
- The `muplus2sigma/capstone` kernel on kaggle.com has:
  - **GPU** enabled,
  - **Internet** enabled (the bootstrap shallow-clones this repo at run time),
  - the **`custom-isic2018` dataset** attached (`dataset_sources`),
  - a **package-manager attachment** providing `pytorch-msssim`.

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

```bash
python tools/py2nb.py runs/shapes/train_classifier.py            # full run
python tools/py2nb.py runs/shapes/train_classifier.py --smoke    # smoke test
python tools/py2nb.py runs/shapes/traverse.py -o /tmp/x.ipynb    # custom output
```

The generated `.ipynb` (default: alongside the `.py`) contains:

1. a markdown header,
2. a **bootstrap cell** (Kaggle setup, see below),
3. the script's own `# %%` cells in order.

`py2nb` bakes the script's CLI argv into the bootstrap cell (`--smoke` etc.),
since notebooks have no command line.

### What the bootstrap cell does

- line-buffers stdout/stderr so prints aren't stuck in the 8KB block buffer,
- `%matplotlib inline` so figures render in the notebook,
- shallow-clones the repo into `/kaggle/working/capstone` and prints the
  commit SHA (the notebook's code cells are self-contained; the clone is
  needed for the repo's *modules* — `datasets.py`, `models.py`, …),
- finds the attached dataset by walking `/kaggle/input` for its class folders
  (the mount path is nested as
  `/kaggle/input/datasets/<owner>/<slug>/…` and has changed before — the walk
  makes the bootstrap layout-agnostic) and symlinks it to
  `<clone>/dataset/<name>`,
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
> and remains fine when inline plots aren't needed.

A push folder holds exactly two files: `kernel-metadata.json` + the generated
`.ipynb`. Example metadata:

```json
{
  "id": "muplus2sigma/capstone",
  "title": "capstone",
  "code_file": "train_classifier.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_tpu": "false",
  "enable_internet": "true",
  "machine_shape": "",
  "dataset_sources": ["muplus2sigma/custom-isic2018"],
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
kaggle kernels status muplus2sigma/capstone    # QUEUED / RUNNING / COMPLETE / ERROR
kaggle kernels logs  muplus2sigma/capstone     # full stdout+stderr (after run ends)
kaggle kernels files muplus2sigma/capstone     # list /kaggle/working artifacts
kaggle kernels output muplus2sigma/capstone -p out/   # download artifacts + logs
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
kaggle kernels output muplus2sigma/capstone -p out/          # everything
kaggle kernels output muplus2sigma/capstone -p out/ \
    --file-pattern '.*\.png'                                 # just the plots
kaggle kernels output muplus2sigma/capstone -p out/ \
    --file-pattern '.*conv_ae.*epoch_.*\.pth'                # AE checkpoints
```

`kaggle kernels output` also drops a `<kernel>.log` copy of the run's logs
next to the downloaded files. Checkpoints land under
`out/capstone/checkpoints/...` (paths inside `/kaggle/working` are
preserved); copy them into the local `checkpoints/` tree as needed for the
traversal notebook.

## Smoke-first workflow

1. `python tools/py2nb.py runs/shapes/<script>.py --smoke`
2. push; wait for `COMPLETE`; `kaggle kernels logs` — look for
   `[bootstrap] GPU: Tesla T4` and the `[SMOKE] …` marker lines.
3. If green, push the full (non-`--smoke`) notebook and let it train.

## Script → notebook gotchas (things already handled — don't redo)

- `sys.argv` inside a kernel is ipykernel's, not ours — argparse needs the
  bootstrap's `sys.argv` reset.
- `__file__` inside a kernel points at `/kaggle/src/script.py` (script
  kernels) or is meaningless (notebooks) — the bootstrap re-points it into
  the clone.
- Repo modules must be imported *after* the bootstrap puts the clone on
  `sys.path`; HF's `datasets` package otherwise shadows our `datasets.py`.
- stdout is block-buffered in non-TTY runs; the bootstrap's
  `reconfigure(line_buffering=True)` keeps prints from being lost on hangs.
