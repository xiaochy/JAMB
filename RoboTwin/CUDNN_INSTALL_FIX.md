# Fix: `ImportError: libcudnn.so.9` during `bash script/_install.sh`

## Symptom

`script/_install.sh` fails while building `pytorch3d` and `curobo`, both during
the `pip install ... --no-build-isolation` step. The build's metadata-generation
phase does `import torch` and crashes:

```
File ".../site-packages/torch/__init__.py", line 290, in <module>
    from torch._C import *  # noqa: F403
ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory
```

Running `python -c "import torch"` directly in the activated `RoboTwin` conda
env reproduces the same error — it's not a build-tool issue, torch itself can't
load.

## Root cause

`nvidia-cudnn-cu12` was installed once with `pip install --user`, which puts it
in `~/.local/lib/python3.10/site-packages` (user-site). Because user-site
packages are visible from *every* Python environment (`site.ENABLE_USER_SITE`
defaults to `True`), `pip install` inside the `RoboTwin` conda env saw cuDNN as
"already satisfied" and never copied it into the env's own `site-packages`.

Torch's compiled extension only searches for CUDA libraries (`libcudnn.so.9`,
`libcublas.so`, etc.) relative to its own `site-packages/torch` directory — it
does not look in user-site. So even though `pip show nvidia-cudnn-cu12`
reports the package as installed, torch can't find the actual `.so` file.

Diagnose with:

```bash
pip show nvidia-cudnn-cu12        # check "Location:" — should be inside the env, not ~/.local
find ~/.local/lib -iname "libcudnn.so*" 2>/dev/null   # non-empty output = the package leaked into user-site
```

## Fix

```bash
conda activate RoboTwin
pip uninstall -y nvidia-cudnn-cu12          # removes the user-site copy
pip install nvidia-cudnn-cu12==9.1.0.70     # reinstalls scoped to the env (match torch's pinned version)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Once `import torch` succeeds, re-run just the steps that previously failed —
no need to restart the whole install script:

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

cd envs/curobo   # repo is already cloned from the first run; re-cloning is unnecessary
pip install -e . --no-build-isolation
cd ../..
```

## General advice for this class of problem

If a *different* package later fails the same way (`ImportError: lib*.so*: cannot
open shared object file`), the underlying cause is almost always one of:

1. **A dependency leaked into user-site (`~/.local`).** Any `pip install --user`
   run outside the conda env — or even an old global `pip install` from before
   the env existed — stays visible inside the env and silently satisfies
   requirements without actually being usable by it. Check with:
   ```bash
   pip show <package>          # look at "Location:" — must be under the conda env path
   python -c "import site; print(site.ENABLE_USER_SITE, site.getusersitepackages())"
   ```
   Fix by uninstalling the user-site copy and reinstalling inside the active env
   (as above), or by running pip with `PYTHONNOUSERSITE=1 pip install ...` to
   force it to ignore user-site when resolving "already satisfied" packages.

2. **A torch/CUDA version mismatch.** The `nvidia-*-cu12` wheels (cublas, cudnn,
   nccl, etc.) are pinned to specific versions per torch release. If you see this
   error after manually upgrading/downgrading torch, reinstall torch via its
   official index so pip pulls matching `nvidia-*-cu12` versions automatically,
   rather than installing torch and CUDA libs separately:
   ```bash
   pip install torch==<version> --index-url https://download.pytorch.org/whl/cu121
   ```

3. **System CUDA libraries shadowing the pip-installed ones**, or a
   stale `LD_LIBRARY_PATH` pointing at a different CUDA toolkit. Check
   `echo $LD_LIBRARY_PATH` and `ldconfig -p | grep libcudnn` if the above two
   don't explain it — system-wide installs can take priority over the
   site-packages copy depending on load order.

**Before re-running the full install script after any such fix:** confirm
`python -c "import torch"` succeeds standalone first. Both `pytorch3d` and
`curobo` (and likely other source-built deps in this repo) `import torch`
inside their `setup.py`/`pyproject.toml` build backend, so a broken torch
import will always surface as a confusing build-metadata error in those steps
rather than a clear torch error — that indirection is what made this issue
non-obvious initially.
