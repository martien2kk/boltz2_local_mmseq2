# Boltz-2 with local MMseqs2

A fork of [Boltz](https://github.com/jwohlwend/boltz) 2.2.1 that adds one feature: **generating MSAs locally with MMseqs2 on the GPU**, instead of sending sequences to the public ColabFold server.

Boltz needs a multiple sequence alignment for every protein chain, and building those alignments is usually the slowest part of a prediction. Upstream gives you two ways to get them: precompute them yourself and pass them in, or use `--use_msa_server` and let the public ColabFold server build them, which means queueing behind other users and sending your sequences off your machine.

This fork adds a third. Point it at your own copies of the databases and it searches them with MMseqs2 on the GPU, as part of the usual prediction run:

```
boltz predict input_path --use_local_msa --mmseqs_db_dir /path/to/mmseqs_databases
```

- **Private:** sequences never leave your machine.
- **Batched:** all proteins across all inputs are deduplicated and searched together, so a protein shared by many complexes is searched once. Results live in a temporary folder for the run and are deleted afterwards, so a run depends only on its inputs and the databases.
- **Same MSAs as AlphaFold3:** it searches UniRef90, Small BFD, MGnify and UniProt with the same settings as [AlphaFast](https://github.com/RomeroLab/alphafast). UniProt hits are keyed by species so Boltz can pair chains.

Options and details are in [src/boltz/data/msa/local_mmseqs.md](src/boltz/data/msa/local_mmseqs.md). Everything else is unchanged: for the model itself, inference options, affinity prediction, training, evaluation, licensing and citation, see the [upstream Boltz README](https://github.com/jwohlwend/boltz/blob/main/README.md).

## Requirements

**1. MMseqs2 with GPU support.** Either:

- Download the GPU build (`mmseqs-linux-gpu.tar.gz`) from the [MMseqs2 releases](https://github.com/soedinglab/MMseqs2/releases) and put `mmseqs` on your `PATH`.
- Or, if you use [AlphaFast](https://github.com/RomeroLab/alphafast), copy the binary out of its official container. It is self-contained, and it is the build that created AlphaFast's databases:

  ```
  singularity pull alphafast.sif docker://romerolabduke/alphafast:latest
  singularity exec alphafast.sif cat /usr/local/bin/mmseqs > $CONDA_PREFIX/bin/mmseqs
  chmod +x $CONDA_PREFIX/bin/mmseqs
  mmseqs search -h | grep -- "--gpu "    # should list the --gpu option
  ```

Use the same MMseqs2 build that created your databases where possible, since the padded database format can change between versions. This fork was tested with MMseqs2 `d45e0c44404715475da3e1f06df6529d4c83e49e` from AlphaFast's container.

**2. The databases.** UniRef90, Small BFD, MGnify and UniProt in MMseqs2 padded format, in one folder (about 390 GB). AlphaFast's setup script downloads them prebuilt:

```
git clone https://github.com/RomeroLab/alphafast.git
./alphafast/scripts/setup_databases.sh /path/to/databases --protein-only
boltz predict input_path --use_local_msa --mmseqs_db_dir /path/to/databases/mmseqs
```

The script needs about 800 GB of free disk space while it runs. See [AlphaFast's database instructions](https://github.com/RomeroLab/alphafast#step-3-download-and-convert-databases) for options, including building the databases from source.

**Third-party software and data.** This repository contains only Boltz and the local MSA code, under the MIT license. MMseqs2, the AlphaFast container and scripts, and the sequence databases are not included or redistributed here. Each is downloaded from its own source and has its own license and terms. Notably, AlphaFast is licensed [CC-BY-NC-SA 4.0](https://github.com/RomeroLab/alphafast/blob/main/LICENSE) (non-commercial), so check those terms before using its container or scripts for commercial work.

## Installing this fork

This fork is published under the package name `boltz`, the same name as upstream. Pip keeps only one package per name, so **this fork replaces upstream Boltz rather than installing alongside it**. Do not install both. If you already have upstream Boltz, see [Replacing an existing Boltz install](#replacing-an-existing-boltz-install) below.

**1. Create an environment.** Any Python 3.10, 3.11 or 3.12 environment works; the example below uses conda. A dedicated environment is recommended, since installing upstream Boltz into the same one later would overwrite this fork.

```
conda create -n boltz2 python=3.11
conda activate boltz2
```

**2. Install PyTorch.** On NVIDIA Blackwell GPUs (RTX 50 series, RTX PRO 6000/4500 Blackwell, B200), install a build for CUDA 12.8 or newer **before** the next step, so it isn't resolved to one that lacks Blackwell support:

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

On older GPUs, skip this step and let the next one install PyTorch.

**3. Install the fork.**

```
git clone https://github.com/martien2kk/boltz2_local_mmseq2.git
cd boltz2_local_mmseq2
pip install -e .[cuda]
```

This is an editable install, so the clone directory *is* the installed package: edits to `src/boltz/` take effect on the next run, but moving or deleting the clone breaks the environment. For an install that does not depend on a local clone, use `pip install "boltz[cuda] @ git+https://github.com/martien2kk/boltz2_local_mmseq2.git"` instead.

**4. Put the MMseqs2 GPU binary on `PATH`**, as described under [Requirements](#requirements). Installing this package does not provide it. `$CONDA_PREFIX/bin` is a convenient location, since it is on `PATH` whenever the environment is active.

**5. Check the install.**

```
pip show boltz | grep -E "Version|Editable"   # Version: 2.2.1+localmsa
boltz predict --help | grep use_local_msa     # the option is present
mmseqs search -h | grep -- "--gpu "           # GPU build of MMseqs2
python -m pytest tests/test_local_msa.py
```

Set `BOLTZ_MMSEQS_DB_DIR` before the pytest to also run a real search against your databases.

Batch jobs that do not activate the environment need `PATH` set so `mmseqs` can be found, for example `export PATH="$(dirname "$BOLTZ_BIN"):$PATH"`.

## Replacing an existing Boltz install

If `pip show boltz` reports a version without the `+localmsa` suffix, that environment has upstream Boltz. Either make a new environment with the steps above, which leaves the existing one untouched, or replace it in place:

```
conda activate <existing-env>
pip uninstall boltz
git clone https://github.com/martien2kk/boltz2_local_mmseq2.git
cd boltz2_local_mmseq2
pip install -e .[cuda]
```

Installing over the old version without uninstalling usually works, but uninstalling first avoids leaving behind files from the previous install.

Downloaded model weights live in `~/.boltz` and are shared by both, so they are not downloaded again. Existing inputs, scripts and outputs are unaffected: everything except the added options works exactly as in upstream Boltz.

Afterwards, `pip install boltz` in that environment would replace this fork with upstream again and remove `--use_local_msa`. If that happens, reinstall with the commands above.
