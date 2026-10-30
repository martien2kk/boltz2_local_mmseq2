# Local MSA generation with MMseqs2

`local_mmseqs.py` builds Boltz MSAs on your own machine with MMseqs2 (GPU) instead of the public ColabFold server. It searches the AlphaFold3 sequence databases and writes one Boltz MSA CSV per protein sequence. Your sequences never leave the machine.

## Why use it

- **Privacy.** `--use_msa_server` sends sequences to `api.colabfold.com`. A local search sends nothing.
- **Speed at scale.** Many sequences are searched together in one GPU pass per database, and every result is cached. A protein used in 100 complexes is searched once.
- **Same MSAs as AlphaFold3.** The search settings match [AlphaFast](https://github.com/RomeroLab/alphafast)'s MMseqs2-GPU defaults, so Boltz sees the same evolutionary information as an AF3 run on the same databases.

## Requirements

### MMseqs2 with GPU support

Any recent MMseqs2 build with `--gpu` support works. Check with:

```bash
mmseqs search -h | grep -- "--gpu "
```

Options:

- Download the GPU build (`mmseqs-linux-gpu.tar.gz`) from the [MMseqs2](https://github.com/soedinglab/MMseqs2) releases.
- If you use AlphaFast, copy the binary out of its container. It is self-contained:
  ```bash
  singularity exec alphafast.sif cat /usr/local/bin/mmseqs > $CONDA_PREFIX/bin/mmseqs
  chmod +x $CONDA_PREFIX/bin/mmseqs
  ```

Use the same MMseqs2 build that created the databases where possible. Padded database formats can change between versions.

### Databases

One directory containing these MMseqs2 padded databases:

| Database | Files | Used for |
|---|---|---|
| UniRef90 | `uniref90_padded*` | Unpaired MSA |
| Small BFD | `small_bfd_padded*` | Unpaired MSA |
| MGnify | `mgnify_padded*` | Unpaired MSA |
| UniProt | `uniprot_padded*` | Species-keyed rows for chain pairing |

AlphaFast's `scripts/setup_databases.sh` builds exactly this layout (its `mmseqs/` folder, about 390 GB). The search stops with an error naming any missing database.

## Usage

### From the command line

```bash
boltz predict job.yaml --use_local_msa --mmseqs_db_dir /opt/AlphaFast/mmseqs --out_dir results/
```

This works like `--use_msa_server`: every protein without an `msa:` field gets one generated, but locally. All such sequences across all inputs are searched together before preprocessing starts. Each job's `msa/` output folder gets a copy of its CSVs.

| Option | Default | Meaning |
|---|---|---|
| `--use_local_msa` | off | Turn on local MSA generation |
| `--mmseqs_db_dir` | `$BOLTZ_MMSEQS_DB_DIR` | Folder with the four padded databases (required) |
| `--mmseqs_binary` | `mmseqs` | MMseqs2 binary name or path |
| `--msa_cache_dir` | `<cache>/msa_cache` (`~/.boltz/msa_cache`) | Where finished CSVs are stored |
| `--mmseqs_sensitivity` | 7.5 | MMseqs2 `-s` |
| `--max_paired_msa_rows` | 4096 | Cap on taxonomy-keyed rows (see below) |

`--use_local_msa` cannot be combined with `--use_msa_server`. Setting the database folder once in your shell saves typing:

```bash
export BOLTZ_MMSEQS_DB_DIR=/opt/AlphaFast/mmseqs
boltz predict job.yaml --use_local_msa --out_dir results/
```

### From Python

```python
from boltz.data.msa.local_mmseqs import search_sequences

paths = search_sequences(
    sequences=["MKTAYIAKQR...", "GSHMSLFDFF..."],
    db_dir="/opt/AlphaFast/mmseqs",
    cache_dir="~/.boltz/msa_cache",
)
# {"MKTAYIAKQR...": PosixPath(".../<hash>.csv"), ...}
```

Each returned CSV can be used directly in a Boltz YAML:

```yaml
sequences:
  - protein:
      id: A
      sequence: MKTAYIAKQR...
      msa: /home/me/.boltz/msa_cache/<hash>.csv
```

`mmseqs` must be on `PATH` (for example, with the conda env activated), or passed as `mmseqs="/path/to/mmseqs"`.

## How it works

```
sequences ─► skip any already in the cache
          ─► mmseqs createdb                        (all new sequences, one query DB)
          ─► for each of uniref90, small_bfd, mgnify, uniprot:
               mmseqs search -a --gpu 1             (find and align hits)
               mmseqs result2msa --msa-format-mode 2 (aligned FASTA, headers kept)
               mmseqs unpackdb                      (one file per query)
          ─► combine rows per sequence ─► <cache>/<sha256>.csv
```

`--msa-format-mode 2` writes every row at the query's length, with insertions removed. Boltz reads that directly, so no further alignment step is needed.

### Search settings

| Setting | Value | Argument |
|---|---|---|
| Sensitivity (`-s`) | 7.5 | `sensitivity` |
| E-value (`-e`) | 1e-4 | `e_value` |
| Max hits, UniRef90 | 10,000 | `UNPAIRED_DATABASES` |
| Max hits, Small BFD | 5,000 | `UNPAIRED_DATABASES` |
| Max hits, MGnify | 5,000 | `UNPAIRED_DATABASES` |
| Max hits, UniProt | 50,000 | `PAIRING_MAX_HITS` |

These values match AlphaFast's defaults. The code is an independent implementation; no AlphaFast code is included.

## The MSA CSV

Boltz's MSA CSV has two columns (rows below are shortened and illustrative):

```
key,sequence
-1,VSKKPSLSVQPGPIVAPEETLTLQ...      ← query, always first
9606,VSKKPSLSVQPGPIVAPEETLTLQ...    ← UniProt hit, key = NCBI taxonomy id (OX=)
10090,VSKRPSLSVQPGPLV-PEETLTLQ...   ← UniProt hit from another species
-1,-SKKPSLSVHPGPIVAPGESVTLQ...      ← UniRef90 / Small BFD / MGnify hit, unpaired
```

### How pairing works

When a complex has several protein chains, Boltz lines up rows that share the same `key` across chains, so a mouse homolog of chain A sits next to a mouse homolog of chain B. Because pairing happens when Boltz runs, not when the CSV is written, **each sequence's CSV works with any partner**. That is what makes per-sequence caching possible.

Rows with key `-1` are never paired. For single-chain jobs, keyed rows simply act as extra unpaired rows.

### Row order and limits

1. The query.
2. UniProt hits with a taxonomy id, best first, up to `max_paired_rows` (default 4,096).
3. UniRef90, then Small BFD, then MGnify hits, best first, until `max_rows` (default 16,384) is reached.

Duplicate sequences are dropped, keeping the first copy. Boltz itself keeps only the first copy of a duplicate, so keyed rows must come before their unpaired twins.

**Why cap the keyed rows?** At prediction time Boltz reads only the first `--max_msa_seqs` rows (default 8,192). UniProt can return up to 50,000 hits, and without a cap they would fill that whole window and push out every unpaired hit. With the default cap, at most half of the default window is keyed rows.

## Caching

- Each CSV is named by the first 24 hex characters of the sequence's SHA-256. Identical sequences share one file, whatever chain or job they come from.
- A CSV is written to a temporary name and renamed when complete, so an interrupted run never leaves a partial file in the cache.
- To redo a search (for example, after changing settings or updating databases), delete the CSV, or the whole cache folder.

## Example: PDB 7KFK (LILRB1 + RIFIN)

Run on one RTX PRO 4500 Blackwell (32 GB) against the AlphaFast databases:

| Chain | Length | Rows | Keyed rows |
|---|---|---|---|
| LILRB1 D1–D2 | 233 | 8,310 | 4,096 (capped) |
| RIFIN | 151 | 296 | 282 |

The search took 236 s for both sequences. A repeat call returned instantly from the cache. The RIFIN MSA was identical in size to AlphaFast's. The LILRB1 MSA differs only by the keyed-row cap.

## Limitations

- **Proteins only.** Boltz does not use MSAs for DNA, RNA or ligands.
- **No templates.** Template search is not included.
- **GPU-oriented.** The padded databases are built for GPU search. `gpu=False` exists but has not been tested on them.
- **Species pairing assumes co-evolution.** For complexes whose chains come from different organisms (for example, host–pathogen), few rows share a key. Compare runs with and without pairing.
- **Disk and memory.** The databases are about 390 GB. GPU search streams them from disk, so fast local storage (NVMe) helps a lot.
