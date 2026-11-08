# Local MSA generation with MMseqs2

`local_mmseqs.py` builds Boltz MSAs on your own machine with MMseqs2 (GPU) instead of the public ColabFold server. It searches the AlphaFold3 sequence databases and writes one Boltz MSA CSV per protein sequence. Your sequences never leave the machine.

## Why use it

- **Privacy.** `--use_msa_server` sends sequences to `api.colabfold.com`. A local search sends nothing.
- **Speed at scale.** Many sequences are searched together in one GPU pass per database, and identical sequences are collapsed first. A protein used in 100 complexes in one run is searched once.
- **AF3-style search settings.** The databases, sensitivity, E-value and per-database hit limits follow [AlphaFast](https://github.com/RomeroLab/alphafast)'s MMseqs2-GPU defaults, so the hits come from the same searches an AF3 run on those databases would do. How rows are selected from those hits, and how chains are paired, are Boltz's own and are not AF3's — see [Limitations](#limitations).

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

This works like `--use_msa_server`: every protein without an `msa:` field gets one generated, but locally. All such sequences across all inputs are deduplicated and searched together before preprocessing starts. Each job's `msa/` output folder gets a copy of its CSVs.

The search results live in a temporary folder for the duration of the run and are deleted when it ends, so a run depends only on its inputs and the databases. To reuse MSAs across runs, keep the CSVs from a job's `msa/` folder and reference them with `msa:` in your YAML, or call `search_sequences` yourself with a lasting `cache_dir` (see [From Python](#from-python)).

| Option | Default | Meaning |
|---|---|---|
| `--use_local_msa` | off | Turn on local MSA generation |
| `--mmseqs_db_dir` | `$BOLTZ_MMSEQS_DB_DIR` | Folder with the four padded databases (required) |
| `--mmseqs_binary` | `mmseqs` | MMseqs2 binary name or path |
| `--mmseqs_sensitivity` | 7.5 | MMseqs2 `-s` |
| `--mmseqs_tmp_dir` | system temp (`$TMPDIR`, usually `/tmp`) | Where MMseqs2 intermediates go |
| `--msa_gpus` | 1 | GPUs to spread the database searches over (at most 4) |
| `--msa_batch_size` | no limit | Search in groups of this many sequences (see below) |
| `--max_paired_msa_rows` | 4096 | Cap on taxonomy-keyed rows, must be below `--max_msa_seqs` (see below) |

Intermediate MMseqs2 files can be large — the search results for 50,000 UniProt hits per query, before they are reduced to CSV rows. On a cluster, the default temp directory is often small or on a network filesystem, so point `--mmseqs_tmp_dir` (or `BOLTZ_MMSEQS_TMP_DIR`) at node-local scratch:

```bash
boltz predict job.yaml --use_local_msa --mmseqs_tmp_dir "$SLURM_TMPDIR" --out_dir results/
```

Free space there is checked before the search starts and a warning is logged if it looks too small. The estimate is rough, so it warns rather than stops; each database also logs the size of the intermediates it actually produced, which is the number to plan with.

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
    cache_dir="~/my_msas",
)
# {"MKTAYIAKQR...": PosixPath(".../<hash>.csv"), ...}
```

Unlike `boltz predict`, this writes to the directory you name and leaves the CSVs there. Call it again with the same `cache_dir` and sequences already present are returned without being searched again — this is how to build MSAs once and reuse them.

Each returned CSV can be used directly in a Boltz YAML:

```yaml
sequences:
  - protein:
      id: A
      sequence: MKTAYIAKQR...
      msa: /home/me/my_msas/<hash>.csv
```

`mmseqs` must be on `PATH` (for example, with the conda env activated), or passed as `mmseqs="/path/to/mmseqs"`.

## How it works

```
sequences ─► drop duplicates, and any already written in this run
          ─► mmseqs createdb                        (all new sequences, one query DB)
          ─► for each of uniref90, mgnify, uniprot, small_bfd:
               mmseqs search -a --gpu 1             (one database at a time)
             and, in a worker thread while the next database searches:
               mmseqs result2msa --msa-format-mode 2 (aligned FASTA, headers kept)
               mmseqs unpackdb                      (one file per query)
               keep only the hits that can become rows, then delete the rest
          ─► combine rows per sequence ─► <cache_dir>/<sha256>.csv
```

`--msa-format-mode 2` writes every row at the query's length, with insertions removed. Boltz reads that directly, so no further alignment step is needed.

By default the searches run one database at a time, since they share one GPU, but each database's post-processing is CPU work and runs in a worker thread while the next search is underway.

`--msa_gpus N` runs N searches at once instead, one per GPU, up to one per database. There are four databases, so asking for more than four is clamped to four and warns: beyond that the extra GPUs would sit idle while the threads were still divided by the number asked for. A search holds a GPU only while it runs, and `--threads` is divided between them. The devices come from `CUDA_VISIBLE_DEVICES` where a scheduler has set it, so an allocation of GPUs 2 and 3 uses those and not 0 and 1. Whether this helps depends on where the databases live: a search streams its whole database off disk, so several at once can contend for bandwidth rather than go faster, especially on shared storage. The per-database log lines name the GPU and split search from post-processing, so one run tells you which it is — if search times grow roughly in step with the GPU count, drop back to 1. The databases are searched in the order above, chosen so the one needing the least post-processing comes last, where nothing follows it to overlap with. Rows are merged in a fixed order independent of this, so the MSAs do not depend on the search schedule. On one RTX PRO 4500 Blackwell, 7 sequences took 337.9 s with the overlap against 372.9 s without, and the MSAs were byte-identical.

### Memory, and when to batch

All sequences are searched together, so what the search holds in memory grows with the number of sequences in the run. Hits are reduced to what can still become CSV rows as each query's file is read, which is what keeps that growth modest:

- UniProt returns up to 50,000 hits per query, but only taxonomy-keyed ones become rows and only the first `--max_paired_msa_rows` (4,096) of those. The rest are dropped as the file is read rather than held until the end.
- Unpaired hits are kept as bare sequences; their headers are never used, since every row built from them is keyed `-1`.
- Repeats of a sequence already kept are dropped at the same point.
- Each database's intermediates are deleted as soon as its hits have been read, so temporary space holds what the searches still running need, not every database's output at once.

None of this changes the CSVs — what is dropped is exactly what row building would have ignored. The test suite checks that property directly.

If a run is still too large, `--msa_batch_size N` searches the sequences in groups of N:

```bash
boltz predict inputs/ --use_local_msa --msa_batch_size 200 --out_dir results/
```

Reach for this last. Every group reads all four databases again, and a search streams its whole database off disk, so two groups cost close to two full searches. Groups do share one cache directory, so a sequence searched in an earlier group is not searched again in a later one.

### GPU memory

The prefilter streams a padded database through device memory, so a search needs most of the card to itself. Two searches on one GPU, or a search alongside a loaded model, make the prefilter fail with a bare `CUDA error: out of memory` from inside libmarv — minutes into the run, naming neither the device nor what else was on it.

Before the first search, each device it will use is checked with `nvidia-smi`, and a device with less than 85% of its memory free is refused by name, along with the process holding it:

```
RuntimeError: GPU 0 has only 22937 of 32623 MiB free (70%). MMseqs2 streams a
padded database through device memory and needs most of the card, so the search
would likely die part-way with a CUDA out-of-memory error. In use by: pid 740595
(python) holding 8446 MiB. ...
```

A desktop session holding a few hundred MiB is well inside the threshold. This is a snapshot, not a reservation: a process that starts just after the check can still take the memory, so a scheduler remains the only real guard. Set `BOLTZ_SKIP_GPU_CHECK=1` to search anyway, and note that a job scheduler's GPU allocation does not stop a process started outside it from sharing the card.

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

**This is not how `--use_msa_server` pairs.** There, ColabFold pairs the chains of one complex on the server and the CSV's `key` is a *row number*, so row 5 of chain A is paired with row 5 of chain B and the CSV only means anything alongside the partners it was built with. Here the key is a taxonomy id and the pairing itself is left to Boltz. Both end up in the same Boltz code, which groups rows by key; they differ in what the key means and in who chose the groups.

One consequence is worth knowing. When a species has several hits for the same chain — paralogs, or a duplicated domain — Boltz builds as many rows as the largest number of occurrences and cycles the shorter chain's hits to fill them (`idx = i % len(seq_idxs)` in `featurizerv2.py`, commented "roll over the sequence index to maximize diversity"). Species matching cannot say which copy of A binds which copy of B, so some of those pairings are arbitrary. Server-paired MSAs do not reach that path, because a row number occurs once per chain.

### Row order and limits

1. The query.
2. UniProt hits with a taxonomy id, best first, up to `max_paired_rows` (default 4,096).
3. UniRef90, then Small BFD, then MGnify hits, best first, until `max_rows` (default 16,384) is reached.

Duplicate sequences are dropped, keeping the first copy. Boltz itself keeps only the first copy of a duplicate, so keyed rows must come before their unpaired twins.

**Why cap the keyed rows?** At prediction time Boltz reads only the first `--max_msa_seqs` rows (default 8,192). UniProt can return up to 50,000 hits, and without a cap they would fill that whole window and push out every unpaired hit. With the default cap, at most half of the default window is keyed rows.

The two have to be set together. Lowering `--max_msa_seqs` without lowering `--max_paired_msa_rows` recreates exactly the problem the cap exists to prevent: with `--max_msa_seqs 1024` and the default cap, Boltz reads 1,024 keyed rows and no unpaired row at all. A paired cap at or above the read limit is now refused, and one above half of it warns.

## Deduplication

- Each CSV is named by the first 24 hex characters of the sequence's SHA-256. Identical sequences share one file, whatever chain or input they come from, so a receptor repeated across many complexes is searched once per run.
- `boltz predict` points `cache_dir` at a temporary folder that is removed when the run ends. Nothing persists between runs, and no state outside the output directory affects a run's result.
- `search_sequences` itself will reuse whatever is already in the `cache_dir` it is given. Pass a lasting path when calling it from Python and it behaves as a cache across calls.
- A CSV is written to a temporary name and renamed when complete, so an interrupted run never leaves a partial file behind. Note that the temporary name is derived from the final one, so two processes writing the *same* sequence into the *same* directory at once can race; give concurrent jobs separate directories.

## Example: PDB 7KFK (LILRB1 + RIFIN)

Run on one RTX PRO 4500 Blackwell (32 GB) against the AlphaFast databases:

| Chain | Length | Rows | Keyed rows |
|---|---|---|---|
| LILRB1 D1–D2 | 233 | 8,310 | 4,096 (capped) |
| RIFIN | 151 | 296 | 282 |

The search took 236 s for both sequences. A repeat `search_sequences` call with the same `cache_dir` returned instantly. The RIFIN MSA came out the same size as AlphaFast's, and the LILRB1 one differs by the keyed-row cap. Note that this compares row *counts*, not row contents, and says nothing about the predictions: it is a sanity check that the search reaches the same depth, not evidence that the two produce the same alignment.

## Limitations

- **Proteins only.** Boltz does not use MSAs for DNA, RNA or ligands.
- **No templates.** Template search is not included.
- **GPU-oriented.** The padded databases are built for GPU search. `gpu=False` exists but has not been tested on them.
- **Species pairing assumes co-evolution.** For complexes whose chains come from different organisms (for example, host–pathogen), few rows share a key. Compare runs with and without pairing.
- **Pairing differs from `--use_msa_server`, and has not been benchmarked.** The key is a taxonomy id rather than a server-assigned row number, so paralogs within one species are paired arbitrarily (see [How pairing works](#how-pairing-works)). Which approach predicts complexes better is not known: it would take running both over a held-out set of known structures and comparing DockQ or ipTM, which has not been done. Expect single chains to behave the same way and complexes to differ.
- **Row selection is not AF3's.** The search settings match AlphaFast's, but which hits become rows, in what order, and under which caps is this module's own choice, and AF3 pairs chains with its own code. Matching search settings does not make the alignment the model sees identical.
- **Disk and memory.** The databases are about 390 GB. GPU search streams them from disk, so fast local storage (NVMe) helps a lot.
