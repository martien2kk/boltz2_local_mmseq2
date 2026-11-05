"""Local MSA generation with MMseqs2 for Boltz.

Searches AlphaFold3-style sequence databases stored in MMseqs2 padded format
(as built for MMseqs2-GPU) and writes one Boltz MSA CSV per protein sequence.
Search settings and per-database hit limits follow AlphaFast's MMseqs2-GPU
defaults, so MSAs match those used for AlphaFold3 runs on the same databases.

Each CSV holds the query, then UniProt hits keyed by NCBI taxonomy id (Boltz
pairs rows across chains that share a key), then UniRef90, Small BFD and
MGnify hits with key -1 (unpaired). CSVs are named by sequence hash, so a
sequence shared by several inputs is searched once per run. Boltz points
cache_dir at a directory that lives only for the run, so nothing persists
between runs; pass a lasting path yourself to reuse MSAs across runs.
"""

import hashlib
import logging
import os
import queue
import re
import shutil
import string
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Database name -> maximum hits kept per query
UNPAIRED_DATABASES = {"uniref90": 10000, "small_bfd": 5000, "mgnify": 5000}
PAIRING_DATABASE = "uniprot"
PAIRING_MAX_HITS = 50000

# Order the GPU searches are run in. One database's post-processing overlaps
# the next one's search, so whichever needs the least post-processing goes
# last, where nothing follows it to hide behind. Hits are merged in
# UNPAIRED_DATABASES order whatever this says, so it does not affect the MSAs.
SEARCH_ORDER = ("uniref90", "mgnify", "uniprot", "small_bfd")

# Rough temporary space one query needs for MMseqs2 intermediates: the search
# result database, the aligned FASTA from result2msa, and the unpacked copy of
# it. It scales with hits and query length, so it is an order-of-magnitude
# guide for the preflight warning, not a measured bound. The actual peak is
# logged at the end of a search, so a run tells you your own number.
TMP_BYTES_PER_SEQUENCE = 250 * 1024**2

TAXON_ID = re.compile(r"\bOX=(\d+)")

# Letters Boltz has a protein token for. A query sequence holding anything
# else is rejected rather than cleaned up: the sequence is written straight
# into a FASTA and a CSV, and Boltz parses the same input separately, so a
# sequence this module silently altered would no longer be the one Boltz
# folds. See checked_sequence.
PROTEIN_LETTERS = frozenset(string.ascii_uppercase)


def checked_sequence(sequence: str, path: Path) -> str:
    """Return a protein sequence, or raise if it is not one Boltz can use.

    This is a guard as much as a check. Sequences are written into a FASTA as
    `>{index}\\n{sequence}`, and a newline or ">" inside one would start a
    FASTA record of its own, shifting every later query's index away from the
    sequence it belongs to. They are also written into a CSV as `{key},{seq}`,
    where a comma or newline breaks the row. Allowing only letters makes all
    of that impossible rather than merely unlikely.

    Parameters
    ----------
    sequence : str
        The sequence as it appears in the input.
    path : Path
        The input file it came from, named in the error.

    Returns
    -------
    str
        The sequence, unchanged.

    """
    bad = sorted(set(sequence) - PROTEIN_LETTERS)
    if not sequence or bad:
        shown = ", ".join(repr(c) for c in bad)
        reason = f"characters Boltz has no protein token for: {shown}"
        msg = f"{path}: protein sequence is {'empty' if not sequence else reason}."
        raise ValueError(msg)
    return sequence


def sequence_id(sequence: str) -> str:
    """Return a stable file name for a protein sequence."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:24]


def cached_msa_path(cache_dir: Path, sequence: str) -> Path:
    """Return the cache location of a sequence's MSA CSV."""
    return Path(cache_dir) / f"{sequence_id(sequence)}.csv"


def collect_sequences(paths: list[Path]) -> list[str]:
    """Find protein sequences that need a generated MSA in Boltz inputs.

    A protein needs one when its MSA is unset, as in Boltz's own parsing:
    no msa field (or null / "" / 0) in YAML, or no MSA path in a FASTA header.

    Parameters
    ----------
    paths : list[Path]
        Boltz input files (.yaml, .yml, .fasta, .fa, .fas).

    Returns
    -------
    list[str]
        Unique sequences, in order of first appearance.

    """
    sequences = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in (".yml", ".yaml"):
            with path.open() as f:
                doc = yaml.safe_load(f)
            for entry in doc.get("sequences", []):
                protein = entry.get("protein")
                if protein and protein.get("msa") in (None, "", 0):
                    sequences.append(
                        checked_sequence(str(protein["sequence"]), path)
                    )
        elif suffix in (".fa", ".fas", ".fasta"):
            for header, seq in read_fasta(path.read_text()):
                fields = header.split()[0].split("|")
                is_protein = len(fields) >= 2 and fields[1].lower() == "protein"
                if is_protein and (len(fields) < 3 or fields[2] == ""):
                    sequences.append(checked_sequence(seq, path))
    return list(dict.fromkeys(sequences))


def check_setup(db_dir: Path, mmseqs: str) -> str:
    """Check that MMseqs2 and the padded databases are available.

    Parameters
    ----------
    db_dir : Path
        Directory containing <name>_padded MMseqs2 databases.
    mmseqs : str
        MMseqs2 binary name or path.

    Returns
    -------
    str
        The resolved path of the MMseqs2 binary.

    """
    binary = shutil.which(mmseqs)
    if binary is None:
        msg = f"MMseqs2 binary not found: {mmseqs}. See docs/local_msa.md."
        raise FileNotFoundError(msg)

    names = [*UNPAIRED_DATABASES, PAIRING_DATABASE]
    missing = [n for n in names if not (Path(db_dir) / f"{n}_padded.dbtype").exists()]
    if missing:
        msg = f"Missing MMseqs2 databases in {db_dir}: " + ", ".join(
            f"{n}_padded" for n in missing
        )
        raise FileNotFoundError(msg)
    return binary


def read_fasta(text: str) -> list[tuple[str, str]]:
    """Parse (header, sequence) pairs from FASTA text."""
    entries = []
    header, seq = None, []
    for line in text.replace("\x00", "").splitlines():
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(seq)))
            header, seq = line[1:], []
        elif line.strip():
            seq.append(line.strip())
    if header is not None:
        entries.append((header, "".join(seq)))
    return entries


def reduce_paired(
    entries: list[tuple[str, str]], limit: int
) -> list[tuple[str, str]]:
    """Keep only the paired hits build_rows could still use.

    Hits arrive best first. build_rows keeps at most max_paired_rows of them,
    ignores any without a taxonomy id, and drops repeats of a sequence it has
    already taken, so everything past that point is read and held for nothing.
    UniProt returns up to PAIRING_MAX_HITS (50,000) per query against a budget
    of 4,096, which makes this most of the memory the search ever uses.

    Parameters
    ----------
    entries : list[tuple[str, str]]
        Hits as (header, aligned sequence), best first.
    limit : int
        How many to keep. Pass one more than build_rows can use, since one of
        them may turn out to repeat the query and be dropped there instead.

    Returns
    -------
    list[tuple[str, str]]
        The kept hits, in their original order.

    """
    kept: list[tuple[str, str]] = []
    seen: set[str] = set()
    for header, aligned in entries:
        if len(kept) >= limit:
            break
        if not TAXON_ID.search(header):
            continue
        ungapped = aligned.replace("-", "").upper()
        if ungapped not in seen:
            seen.add(ungapped)
            kept.append((header, aligned))
    return kept


def reduce_unpaired(entries: list[tuple[str, str]], limit: int) -> list[str]:
    """Keep only the unpaired hits build_rows could still use.

    As reduce_paired, but build_rows never looks at an unpaired hit's header
    (every such row is keyed -1), so the header is dropped here rather than
    carried through the merge.

    Parameters
    ----------
    entries : list[tuple[str, str]]
        Hits as (header, aligned sequence), best first.
    limit : int
        How many to keep; max_rows, which no single database can exceed.

    Returns
    -------
    list[str]
        The kept aligned sequences, in their original order.

    """
    kept: list[str] = []
    seen: set[str] = set()
    for _, aligned in entries:
        if len(kept) >= limit:
            break
        ungapped = aligned.replace("-", "").upper()
        if ungapped not in seen:
            seen.add(ungapped)
            kept.append(aligned)
    return kept


def check_disk_space(tmp_dir: Optional[Path], n_sequences: int) -> None:
    """Warn if the temporary directory looks too small for the search.

    MMseqs2 intermediates are far larger than the MSAs they reduce to, and on
    a cluster the default temp directory is often small and shared. Filling it
    takes down other jobs on the node, so say so before the search rather than
    after. The estimate is rough, so this warns and continues.

    Parameters
    ----------
    tmp_dir : Path, optional
        Where intermediates will go. None means the system temp directory.
    n_sequences : int
        How many sequences are about to be searched.

    """
    target = Path(tmp_dir) if tmp_dir is not None else Path(tempfile.gettempdir())
    try:
        free = shutil.disk_usage(target).free
    except OSError as err:  # unreadable mount point, missing path
        logger.warning("Could not check free space on %s: %s", target, err)
        return

    needed = n_sequences * TMP_BYTES_PER_SEQUENCE
    gib = 1024**3
    if free < needed:
        logger.warning(
            "%s has %.1f GiB free; %d sequence(s) may need about %.1f GiB of "
            "MMseqs2 intermediates. Point --mmseqs_tmp_dir at larger storage, "
            "or lower --msa_batch_size.",
            target, free / gib, n_sequences, needed / gib,
        )  # fmt: skip
    else:
        logger.info(
            "Temporary directory %s: %.1f GiB free, about %.1f GiB estimated.",
            target, free / gib, needed / gib,
        )  # fmt: skip


def directory_size(path: Path) -> int:
    """Return the total size in bytes of every file under a directory."""
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:  # removed while walking
            continue
    return total


def build_rows(
    query: str,
    paired_hits: list[tuple[str, str]],
    unpaired_hits: list[str],
    max_paired_rows: int,
    max_rows: int,
) -> list[tuple[int, str]]:
    """Combine hits into Boltz MSA rows of (key, aligned sequence).

    Parameters
    ----------
    query : str
        The query sequence, always the first row.
    paired_hits : list[tuple[str, str]]
        UniProt hits as (header, aligned sequence), best first.
    unpaired_hits : list[str]
        Other hits as aligned sequences, best first. Headers are dropped by
        reduce_unpaired, since every row built from them is keyed -1.
    max_paired_rows : int
        Maximum number of taxonomy-keyed rows.
    max_rows : int
        Maximum total number of rows.

    Returns
    -------
    list[tuple[int, str]]
        Rows with keyed UniProt hits first. Duplicate sequences are dropped,
        keeping the first (keyed) copy, as Boltz keeps only the first copy.

    """
    rows = [(-1, query)]
    seen = {query.upper()}

    def add(key: int, aligned: str) -> None:
        ungapped = aligned.replace("-", "").upper()
        if ungapped not in seen:
            seen.add(ungapped)
            rows.append((key, aligned))

    for header, aligned in paired_hits:
        if len(rows) > max_paired_rows or len(rows) >= max_rows:
            break
        match = TAXON_ID.search(header)
        if match:
            add(int(match.group(1)), aligned)

    for aligned in unpaired_hits:
        if len(rows) >= max_rows:
            break
        add(-1, aligned)

    return rows


def write_csv(path: Path, rows: list[tuple[int, str]]) -> None:
    """Write MSA rows as a Boltz CSV, atomically.

    A sequence holding a comma or a newline would split its row in two and
    silently shift every column after it, so refuse rather than write one.
    checked_sequence keeps queries clean; this covers the aligned rows, which
    come from the databases.
    """
    lines = ["key,sequence"]
    for key, seq in rows:
        if {",", "\n", "\r"} & set(seq):
            msg = f"Refusing to write a row that would break the CSV: {path.name}"
            raise ValueError(msg)
        lines.append(f"{key},{seq}")

    tmp = path.with_suffix(".csv.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(path)


def _run(cmd: list[str], device: Optional[str] = None) -> None:
    """Run an MMseqs2 command, raising with its stderr on failure.

    When device is given, the command sees only that CUDA device.
    """
    env = None
    if device is not None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)  # noqa: S603
    if proc.returncode != 0:
        msg = f"{' '.join(cmd[:2])} failed:\n{proc.stderr[-2000:]}{proc.stdout[-2000:]}"
        raise RuntimeError(msg)


def visible_devices(n_gpus: int) -> list[str]:
    """Return CUDA device ids to spread searches over.

    CUDA_VISIBLE_DEVICES holds absolute device ids, so a child given "0" sees
    physical device 0, not the first device of the parent's allocation. Take
    the ids from the parent's allocation where there is one, so a scheduler
    that hands out GPUs 2 and 3 is honoured.

    Parameters
    ----------
    n_gpus : int
        How many devices to use at once.

    Returns
    -------
    list[str]
        Device ids, at most n_gpus of them.

    """
    n_gpus = max(1, n_gpus)  # never return an empty device list
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    allocated = [d.strip() for d in visible.split(",") if d.strip()]
    if not allocated:
        allocated = [str(i) for i in range(n_gpus)]
    if n_gpus > len(allocated):
        logger.warning(
            "Asked for %d GPUs but only %d visible (%s); using %d.",
            n_gpus, len(allocated), ",".join(allocated), len(allocated),
        )  # fmt: skip
    return allocated[:n_gpus]


def _gpu_search(  # noqa: PLR0913
    mmseqs: str,
    query_db: Path,
    target_db: Path,
    max_hits: int,
    work: Path,
    sensitivity: float,
    e_value: float,
    threads: int,
    gpu: bool,
    device: Optional[str] = None,
) -> dict:
    """Search all queries against one database, the GPU half only.

    Returns
    -------
    dict
        What _postprocess needs to turn the result into hits.

    """
    work.mkdir()
    result = work / "result"
    search = [
        mmseqs, "search", str(query_db), str(target_db), str(result), str(work / "tmp"),
        "-a",
        "-s", str(sensitivity),
        "-e", str(e_value),
        "--max-seqs", str(max_hits),
        "--threads", str(threads),
    ]  # fmt: skip
    if gpu:
        search += ["--gpu", "1"]

    start = time.time()
    _run(search, device=device)
    return {
        "result": result,
        "target_db": target_db,
        "work": work,
        "device": device,
        "search_time": time.time() - start,
    }


def _postprocess(
    mmseqs: str, query_db: Path, gpu_result: dict, paired: bool, limit: int
) -> dict[int, list]:
    """Turn one database's search result into hits. Uses the CPU only, so it
    runs in a worker thread while the next database searches on the GPU.

    Each query's hits are reduced as they are read, so only what build_rows
    can still use is ever held. The database's intermediates are deleted
    straight afterwards, which bounds temporary space at what the searches
    still running need rather than at every database's output at once.

    Parameters
    ----------
    mmseqs : str
        MMseqs2 binary path.
    query_db : Path
        The shared query database.
    gpu_result : dict
        What _gpu_search returned.
    paired : bool
        Whether this is the pairing database, whose hits keep their headers.
    limit : int
        Hits kept per query, after dropping repeats of a sequence.

    Returns
    -------
    dict[int, list]
        Query index -> hits, best first, query row excluded. Pairing hits are
        (header, aligned sequence); the rest are aligned sequences.

    """
    work, result = gpu_result["work"], gpu_result["result"]
    target_db = gpu_result["target_db"]
    msa, out = work / "msa", work / "out"

    start = time.time()
    _run([mmseqs, "result2msa", str(query_db), str(target_db), str(result), str(msa),
          "--msa-format-mode", "2"])  # fmt: skip
    _run([mmseqs, "unpackdb", str(msa), str(out), "--unpack-name-mode", "0"])

    reduce = reduce_paired if paired else reduce_unpaired
    hits = {}
    for f in out.iterdir():
        # One file at a time, reduced before the next is read.
        hits[int(f.name)] = reduce(read_fasta(f.read_text())[1:], limit)

    peak = directory_size(work)
    shutil.rmtree(work, ignore_errors=True)

    device = gpu_result["device"]
    logger.info(
        "%s%s: %.1f s search, %.1f s post-processing, %.2f GiB intermediates",
        target_db.name,
        f" (GPU {device})" if device is not None else "",
        gpu_result["search_time"],
        time.time() - start,
        peak / 1024**3,
    )
    return hits


def search_sequences(  # noqa: PLR0913
    sequences: list[str],
    db_dir: Path,
    cache_dir: Path,
    mmseqs: str = "mmseqs",
    sensitivity: float = 7.5,
    e_value: float = 1e-4,
    threads: Optional[int] = None,
    gpu: bool = True,
    n_gpus: int = 1,
    tmp_dir: Optional[Path] = None,
    max_paired_rows: int = 4096,
    max_rows: int = 16384,
) -> dict[str, Path]:
    """Build and cache Boltz MSA CSVs for protein sequences.

    All sequences missing from the cache are searched together, one MMseqs2
    run per database, so the databases are read once per batch.

    Parameters
    ----------
    sequences : list[str]
        Protein sequences.
    db_dir : Path
        Directory containing uniref90, small_bfd, mgnify and uniprot
        MMseqs2 padded databases (<name>_padded).
    cache_dir : Path
        Directory where MSA CSVs are stored.
    mmseqs : str
        MMseqs2 binary name or path.
    sensitivity : float
        MMseqs2 search sensitivity (-s).
    e_value : float
        MMseqs2 e-value cutoff (-e).
    threads : int, optional
        CPU threads. Defaults to all available.
    gpu : bool
        Whether to search on the GPU.
    n_gpus : int
        How many GPUs to spread the database searches over.
    tmp_dir : Path, optional
        Where to put intermediate files.
    max_paired_rows : int
        Maximum taxonomy-keyed UniProt rows per MSA.
    max_rows : int
        Maximum total rows per MSA.

    Returns
    -------
    dict[str, Path]
        Sequence -> path of its MSA CSV.

    """
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = list(dict.fromkeys(s for s in sequences if not cached_msa_path(cache_dir, s).exists()))

    if todo:
        mmseqs = check_setup(db_dir, mmseqs)
        threads = threads or len(os.sched_getaffinity(0))
        start = time.time()
        logger.info("Searching %d sequence(s) with MMseqs2", len(todo))
        check_disk_space(tmp_dir, len(todo))

        with tempfile.TemporaryDirectory(prefix="boltz_mmseqs_", dir=tmp_dir) as tmp:
            work = Path(tmp)
            fasta = work / "query.fasta"
            fasta.write_text("".join(f">{i}\n{s}\n" for i, s in enumerate(todo)))
            query_db = work / "queryDB"
            _run([mmseqs, "createdb", str(fasta), str(query_db)])

            # One task per database. A device is held only for the search, so
            # with one GPU the searches run one at a time and each database's
            # post-processing overlaps the next search. With several GPUs that
            # many searches run at once, one per device.
            max_hits_by_db = {**UNPAIRED_DATABASES, PAIRING_DATABASE: PAIRING_MAX_HITS}
            # One search per database at most, so asking for more devices than
            # there are databases would leave the extra ones idle while still
            # dividing the threads by the number asked for.
            usable = min(n_gpus, len(SEARCH_ORDER))
            if n_gpus > usable:
                logger.warning(
                    "Asked for %d GPUs but there are only %d databases to "
                    "search; using %d.",
                    n_gpus, len(SEARCH_ORDER), usable,
                )  # fmt: skip
            devices = visible_devices(usable) if gpu else [None]
            search_threads = max(1, threads // len(devices))
            free_devices = queue.Queue()
            for device in devices:
                free_devices.put(device)
            cpu_slots = threading.Semaphore(len(devices))

            def one_database(name: str) -> dict[int, list]:
                device = free_devices.get()
                try:
                    gpu_result = _gpu_search(
                        mmseqs, query_db, Path(db_dir) / f"{name}_padded",
                        max_hits_by_db[name], work / name, sensitivity, e_value,
                        search_threads, gpu, device,
                    )  # fmt: skip
                finally:
                    free_devices.put(device)
                # How many hits build_rows could still use from one database.
                # It needs its remaining row budget, plus one spare for every
                # row already placed, since a kept hit may repeat one of them
                # and be dropped. Those two add up to the total row cap: for
                # the pairing database max_paired_rows against the query
                # alone, and for the rest max_rows.
                paired = name == PAIRING_DATABASE
                limit = max_paired_rows + 1 if paired else max_rows
                with cpu_slots:
                    return _postprocess(mmseqs, query_db, gpu_result, paired, limit)

            logger.info(
                "Searching %d database(s) over %d device(s), %d threads each",
                len(SEARCH_ORDER), len(devices), search_threads,
            )  # fmt: skip
            with ThreadPoolExecutor(max_workers=len(SEARCH_ORDER)) as executor:
                pending = {n: executor.submit(one_database, n) for n in SEARCH_ORDER}
                results = {n: f.result() for n, f in pending.items()}

            # Hits are matched to sequences by position in the query FASTA, so
            # a result index outside it means the two have come apart and the
            # hits cannot be trusted to belong to the sequences they land on.
            # checked_sequence should make this unreachable; say so clearly if
            # it ever happens rather than failing later on a missing key.
            expected = set(range(len(todo)))
            for name, hits in results.items():
                unexpected = sorted(set(hits) - expected)
                if unexpected:
                    msg = (
                        f"{name} returned results for queries "
                        f"{unexpected} but only {len(todo)} were searched. "
                        "The query database does not match the sequences."
                    )
                    raise RuntimeError(msg)
                missing = sorted(expected - set(hits))
                if missing:
                    logger.warning(
                        "%s returned no result for %d of %d queries (%s)",
                        name, len(missing), len(todo), missing[:10],
                    )  # fmt: skip

            # Merge in UNPAIRED_DATABASES order, not search order: it decides
            # which rows max_rows keeps, so it has to stay fixed.
            unpaired = {i: [] for i in range(len(todo))}
            for name in UNPAIRED_DATABASES:
                for i, hits in results[name].items():
                    unpaired[i].extend(hits)
            paired = results[PAIRING_DATABASE]

            for i, seq in enumerate(todo):
                rows = build_rows(
                    seq, paired.get(i, []), unpaired[i], max_paired_rows, max_rows
                )
                write_csv(cached_msa_path(cache_dir, seq), rows)

        logger.info("MMseqs2 search finished in %.1f s", time.time() - start)

    return {s: cached_msa_path(cache_dir, s) for s in sequences}
