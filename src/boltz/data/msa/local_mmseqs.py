"""Local MSA generation with MMseqs2 for Boltz.

Searches AlphaFold3-style sequence databases stored in MMseqs2 padded format
(as built for MMseqs2-GPU) and writes one Boltz MSA CSV per protein sequence.
Search settings and per-database hit limits follow AlphaFast's MMseqs2-GPU
defaults, so MSAs match those used for AlphaFold3 runs on the same databases.

Each CSV holds the query, then UniProt hits keyed by NCBI taxonomy id (Boltz
pairs rows across chains that share a key), then UniRef90, Small BFD and
MGnify hits with key -1 (unpaired). CSVs are cached by sequence hash, so a
sequence is only ever searched once.
"""

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Database name -> maximum hits kept per query
UNPAIRED_DATABASES = {"uniref90": 10000, "small_bfd": 5000, "mgnify": 5000}
PAIRING_DATABASE = "uniprot"
PAIRING_MAX_HITS = 50000

TAXON_ID = re.compile(r"\bOX=(\d+)")


def sequence_id(sequence: str) -> str:
    """Return a stable file name for a protein sequence."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:24]


def cached_msa_path(cache_dir: Path, sequence: str) -> Path:
    """Return the cache location of a sequence's MSA CSV."""
    return Path(cache_dir) / f"{sequence_id(sequence)}.csv"


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


def build_rows(
    query: str,
    paired_hits: list[tuple[str, str]],
    unpaired_hits: list[tuple[str, str]],
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
    unpaired_hits : list[tuple[str, str]]
        Other hits as (header, aligned sequence), best first.
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

    for _, aligned in unpaired_hits:
        if len(rows) >= max_rows:
            break
        add(-1, aligned)

    return rows


def write_csv(path: Path, rows: list[tuple[int, str]]) -> None:
    """Write MSA rows as a Boltz CSV, atomically."""
    tmp = path.with_suffix(".csv.tmp")
    lines = ["key,sequence"] + [f"{key},{seq}" for key, seq in rows]
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(path)


def _run(cmd: list[str]) -> None:
    """Run an MMseqs2 command, raising with its stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        msg = f"{' '.join(cmd[:2])} failed:\n{proc.stderr[-2000:]}{proc.stdout[-2000:]}"
        raise RuntimeError(msg)


def _search_database(  # noqa: PLR0913
    mmseqs: str,
    query_db: Path,
    target_db: Path,
    max_hits: int,
    work: Path,
    sensitivity: float,
    e_value: float,
    threads: int,
    gpu: bool,
) -> dict[int, list[tuple[str, str]]]:
    """Search all queries against one database.

    Returns
    -------
    dict[int, list[tuple[str, str]]]
        Query index -> hits as (header, aligned sequence), query row excluded.

    """
    work.mkdir()
    result, msa, out = work / "result", work / "msa", work / "out"
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
    _run(search)
    _run([mmseqs, "result2msa", str(query_db), str(target_db), str(result), str(msa),
          "--msa-format-mode", "2"])  # fmt: skip
    _run([mmseqs, "unpackdb", str(msa), str(out), "--unpack-name-mode", "0"])

    return {int(f.name): read_fasta(f.read_text())[1:] for f in out.iterdir()}


def search_sequences(  # noqa: PLR0913
    sequences: list[str],
    db_dir: Path,
    cache_dir: Path,
    mmseqs: str = "mmseqs",
    sensitivity: float = 7.5,
    e_value: float = 1e-4,
    threads: Optional[int] = None,
    gpu: bool = True,
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

        with tempfile.TemporaryDirectory(prefix="boltz_mmseqs_", dir=tmp_dir) as tmp:
            work = Path(tmp)
            fasta = work / "query.fasta"
            fasta.write_text("".join(f">{i}\n{s}\n" for i, s in enumerate(todo)))
            query_db = work / "queryDB"
            _run([mmseqs, "createdb", str(fasta), str(query_db)])

            def search(name: str, max_hits: int) -> dict[int, list[tuple[str, str]]]:
                return _search_database(
                    mmseqs, query_db, Path(db_dir) / f"{name}_padded", max_hits,
                    work / name, sensitivity, e_value, threads, gpu,
                )  # fmt: skip

            unpaired = {i: [] for i in range(len(todo))}
            for name, max_hits in UNPAIRED_DATABASES.items():
                for i, hits in search(name, max_hits).items():
                    unpaired[i].extend(hits)
            paired = search(PAIRING_DATABASE, PAIRING_MAX_HITS)

            for i, seq in enumerate(todo):
                rows = build_rows(
                    seq, paired.get(i, []), unpaired[i], max_paired_rows, max_rows
                )
                write_csv(cached_msa_path(cache_dir, seq), rows)

        logger.info("MMseqs2 search finished in %.1f s", time.time() - start)

    return {s: cached_msa_path(cache_dir, s) for s in sequences}
