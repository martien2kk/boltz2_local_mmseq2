"""Tests for local MMseqs2 MSA generation.

Everything except the last test runs without a GPU, MMseqs2 or databases.
The last test runs a real search and is skipped unless BOLTZ_MMSEQS_DB_DIR
is set and mmseqs is on PATH.
"""

import os
import shutil
from pathlib import Path

import pytest

from boltz.data.msa.local_mmseqs import (
    build_rows,
    cached_msa_path,
    check_setup,
    collect_sequences,
    read_fasta,
    search_sequences,
    sequence_id,
    write_csv,
)
from boltz.data.parse.csv import parse_csv

QUERY = "MKTAYIAKQR"


def uniprot(acc, taxon):
    return f"tr|{acc}|{acc}_X Some protein OS=Some species OX={taxon} GN=X PE=1 SV=1"


def test_sequence_id_is_stable_and_distinct():
    assert sequence_id(QUERY) == sequence_id(QUERY)
    assert sequence_id(QUERY) != sequence_id(QUERY + "A")
    assert cached_msa_path(Path("cache"), QUERY).name == f"{sequence_id(QUERY)}.csv"


def test_read_fasta_handles_wrapping_and_null_bytes():
    text = ">q\nMKTAY\nIAKQR\n>hit desc\n-KTAYIAKQ-\n\x00"
    assert read_fasta(text) == [("q", QUERY), ("hit desc", "-KTAYIAKQ-")]


def test_build_rows_order_and_keys():
    paired = [
        (uniprot("P1", 9606), "MKTAYIAKQK"),
        ("tr|P2|P2_X no taxon here", "MKTAYIAKQL"),
    ]
    unpaired = [("UniRef90_A", "MKTAYIAKQM"), ("MGYP001", "MKTAYIAK--")]
    rows = build_rows(QUERY, paired, unpaired, max_paired_rows=10, max_rows=10)
    assert rows == [
        (-1, QUERY),
        (9606, "MKTAYIAKQK"),
        (-1, "MKTAYIAKQM"),
        (-1, "MKTAYIAK--"),
    ]


def test_build_rows_drops_duplicates_keeping_keyed_copy():
    paired = [(uniprot("P1", 10090), "MKTAYIAKQK")]
    unpaired = [("UniRef90_A", "MKTAYIAKQK"), ("UniRef90_B", "-" + QUERY[1:])]
    rows = build_rows(QUERY, paired, unpaired, max_paired_rows=10, max_rows=10)
    assert rows == [(-1, QUERY), (10090, "MKTAYIAKQK"), (-1, "-KTAYIAKQR")]


def test_build_rows_caps():
    paired = [(uniprot(f"P{i}", i), f"MKTAYIAK{i:02d}") for i in range(10)]
    unpaired = [(f"U{i}", f"MKTAYIAQ{i:02d}") for i in range(10)]

    rows = build_rows(QUERY, paired, unpaired, max_paired_rows=3, max_rows=100)
    assert sum(key != -1 for key, _ in rows) == 3
    assert len(rows) == 1 + 3 + 10

    rows = build_rows(QUERY, paired, unpaired, max_paired_rows=3, max_rows=6)
    assert len(rows) == 6


def test_written_csv_is_read_by_boltz(tmp_path):
    rows = [(-1, QUERY), (9606, "MKTAYIAKQK"), (9606, "MKTAYIAKQL"), (-1, "MKT-YIAKQR")]
    path = tmp_path / "msa.csv"
    write_csv(path, rows)

    msa = parse_csv(path)
    assert len(msa.sequences) == 4
    assert list(msa.sequences["taxonomy"]) == [-1, 9606, 9606, -1]
    assert not path.with_suffix(".csv.tmp").exists()


def test_collect_sequences_yaml(tmp_path):
    path = tmp_path / "job.yaml"
    path.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein: {id: A, sequence: AAAA}\n"
        "  - protein: {id: B, sequence: CCCC, msa: null}\n"
        "  - protein: {id: C, sequence: DDDD, msa: ./c.a3m}\n"
        "  - protein: {id: D, sequence: EEEE, msa: empty}\n"
        "  - protein: {id: [E, F], sequence: AAAA}\n"
        "  - ligand: {id: L, smiles: CCO}\n"
    )
    assert collect_sequences([path]) == ["AAAA", "CCCC"]


def test_collect_sequences_fasta(tmp_path):
    path = tmp_path / "job.fasta"
    path.write_text(
        ">A|protein\nAAAA\n"
        ">B|protein|\nCCCC\n"
        ">C|protein|./c.a3m\nDDDD\n"
        ">D|smiles\nCCO\n"
    )
    assert collect_sequences([path]) == ["AAAA", "CCCC"]


def test_check_setup_reports_missing_binary_and_databases(tmp_path):
    with pytest.raises(FileNotFoundError, match="MMseqs2 binary not found"):
        check_setup(tmp_path, "not-a-real-mmseqs-binary")

    fake = tmp_path / "mmseqs"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    (tmp_path / "uniref90_padded.dbtype").touch()
    with pytest.raises(FileNotFoundError, match="small_bfd_padded"):
        check_setup(tmp_path, str(fake))


def test_search_sequences_uses_cache_without_searching(tmp_path):
    cached = cached_msa_path(tmp_path, QUERY)
    write_csv(cached, [(-1, QUERY)])

    # A missing binary and database would fail if a search were attempted
    paths = search_sequences(
        [QUERY], db_dir=tmp_path / "missing", cache_dir=tmp_path, mmseqs="not-a-real-mmseqs"
    )
    assert paths == {QUERY: cached}


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("BOLTZ_MMSEQS_DB_DIR") or shutil.which("mmseqs") is None,
    reason="needs BOLTZ_MMSEQS_DB_DIR and mmseqs on PATH",
)
def test_search_sequences_real_search(tmp_path):
    # Ubiquitin: found in nearly every eukaryote, so it gets plenty of hits
    ubiquitin = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    paths = search_sequences([ubiquitin], os.environ["BOLTZ_MMSEQS_DB_DIR"], tmp_path)

    msa = parse_csv(paths[ubiquitin])
    taxonomy = msa.sequences["taxonomy"]
    assert len(msa.sequences) > 100
    assert taxonomy[0] == -1
    assert (taxonomy > 0).any()
    assert (taxonomy == -1).sum() > 1
