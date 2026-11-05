"""Tests for local MMseqs2 MSA generation.

Everything except the last test runs without a GPU, MMseqs2 or databases.
The last test runs a real search and is skipped unless BOLTZ_MMSEQS_DB_DIR
is set and mmseqs is on PATH.
"""

import logging
import os
import shutil
from pathlib import Path

import pytest

from boltz.data.msa.local_mmseqs import (
    PAIRING_DATABASE,
    SEARCH_ORDER,
    UNPAIRED_DATABASES,
    build_rows,
    cached_msa_path,
    check_disk_space,
    check_setup,
    collect_sequences,
    read_fasta,
    reduce_paired,
    reduce_unpaired,
    search_sequences,
    sequence_id,
    visible_devices,
    write_csv,
)
from boltz.data.parse.csv import parse_csv

QUERY = "MKTAYIAKQR"


def uniprot(acc, taxon):
    return f"tr|{acc}|{acc}_X Some protein OS=Some species OX={taxon} GN=X PE=1 SV=1"


def test_search_order_covers_every_database_once():
    # A database missing here is never searched; a duplicate is searched twice.
    assert sorted(SEARCH_ORDER) == sorted([*UNPAIRED_DATABASES, PAIRING_DATABASE])
    assert len(set(SEARCH_ORDER)) == len(SEARCH_ORDER)


def test_visible_devices_uses_the_ids_the_scheduler_allocated(monkeypatch):
    # The ids are absolute: told "0", a child opens physical GPU 0, which on a
    # shared node belongs to another job. An allocation must be passed through.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert visible_devices(2) == ["2", "3"]
    assert visible_devices(1) == ["2"]

    # Asking for more than was allocated takes what there is, not GPU 0.
    assert visible_devices(8) == ["2", "3"]

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert visible_devices(3) == ["0", "1", "2"]


def test_visible_devices_never_returns_nothing(monkeypatch):
    # An empty list would divide threads by zero in the search driver.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert visible_devices(0) == ["0"]
    assert visible_devices(-1) == ["0"]


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
    unpaired = ["MKTAYIAKQM", "MKTAYIAK--"]
    rows = build_rows(QUERY, paired, unpaired, max_paired_rows=10, max_rows=10)
    assert rows == [
        (-1, QUERY),
        (9606, "MKTAYIAKQK"),
        (-1, "MKTAYIAKQM"),
        (-1, "MKTAYIAK--"),
    ]


def test_build_rows_drops_duplicates_keeping_keyed_copy():
    paired = [(uniprot("P1", 10090), "MKTAYIAKQK")]
    unpaired = ["MKTAYIAKQK", "-" + QUERY[1:]]
    rows = build_rows(QUERY, paired, unpaired, max_paired_rows=10, max_rows=10)
    assert rows == [(-1, QUERY), (10090, "MKTAYIAKQK"), (-1, "-KTAYIAKQR")]


def test_build_rows_caps():
    paired = [(uniprot(f"P{i}", i), f"MKTAYIAK{i:02d}") for i in range(10)]
    unpaired = [f"MKTAYIAQ{i:02d}" for i in range(10)]

    rows = build_rows(QUERY, paired, unpaired, max_paired_rows=3, max_rows=100)
    assert sum(key != -1 for key, _ in rows) == 3
    assert len(rows) == 1 + 3 + 10

    rows = build_rows(QUERY, paired, unpaired, max_paired_rows=3, max_rows=6)
    assert len(rows) == 6


def test_reduce_paired_drops_what_build_rows_cannot_use():
    entries = [
        (uniprot("P1", 9606), "MKTAYIAKQK"),
        ("tr|P2|P2_X no taxon here", "MKTAYIAKQL"),  # unkeyed, never a row
        (uniprot("P3", 10090), "MKTAYIAKQK"),  # repeat of P1's sequence
        (uniprot("P4", 7227), "MKTAYIAKQM"),
    ]
    assert reduce_paired(entries, 10) == [entries[0], entries[3]]
    assert reduce_paired(entries, 1) == [entries[0]]


def test_reduce_unpaired_drops_headers_and_duplicates():
    entries = [
        ("UniRef90_A", "MKTAYIAKQM"),
        ("UniRef90_B", "MKTAYIAKQM"),  # same sequence
        ("MGYP001", "MKTAYIAK--"),
    ]
    assert reduce_unpaired(entries, 10) == ["MKTAYIAKQM", "MKTAYIAK--"]
    assert reduce_unpaired(entries, 1) == ["MKTAYIAKQM"]


def test_reducing_does_not_change_the_rows_built():
    # The reduction is only safe if what it drops is what build_rows would
    # have ignored: hits past the budget, hits with no taxon id, and repeats
    # of a sequence already taken.
    paired, unpaired = [], []
    for i in range(50):
        paired.append((uniprot(f"P{i}", 9000 + i), f"MKTAYIA{i:03d}"))
        paired.append(("tr|X|X_X no taxon here", f"MKTAYIB{i:03d}"))
        paired.append((uniprot(f"D{i}", 1), f"MKTAYIA{i:03d}"))
        unpaired.append((f"U{i}", f"MKTAYIC{i:03d}"))

    full = build_rows(
        QUERY, paired, [s for _, s in unpaired], max_paired_rows=8, max_rows=20
    )
    reduced = build_rows(
        QUERY,
        reduce_paired(paired, 8 + 1),
        reduce_unpaired(unpaired, 20),
        max_paired_rows=8,
        max_rows=20,
    )
    assert reduced == full


def test_reduce_paired_keeps_one_spare_for_a_hit_repeating_the_query():
    # reduce_paired never sees the query, so one hit it keeps may turn out to
    # repeat it and be dropped by build_rows. The spare is what keeps the row
    # count the same as not reducing at all.
    paired = [(uniprot("Q", 9606), QUERY)]
    paired += [(uniprot(f"P{i}", 9000 + i), f"MKTAYIA{i:03d}") for i in range(10)]

    full = build_rows(QUERY, paired, [], max_paired_rows=4, max_rows=50)
    assert sum(key != -1 for key, _ in full) == 4

    assert build_rows(
        QUERY, reduce_paired(paired, 4 + 1), [], max_paired_rows=4, max_rows=50
    ) == full
    # Without the spare, the query's twin costs a row that should have been used.
    assert build_rows(
        QUERY, reduce_paired(paired, 4), [], max_paired_rows=4, max_rows=50
    ) != full


def test_disk_space_check_warns_but_does_not_raise(tmp_path, caplog):
    # A rough estimate must never stop a run that would have worked.
    with caplog.at_level(logging.WARNING):
        check_disk_space(tmp_path, 10**9)
    assert "free" in caplog.text

    check_disk_space(tmp_path / "does-not-exist", 1)


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
