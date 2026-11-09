"""Tests for local MMseqs2 MSA generation.

Everything except the last test runs without a GPU, MMseqs2 or databases.
The last test runs a real search and is skipped unless BOLTZ_MMSEQS_DB_DIR
is set and mmseqs is on PATH.
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from boltz.main import predict
from boltz.data.msa.local_mmseqs import (
    PAIRING_DATABASE,
    SEARCH_ORDER,
    UNPAIRED_DATABASES,
    build_rows,
    cached_msa_path,
    check_disk_space,
    check_gpu_memory,
    checked_sequence,
    check_setup,
    collect_sequences,
    describe_depths,
    describe_hits,
    elapsed,
    gpu_memory,
    heartbeat,
    log_to_stdout,
    read_fasta,
    read_lookup,
    reduce_paired,
    reduce_unpaired,
    search_sequences,
    sequence_id,
    visible_devices,
    write_csv,
)
from boltz.data.msa import local_mmseqs
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


def test_rejects_sequences_that_would_break_the_fasta_or_csv():
    # Written as ">{i}\n{seq}", a ">" after a newline starts a record of its
    # own and every later query's index stops matching its sequence. Written
    # as "{key},{seq}", a comma splits the row. Letters only makes both
    # impossible instead of unlikely.
    for bad in ["MKTA\n>9\nEVIL", "MKTA\nEVIL", "MKT,AYI", "MKTAYI-", "mktayi", ""]:
        with pytest.raises(ValueError, match="protein sequence is"):
            checked_sequence(bad, Path("job.yaml"))

    assert checked_sequence(QUERY, Path("job.yaml")) == QUERY


def test_rejected_sequence_names_the_file_and_the_characters():
    with pytest.raises(ValueError) as err:
        checked_sequence("MKT@AYI", Path("/inputs/job.yaml"))
    assert "/inputs/job.yaml" in str(err.value)
    assert "'@'" in str(err.value)


def test_collect_sequences_rejects_a_bad_sequence(tmp_path):
    job = tmp_path / "job.yaml"
    job.write_text("sequences:\n  - protein:\n      id: A\n      sequence: MK TA\n")
    with pytest.raises(ValueError, match="protein sequence is"):
        collect_sequences([job])


def test_write_csv_refuses_a_row_that_would_break_the_format(tmp_path):
    path = tmp_path / "msa.csv"
    with pytest.raises(ValueError, match="break the CSV"):
        write_csv(path, [(-1, QUERY), (9606, "MKTA,YIAK")])
    assert not path.exists()


@pytest.mark.skipif(
    not (Path.home() / ".boltz" / "mols").exists(),
    reason="needs the Boltz mol cache (~/.boltz/mols)",
)
def test_collect_sequences_agrees_with_the_boltz_parser():
    # collect_sequences decides which proteins need an MSA without using
    # Boltz's parser, so the two can drift apart: a new way of declaring an
    # MSA upstream would have this module searching sequences Boltz does not
    # want searched. Pin them together on the shipped examples.
    from boltz.data import const
    from boltz.data.mol import load_canonicals
    from boltz.data.parse.yaml import parse_yaml

    mol_dir = Path.home() / ".boltz" / "mols"
    ccd = load_canonicals(mol_dir)
    prot_id = const.chain_type_ids["PROTEIN"]

    examples = sorted(Path(__file__).parent.parent.glob("examples/*.yaml"))
    assert examples, "no example inputs found"

    compared, unparsed = [], []
    for path in examples:
        try:
            target = parse_yaml(path, ccd, mol_dir, boltz2=True)
        except Exception as err:  # noqa: BLE001
            # Boltz's own parser failing is not drift, and this test should
            # not go red for an upstream bug. Skip the input, but record it:
            # an input Boltz cannot read is one this test does not cover.
            unparsed.append(f"{path.name}: {type(err).__name__}: {err}")
            continue

        wanted = {
            target.sequences[chain.entity_id]
            for chain in target.record.chains
            if chain.mol_type == prot_id and chain.msa_id == 0
        }
        assert set(collect_sequences([path])) == wanted, path
        compared.append(path.name)

    assert compared, f"no example could be parsed by Boltz: {unparsed}"
    if unparsed:
        print(f"\nnot covered, Boltz could not parse: {unparsed}")


def local_msa_cli(tmp_path, *extra):
    """Invoke `boltz predict --use_local_msa` with real paths for the rest."""
    job = tmp_path / "job.yaml"
    job.write_text(f"sequences:\n  - protein:\n      id: A\n      sequence: {QUERY}\n")
    return CliRunner().invoke(
        predict,
        [str(job), "--use_local_msa", "--mmseqs_db_dir", str(tmp_path),
         "--out_dir", str(tmp_path / "out"), *extra],
    )  # fmt: skip


def test_rejects_a_paired_cap_that_would_crowd_out_unpaired_rows(tmp_path):
    # Keyed rows are written first and Boltz reads the file from the top, so
    # a paired cap at or above --max_msa_seqs leaves room for no unpaired row
    # at all. The run would otherwise look entirely normal.
    result = local_msa_cli(
        tmp_path, "--max_msa_seqs", "1024", "--max_paired_msa_rows", "4096"
    )
    assert result.exit_code != 0
    assert "must be below" in result.output
    # The value it suggests has to be one that gets past the check.
    assert "--max_paired_msa_rows 512" in result.output

    result = local_msa_cli(
        tmp_path, "--max_msa_seqs", "1024", "--max_paired_msa_rows", "512"
    )
    assert "must be below" not in result.output


def test_log_level_option_sets_the_heartbeat(tmp_path):
    # The option has to reach the module constant: the interval used to be a
    # default argument frozen at import, where setting it did nothing.
    for level, expected in (("quiet", 0.0), ("normal", 300.0), ("debug", 60.0)):
        local_mmseqs.HEARTBEAT_SECONDS = -1.0
        local_msa_cli(tmp_path, "--msa_log_level", level)
        assert local_mmseqs.HEARTBEAT_SECONDS == expected, level
    local_mmseqs.HEARTBEAT_SECONDS = 300.0

    result = local_msa_cli(tmp_path, "--msa_log_level", "nonsense")
    assert result.exit_code != 0
    assert "nonsense" in result.output


def test_warns_when_keyed_rows_take_most_of_the_window(tmp_path):
    result = local_msa_cli(
        tmp_path, "--max_msa_seqs", "1024", "--max_paired_msa_rows", "900"
    )
    assert "leaving few unpaired homologs" in result.output


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--msa_gpus", "0"),
        ("--msa_gpus", "-1"),
        ("--msa_batch_size", "0"),
        ("--mmseqs_sensitivity", "0"),
        ("--mmseqs_sensitivity", "-1"),
        ("--max_paired_msa_rows", "-1"),
    ],
)
def test_resource_options_reject_out_of_range_values(tmp_path, flag, value):
    result = local_msa_cli(tmp_path, flag, value)
    assert result.exit_code != 0
    assert "is not in the range" in result.output


def test_lookup_maps_database_keys_back_to_query_positions(tmp_path):
    # createdb --shuffle is on by default and reorders entries once there are
    # more than about 32, so a key is not a position in the FASTA. unpackdb
    # names its files by key, so reading one as the other hands every protein
    # another protein's MSA. Any test with 32 or fewer sequences passes either
    # way, which is why this one is deliberately larger.
    shuffled = tmp_path / "shuffled.lookup"
    n = 40
    order = [(i, (i * 7) % n) for i in range(n)]  # a permutation, not identity
    shuffled.write_text("".join(f"{k}\t{v}\t0\n" for k, v in order))

    mapping = read_lookup(shuffled, n)
    assert mapping != {i: i for i in range(n)}, "test data is not permuted"
    assert sorted(mapping.values()) == list(range(n))
    assert mapping[1] == 7

    identity = tmp_path / "identity.lookup"
    identity.write_text("".join(f"{i}\t{i}\t0\n" for i in range(n)))
    assert read_lookup(identity, n) == {i: i for i in range(n)}


def test_lookup_that_does_not_cover_the_queries_is_rejected(tmp_path):
    path = tmp_path / "short.lookup"
    path.write_text("0\t0\t0\n1\t1\t0\n")
    with pytest.raises(RuntimeError, match="does not describe"):
        read_lookup(path, 5)


@pytest.mark.skipif(shutil.which("mmseqs") is None, reason="needs mmseqs on PATH")
def test_createdb_key_order_matches_the_lookup(tmp_path):
    # Pins the real behaviour: 40 sequences is past the point where createdb
    # reorders, so this fails if the fork ever stops reading the lookup.
    n = 40
    fasta = tmp_path / "q.fasta"
    fasta.write_text("".join(f">{i}\n{'A' * (20 + i)}\n" for i in range(n)))
    db = tmp_path / "qdb"
    subprocess.run(
        ["mmseqs", "createdb", str(fasta), str(db), "--shuffle", "0"],
        check=True, capture_output=True,
    )  # fmt: skip

    mapping = read_lookup(Path(f"{db}.lookup"), n)
    assert sorted(mapping.values()) == list(range(n))


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


def fake_gpu(free, total, others=()):
    """Stand in for gpu_memory with a fixed reading."""
    return lambda device: (free, total, list(others))


def test_gpu_check_passes_on_an_idle_card(monkeypatch, caplog):
    # A desktop session holding a few hundred MiB must not stop a search.
    monkeypatch.delenv("BOLTZ_SKIP_GPU_CHECK", raising=False)
    monkeypatch.setattr(local_mmseqs, "gpu_memory", fake_gpu(31391, 32623))
    with caplog.at_level(logging.INFO):
        check_gpu_memory(["0"])
    assert "31391 of 32623" in caplog.text


def test_gpu_check_refuses_a_card_another_search_is_using(monkeypatch):
    # The failure this replaces was a bare CUDA OOM from inside libmarv,
    # raised minutes in and naming neither the device nor the other process.
    monkeypatch.delenv("BOLTZ_SKIP_GPU_CHECK", raising=False)
    monkeypatch.setattr(
        local_mmseqs,
        "gpu_memory",
        fake_gpu(2048, 32623, ["pid 4242 (mmseqs) holding 29000 MiB"]),
    )
    with pytest.raises(RuntimeError) as err:
        check_gpu_memory(["0"])
    message = str(err.value)
    assert "GPU 0" in message
    assert "2048 of 32623" in message
    assert "pid 4242 (mmseqs)" in message
    assert "BOLTZ_SKIP_GPU_CHECK" in message


def test_gpu_check_can_be_turned_off(monkeypatch, caplog):
    monkeypatch.setenv("BOLTZ_SKIP_GPU_CHECK", "1")
    monkeypatch.setattr(local_mmseqs, "gpu_memory", fake_gpu(0, 32623))
    with caplog.at_level(logging.WARNING):
        check_gpu_memory(["0"])
    assert "BOLTZ_SKIP_GPU_CHECK" in caplog.text

    # An unset-looking value leaves the check on.
    monkeypatch.setenv("BOLTZ_SKIP_GPU_CHECK", "0")
    with pytest.raises(RuntimeError):
        check_gpu_memory(["0"])


def test_gpu_check_continues_when_nvidia_smi_cannot_be_read(monkeypatch, caplog):
    # Advisory, not a gate: an unreadable card must not stop a run that
    # would have worked.
    monkeypatch.delenv("BOLTZ_SKIP_GPU_CHECK", raising=False)
    monkeypatch.setattr(local_mmseqs, "gpu_memory", lambda device: None)
    with caplog.at_level(logging.WARNING):
        check_gpu_memory(["0"])
    assert "nvidia-smi" in caplog.text


def test_gpu_check_names_the_first_busy_device_of_several(monkeypatch):
    readings = {"2": (31000, 32623, []), "3": (100, 32623, [])}
    monkeypatch.delenv("BOLTZ_SKIP_GPU_CHECK", raising=False)
    monkeypatch.setattr(local_mmseqs, "gpu_memory", lambda d: readings[d])
    with pytest.raises(RuntimeError, match="GPU 3"):
        check_gpu_memory(["2", "3"])


def test_log_messages_reach_stdout(capsys):
    # Boltz configures no logging, so without this the whole MSA step is
    # silent: the messages are emitted and discarded.
    package = logging.getLogger("boltz")
    before = list(package.handlers)
    try:
        package.handlers = []
        logging.getLogger("boltz.data.msa.local_mmseqs").info("hello")
        assert "hello" not in capsys.readouterr().out

        log_to_stdout(logging.INFO)
        logging.getLogger("boltz.data.msa.local_mmseqs").info("hello")
        assert "hello" in capsys.readouterr().out

        # Called twice, every line must still appear exactly once.
        log_to_stdout(logging.INFO)
        logging.getLogger("boltz.data.msa.local_mmseqs").info("once")
        assert capsys.readouterr().out.count("once") == 1
    finally:
        package.handlers = before
        package.propagate = True


def test_log_setup_leaves_other_packages_alone():
    # Turning on INFO for the MSA step must not turn on INFO for torch,
    # lightning and urllib3, which is what basicConfig would have done.
    # Checked on the root logger, not on torch's: under pytest every logger
    # carries pytest's own capture handlers, so counting those proves nothing.
    root = logging.getLogger()
    package = logging.getLogger("boltz")
    root_level, root_handlers = root.level, list(root.handlers)
    before, level = list(package.handlers), package.level
    try:
        log_to_stdout(logging.INFO)
        assert root.level == root_level, "root logger level was changed"
        assert list(root.handlers) == root_handlers, "handler added to root"
        assert any(
            getattr(h, "_boltz_msa_handler", False) for h in package.handlers
        ), "handler was not added to the boltz logger"
    finally:
        package.handlers, package.propagate = before, True
        package.setLevel(level)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.5, "0.5 s"), (89, "89.0 s"), (90, "1.5 min"), (3600, "60.0 min"),
     (5400, "1.50 h")],
)
def test_elapsed_switches_unit(seconds, expected):
    assert elapsed(seconds) == expected


def test_describe_hits_reports_the_median_not_the_mean():
    # Counts are capped per database, so queries sitting at the cap would
    # drag a mean away from what a typical query got.
    hits = {0: [1] * 2, 1: [1] * 3, 2: [1] * 10000}
    summary = describe_hits(hits)
    assert "median 3/query" in summary
    assert "range 2-10000" in summary
    assert "10005 hits kept" in summary


def test_describe_hits_counts_queries_with_nothing():
    assert "1 query with none" in describe_hits({0: [], 1: [1]})
    assert "2 queries with none" in describe_hits({0: [], 1: [], 2: [1]})
    assert describe_hits({}) == "no queries"


def test_describe_depths_flags_shallow_msas():
    summary = describe_depths([1, 2, 500, 4096])
    assert "median 500 rows" in summary
    assert "range 1-4096" in summary
    assert "2 of 4 below 10 rows" in summary

    assert "below" not in describe_depths([50, 500])
    assert describe_depths([]) == "no MSAs written"


def test_heartbeat_reports_while_a_step_runs(caplog):
    with caplog.at_level(logging.INFO, logger="boltz.data.msa.local_mmseqs"):
        with heartbeat("uniref90 search", interval=0.05):
            time.sleep(0.3)
    assert "still running after" in caplog.text
    assert "uniref90 search" in caplog.text


def test_heartbeat_stops_when_the_step_finishes(caplog):
    with heartbeat("x", interval=0.05):
        pass
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="boltz.data.msa.local_mmseqs"):
        time.sleep(0.3)
    assert "still running" not in caplog.text


def test_heartbeat_interval_is_read_at_call_time(monkeypatch, caplog):
    # A default argument would freeze HEARTBEAT_SECONDS at import, so setting
    # it would silently do nothing -- which is exactly what happened.
    monkeypatch.setattr(local_mmseqs, "HEARTBEAT_SECONDS", 0.05)
    with caplog.at_level(logging.INFO, logger="boltz.data.msa.local_mmseqs"):
        with heartbeat("uniref90 search"):
            time.sleep(0.3)
    assert "still running after" in caplog.text


def test_heartbeat_can_be_switched_off(caplog):
    with caplog.at_level(logging.INFO, logger="boltz.data.msa.local_mmseqs"):
        with heartbeat("x", interval=0):
            time.sleep(0.15)
    assert "still running" not in caplog.text


def test_heartbeat_does_not_swallow_an_error():
    with pytest.raises(ValueError, match="boom"):
        with heartbeat("x", interval=0.05):
            raise ValueError("boom")


def test_gpu_memory_returns_nothing_without_nvidia_smi(monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert gpu_memory("0") is None


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
