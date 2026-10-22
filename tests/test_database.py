import duckdb
import pytest

from fantasy_ml.data.database import (
    get_database_summary,
    initialize_database,
)


def make_parquet(path, values):
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            "CREATE TABLE sample AS SELECT * FROM "
            "(VALUES " + ", ".join(["(?)"] * len(values)) +
            ") AS t(value)",
            values,
        )
        con.execute(
            "COPY sample TO ? (FORMAT PARQUET)",
            [str(path)],
        )
    finally:
        con.close()


def test_initialization_imports_data_and_records_metadata(tmp_path):
    source = tmp_path / "sample.parquet"
    database = tmp_path / "fantasy.duckdb"
    make_parquet(source, [10, 20, 30])

    report = initialize_database(
        database, {"sample_stats": source}
    )

    assert report["sample_stats"]["status"] == "imported"
    assert report["sample_stats"]["rows"] == 3

    summary = get_database_summary(database)
    assert summary["sample_stats"]["rows"] == 3


def test_repeated_initialization_skips_unchanged_source(tmp_path):
    source = tmp_path / "sample.parquet"
    database = tmp_path / "fantasy.duckdb"
    make_parquet(source, [10, 20, 30])

    initialize_database(database, {"sample_stats": source})
    second = initialize_database(database, {"sample_stats": source})

    assert second["sample_stats"]["status"] == "skipped"
    assert second["sample_stats"]["rows"] == 3


def test_changed_source_replaces_table_without_duplicates(tmp_path):
    source = tmp_path / "sample.parquet"
    database = tmp_path / "fantasy.duckdb"
    make_parquet(source, [10, 20, 30])

    initialize_database(database, {"sample_stats": source})
    make_parquet(source, [10, 20, 30, 40, 50])

    # Ensure the filesystem modification time changes even on
    # filesystems with coarse timestamp resolution.
    import os
    os.utime(source, ns=(source.stat().st_atime_ns,
                         source.stat().st_mtime_ns + 2_000_000_000))

    updated = initialize_database(database, {"sample_stats": source})

    assert updated["sample_stats"]["status"] == "imported"
    assert updated["sample_stats"]["rows"] == 5


def test_missing_source_raises_without_creating_a_data_table(tmp_path):
    database = tmp_path / "fantasy.duckdb"
    missing = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError):
        initialize_database(database, {"sample_stats": missing})

    con = duckdb.connect(str(database), read_only=True)
    try:
        exists = con.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = 'sample_stats'
        """).fetchone()[0]
        assert exists == 0
    finally:
        con.close()
