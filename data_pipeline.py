"""Воспроизводимая подготовка небольшого среза Yambda-50M."""

from pathlib import Path
import os

import duckdb
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ID = "yandex/yambda"
REVISION = "eb2b2586d41846153dfb8d0fd7f63e9509fb5b21"
SEED = 42
USER_SHARE = 0.15
WINDOW_DAYS = 150
CATALOG_SIZE = 3000
SOURCE_FILES = {
    "events": "flat/50m/multi_event.parquet",
    "artists": "artist_item_mapping.parquet",
    "albums": "album_item_mapping.parquet",
}


def work_directory():
    configured = os.environ.get("YAMBDA_WORK_DIR")
    path = Path(configured).expanduser() if configured else Path(__file__).parent / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_paths(work):
    return {
        key: Path(hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            filename=filename,
            cache_dir=str(work / "hf_cache"),
        ))
        for key, filename in SOURCE_FILES.items()
    }


def _sql_path(path):
    return str(path).replace("'", "''")


def prepare(work=None):
    """Скачать источники и создать локальные файлы среза; возвращает их пути и границы."""
    work = Path(work) if work else work_directory()
    work.mkdir(parents=True, exist_ok=True)
    paths = source_paths(work)
    db = duckdb.connect(str(work / "analysis.duckdb"))
    db.execute("SET memory_limit='4GB'")
    db.execute("SET threads=2")
    db.execute(f"CREATE OR REPLACE VIEW source AS SELECT * FROM read_parquet('{_sql_path(paths['events'])}')")

    users = db.execute("SELECT DISTINCT uid FROM source ORDER BY uid").fetchnumpy()["uid"]
    selected = np.sort(np.random.default_rng(SEED).choice(
        users, size=round(len(users) * USER_SHARE), replace=False
    ))
    db.register("selected_users", pd.DataFrame({"uid": selected}))
    cohort = work / f"cohort_{len(selected)}_users_seed_{SEED}.parquet"
    if not cohort.exists():
        db.execute(f"""COPY (
            SELECT s.* FROM source s SEMI JOIN selected_users u USING(uid)
        ) TO '{_sql_path(cohort)}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    db.execute(f"CREATE OR REPLACE VIEW cohort AS SELECT * FROM read_parquet('{_sql_path(cohort)}')")
    db.execute("CREATE OR REPLACE VIEW cohort_clean AS SELECT DISTINCT * FROM cohort")

    t_finish = int(db.execute('SELECT MAX("timestamp") FROM cohort').fetchone()[0])
    t_start = t_finish - WINDOW_DAYS * 86400
    t_train = t_start + int(0.70 * (t_finish - t_start))
    t_val = t_start + int(0.85 * (t_finish - t_start))
    db.execute(f"""CREATE OR REPLACE TABLE catalog AS
        WITH popularity AS (
            SELECT item_id, COUNT(DISTINCT uid) AS n_users, COUNT(*) AS n_events
            FROM cohort_clean WHERE "timestamp">={t_start} AND "timestamp"<{t_train}
              AND event_type='listen' AND played_ratio_pct>50
            GROUP BY item_id
        )
        SELECT item_id, n_users, n_events FROM popularity
        ORDER BY n_users DESC, n_events DESC, item_id LIMIT {CATALOG_SIZE}""")

    events_file = work / "prepared_events.parquet"
    db.execute(f"""COPY (
        SELECT c.* FROM cohort_clean c SEMI JOIN catalog k USING(item_id)
        WHERE c."timestamp">={t_start} AND c."timestamp"<={t_finish}
    ) TO '{_sql_path(events_file)}' (FORMAT PARQUET, COMPRESSION ZSTD)""")

    db.execute(f"CREATE OR REPLACE VIEW artists AS SELECT * FROM read_parquet('{_sql_path(paths['artists'])}')")
    db.execute(f"CREATE OR REPLACE VIEW albums AS SELECT * FROM read_parquet('{_sql_path(paths['albums'])}')")
    metadata = db.execute("""
        WITH artist_one AS (
            SELECT item_id, MIN(artist_id) AS artist_id, COUNT(*) AS n_artists
            FROM artists SEMI JOIN catalog USING(item_id) GROUP BY item_id
        ), album_counts AS (
            SELECT album_id, COUNT(DISTINCT item_id) AS catalog_tracks
            FROM albums SEMI JOIN catalog USING(item_id) GROUP BY album_id
        ), album_candidates AS (
            SELECT a.item_id, a.album_id, c.catalog_tracks,
                   COUNT(*) OVER (PARTITION BY a.item_id) AS n_albums,
                   ROW_NUMBER() OVER (PARTITION BY a.item_id
                       ORDER BY c.catalog_tracks DESC, a.album_id) AS rank
            FROM albums a JOIN catalog USING(item_id)
            JOIN album_counts c USING(album_id)
        )
        SELECT k.item_id, k.n_users AS train_listeners, k.n_events AS train_positive_events,
               ar.artist_id, ar.n_artists, al.album_id, al.n_albums
        FROM catalog k LEFT JOIN artist_one ar USING(item_id)
        LEFT JOIN album_candidates al ON k.item_id=al.item_id AND al.rank=1
        ORDER BY k.item_id
    """).df()
    meta_file = work / "prepared_metadata.parquet"
    db.register("metadata_result", metadata)
    db.execute(f"COPY metadata_result TO '{_sql_path(meta_file)}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    info = {
        "events_file": str(events_file), "metadata_file": str(meta_file),
        "cohort_file": str(cohort), "source_files": {k: str(v) for k, v in paths.items()},
        "source_users": len(users), "selected_users": len(selected),
        "t_start": t_start, "t_train": t_train, "t_val": t_val,
        "t_finish": t_finish, "seed": SEED, "catalog_size": CATALOG_SIZE,
    }
    db.close()
    return info


def load_prepared(info):
    db = duckdb.connect()
    events = db.execute("SELECT * FROM read_parquet(?)", [info["events_file"]]).df()
    metadata = db.execute("SELECT * FROM read_parquet(?)", [info["metadata_file"]]).df()
    db.close()
    events["event_type"] = events["event_type"].astype("category")
    return events, metadata
