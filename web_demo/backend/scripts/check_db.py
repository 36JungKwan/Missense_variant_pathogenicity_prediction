"""Kiểm tra sức khoẻ PostgreSQL của RB-Net (chỉ đọc).

Chạy:  python scripts/check_db.py
Thoát mã 1 nếu có lỗi (mất kết nối, thiếu bảng/pgvector, vector sai chiều, dòng mồ côi...).
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from app.db.session import engine
from app.services.checkpoint_registry import checkpoint_registry
from app.services.feature_pipeline import EXTRACTOR_TAG
from app.services.inference_service import _prediction_tag

EXPECTED_TABLES = ("variants", "predictions", "inference_features")
EMBEDDING_COLUMNS = ("emb_dna_ref", "emb_dna_alt", "emb_pro_ref", "emb_pro_alt")
EMBEDDING_DIM = 1280


def main() -> int:
    errors, notes = [], []
    try:
        conn = engine.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Cannot connect to database: {exc}")
        return 1

    with conn:
        q = lambda sql, **kw: conn.execute(text(sql), kw)  # noqa: E731

        if q("SELECT 1 FROM pg_extension WHERE extname = 'vector'").first() is None:
            errors.append("pgvector extension is not installed")

        tables = {r[0] for r in q("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
        missing = [t for t in EXPECTED_TABLES if t not in tables]
        if missing:
            errors.append(f"missing tables: {missing} (run scripts/init_db.py)")
            for line in errors:
                print(f"[FAIL] {line}")
            return 1
        if "extracted_features" in tables:
            notes.append("legacy table extracted_features still exists (run scripts/init_db.py to drop it)")

        counts = {t: q(f"SELECT count(*) FROM {t}").scalar() for t in EXPECTED_TABLES}
        print("Row counts: " + ", ".join(f"{t}={n}" for t, n in counts.items()))

        for col in EMBEDDING_COLUMNS:
            bad = q(f"SELECT count(*) FROM inference_features WHERE vector_dims({col}) <> :d", d=EMBEDDING_DIM).scalar()
            if bad:
                errors.append(f"{bad} rows with {col} dimension != {EMBEDDING_DIM}")

        stale_fx = q("SELECT count(*) FROM inference_features WHERE extractor_tag <> :t", t=EXTRACTOR_TAG).scalar()
        if stale_fx:
            notes.append(f"{stale_fx} feature rows from another extractor (current {EXTRACTOR_TAG}); re-extracted on demand")

        lid_missing = q(
            "SELECT count(*) FROM inference_features "
            "WHERE raw_features->'dna_LID' = 'null' OR raw_features->'prot_LID' = 'null'"
        ).scalar()
        if lid_missing:
            notes.append(f"{lid_missing} feature rows have imputed LID; recomputed on next request once FAISS indexes exist")

        orphans = q(
            "SELECT count(*) FROM variants v WHERE NOT EXISTS "
            "(SELECT 1 FROM inference_features f WHERE f.variant_id = v.variant_id)"
        ).scalar()
        if orphans:
            notes.append(f"{orphans} variants without inference_features (legacy/seed rows)")

        try:
            current_tag = _prediction_tag(checkpoint_registry.get(("dna", "prot", "bio", "geom")))
            fresh = q("SELECT count(*) FROM predictions WHERE model_tag = :t", t=current_tag).scalar()
            print(f"Predictions from current checkpoint ({current_tag}): {fresh}/{counts['predictions']}")
        except Exception as exc:  # noqa: BLE001 - checkpoint thiếu không phải lỗi DB
            notes.append(f"cannot resolve the 111 checkpoint: {exc}")

        untagged = q("SELECT count(*) FROM predictions WHERE model_tag IS NULL").scalar()
        if untagged:
            notes.append(f"{untagged} predictions without model_tag (old seed/mock data, never served from cache)")

    for line in notes:
        print(f"[NOTE] {line}")
    for line in errors:
        print(f"[FAIL] {line}")
    print("[OK] Database healthy" if not errors else "[FAIL] Database has problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
