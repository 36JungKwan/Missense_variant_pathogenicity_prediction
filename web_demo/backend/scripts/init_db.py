import sys
import os

# Thêm thư mục backend vào sys.path để import được các module trong app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.db.session import engine, Base
from app.models.variant import Variant, Prediction, InferenceFeature  # noqa: F401 - đăng ký bảng vào metadata

def init_db():
    print("Bắt đầu khởi tạo Database RB-Net...")

    with engine.connect() as conn:
        # Bắt buộc phải kích hoạt extension pgvector trong PostgreSQL
        print("Đang kích hoạt pgvector extension...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    # Xây dựng toàn bộ các bảng vào Database
    print("Đang khởi tạo các bảng (variants, predictions, inference_features)...")
    Base.metadata.create_all(bind=engine)

    # Nâng cấp không phá dữ liệu cho database đã tồn tại (create_all không thêm cột vào bảng cũ).
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS model_tag VARCHAR(128);"))
        # Bảng 768 chiều của pipeline mock cũ; pipeline Pure DL dùng inference_features (1280 chiều).
        conn.execute(text("DROP TABLE IF EXISTS extracted_features;"))
        conn.commit()

    print("Khởi tạo thành công! Database đã sẵn sàng.")

if __name__ == "__main__":
    init_db()
