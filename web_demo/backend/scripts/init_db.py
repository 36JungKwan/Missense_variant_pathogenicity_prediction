import sys
import os

# Thêm thư mục backend vào sys.path để import được các module trong app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.db.session import engine, Base
from app.models.variant import Variant, Prediction, InferenceFeature  # noqa: F401 - đăng ký bảng vào metadata

def init_db():
    print("Bắt đầu khởi tạo Database TriMVPP...")

    with engine.connect() as conn:
        # Bắt buộc phải kích hoạt extension pgvector trong PostgreSQL
        print("Đang kích hoạt pgvector extension...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    # Schema trước TriMVPP (không có tok_dna): đặc trưng và dự đoán cũ không dùng lại được -> xoá rồi tạo lại.
    # Bảng variants được giữ nguyên.
    with engine.connect() as conn:
        exists = conn.execute(text("SELECT to_regclass('public.inference_features')")).scalar()
        current = conn.execute(text(
            "SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' "
            "AND table_name = 'inference_features' AND column_name = 'tok_dna'")).first()
        if exists and not current:
            print("Schema cũ: xoá bảng predictions và inference_features...")
            conn.execute(text("DROP TABLE IF EXISTS predictions, inference_features;"))
            conn.commit()
        # Bảng predictions lưu SHAP (shap_values) cũ: đổi sang Integrated Gradients (ig_values) -> xoá, tạo lại.
        # Đặc trưng đã trích xuất (inference_features) vẫn dùng lại được nên được giữ.
        old_shap = conn.execute(text(
            "SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' "
            "AND table_name = 'predictions' AND column_name = 'shap_values'")).first()
        if old_shap:
            print("Bảng predictions còn cột shap_values: xoá để tạo lại với cột ig_values...")
            conn.execute(text("DROP TABLE IF EXISTS predictions;"))
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
