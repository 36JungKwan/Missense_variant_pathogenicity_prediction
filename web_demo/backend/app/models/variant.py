from sqlalchemy import LargeBinary, Column, String, Float, BigInteger, SmallInteger, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.db.session import Base
from datetime import datetime

class Variant(Base):
    __tablename__ = "variants"

    # Định dạng mã GRCh38, ví dụ: 'chr7:140753336:A>T'
    variant_id = Column(String(64), primary_key=True, index=True)
    gene = Column(String(32), index=True)
    
    # Bắt buộc chứa từ khóa 'missense' theo quy tắc Gatekeeper
    consequence = Column(String(128), nullable=False)
    protein_change = Column(String(64))
    
    # Nhãn chuẩn nhị phân cứng: 'Pathogenic' hoặc 'Benign'
    clinvar_ground_truth = Column(String(16), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Khai báo relationship để dễ truy vấn ORM
    predictions = relationship("Prediction", back_populates="variant", cascade="all, delete-orphan")
    inference_features = relationship("InferenceFeature", back_populates="variant", uselist=False, cascade="all, delete-orphan")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    variant_id = Column(String(64), ForeignKey("variants.variant_id", ondelete="CASCADE"), nullable=False)
    
    # Giá trị từ 0 đến 15 đại diện cho các tổ hợp Tắt/Bật tính năng
    ablation_mask = Column(SmallInteger, nullable=False)
    
    trimvpp_prob = Column(Float, nullable=False)
    classification = Column(String(16), nullable=False)
    
    sota_alphamissense = Column(Float, nullable=True)
    sota_revel = Column(Float, nullable=True)
    
    # Integrated Gradients (full_prediction) của checkpoint ứng với mask này: theo nhánh, token, đặc trưng.
    # NULL cho tới khi trang XAI của biến thể được mở lần đầu (tính khi cần rồi lưu lại).
    ig_values = Column(JSONB, nullable=True)

    # Định danh checkpoint đã sinh ra dòng này ("<file>.pth@<mtime>"); NULL = dữ liệu seed/mock cũ.
    model_tag = Column(String(128), nullable=True)

    variant = relationship("Variant", back_populates="predictions")

    # Composite Index cực kỳ quan trọng giúp query 64 triệu dòng chỉ trong mili-giây
    __table_args__ = (
        Index("ix_variant_mask", "variant_id", "ablation_mask", unique=True),
    )

class InferenceFeature(Base):
    """Đặc trưng thật của pipeline Pure DL (NT-v1 500M / ESM-1b 650M, 1280 chiều) cho mỗi biến thể.

    Lưu lại để đổi tổ hợp ablation không phải chạy lại VEP và Foundation Models.
    """
    __tablename__ = "inference_features"

    variant_id = Column(String(64), ForeignKey("variants.variant_id", ondelete="CASCADE"), primary_key=True)
    extractor_tag = Column(String(128), nullable=False)

    emb_dna_ref = Column(Vector(1280), nullable=False)
    emb_dna_alt = Column(Vector(1280), nullable=False)
    emb_pro_ref = Column(Vector(1280), nullable=False)
    emb_pro_alt = Column(Vector(1280), nullable=False)

    # emb_*: vector token tâm (cho geometry). tok_*: cửa sổ 33 token, float16 little-endian (2 allele, 33, 1280).
    # tok_mask: uint8 (2 nhánh dna/prot, 2 allele, 33).
    tok_dna = Column(LargeBinary, nullable=False)
    tok_pro = Column(LargeBinary, nullable=False)
    tok_mask = Column(LargeBinary, nullable=False)

    # Giá trị thô (chưa chuẩn hoá): 11 bio core + 9 SpliceAI + {dna,prot}_{LLR,LVD_L2,LVD_Cosine,LID,LVD_Relative}
    raw_features = Column(JSONB, nullable=False)
    # Chú thích VEP: transcript, ENSP, HGVSp, AlphaMissense, REVEL, ClinVar thô, cảnh báo đặc trưng
    annotation = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    variant = relationship("Variant", back_populates="inference_features")
