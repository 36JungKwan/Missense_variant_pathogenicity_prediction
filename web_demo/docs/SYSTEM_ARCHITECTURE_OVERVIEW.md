# TriMVPP Variant Pathogenicity Classification System — System Architecture Overview

> Tài liệu đặc tả kỹ thuật trích từ mã nguồn tại `F:/Demo` (cập nhật ngày 26/11/2026, sau khi tích hợp 7 checkpoint TriMVPP và chuyển phần giải thích sang Integrated Gradients + attention map).
> Mục đích: nguồn dữ liệu cho **Chapter 5 — System Design and Implementation** (`docs/chapter5_trimvpp.tex`) và phần minh hoạ giao diện.
> Tên hàm/biến/hằng số được dẫn kèm đường dẫn tệp để tiện đối chiếu. Các điểm cần lưu ý khi viết luận văn được gom ở [Mục 7](#7-ghi-chú-đối-chiếu--các-điểm-cần-lưu-ý-khi-viết-luận-văn).

---

## Mục lục

1. [Tech Stack & Project Directory Structure](#1-tech-stack--project-directory-structure)
2. [Database Schema (PostgreSQL 16 + pgvector)](#2-database-schema-postgresql-16--pgvector)
3. [Backend API Contracts & Communication Protocols](#3-backend-api-contracts--communication-protocols)
4. [End-to-End Inference & Hybrid Lazy-Caching Pipeline](#4-end-to-end-inference--hybrid-lazy-caching-pipeline)
5. [Frontend Architecture & User Workflows](#5-frontend-architecture--user-workflows)
6. [Deployment Topology & Performance Benchmarks](#6-deployment-topology--performance-benchmarks)
7. [Ghi chú đối chiếu & các điểm cần lưu ý khi viết luận văn](#7-ghi-chú-đối-chiếu--các-điểm-cần-lưu-ý-khi-viết-luận-văn)

---

## 1. Tech Stack & Project Directory Structure

### 1.1 Backend (Python 3.10.0, virtualenv `backend/.venv`)

| Nhóm | Thư viện | Phiên bản | Vai trò trong hệ thống |
|---|---|---|---|
| Web framework | `fastapi` | 0.115.0 | REST API, dependency injection, OpenAPI `/docs` |
| | `starlette` | 0.38.6 (phụ thuộc của FastAPI) | Middleware, exception handler |
| | `uvicorn[standard]` | 0.30.6 | ASGI server (cổng 8000) |
| | `sse-starlette` | 2.1.3 | `EventSourceResponse` cho Batch Screening |
| | `python-multipart` | 0.0.12 | Nhận `UploadFile` + `Form` |
| Validation / config | `pydantic` | 2.9.2 | Request/response schema |
| | `pydantic-settings` | 2.5.2 | `Settings` đọc `backend/.env` |
| ORM / DB driver | `SQLAlchemy` | 2.0.35 | ORM + `INSERT … ON CONFLICT` (dialect PostgreSQL) |
| | `psycopg2-binary` | 2.9.9 | Driver PostgreSQL |
| | `pgvector` (Python) | 0.3.5 | Kiểu cột `Vector(1280)` |
| Deep learning | `torch` | 2.4.1+cu121 (CUDA 12.1 wheel) | Foundation models (FP16), TriMVPP, Integrated Gradients |
| | `transformers` | 4.57.5 | `AutoModelForMaskedLM`, `AutoTokenizer` |
| | `tokenizers` | 0.22.2 | Fast tokenizer |
| | `safetensors` | 0.8.0 | Định dạng trọng số FM trong HF cache |
| Vector search | `faiss-cpu` | 1.11.0 | `IndexFlatL2` để tính LID |
| ML classic | `scikit-learn` | 1.7.0 | Scaler/Imputer đã fit (unpickle) |
| | `joblib` | 1.5.1 | Nạp các file `.pkl` |
| Dữ liệu | `numpy` 1.26.4, `pandas` 2.2.3, `pyarrow` 17.0.0, `openpyxl` 3.1.5 | | Chuẩn hoá đặc trưng, đọc bảng upload |
| Genomics I/O | `pyfaidx` | 0.9.0.3 | Truy xuất FASTA hg38 + proteome Ensembl |

Đầu đọc tabix cho SpliceAI là **mã tự viết** (`NativeTabixReader` trong `app/ml_core/spliceai_lookup.py`), đọc trực tiếp các block BGZF qua `.tbi`, **không** phụ thuộc `pysam`/`tabix`. Phần giải thích (Integrated Gradients, attention) cũng **tự cài đặt** bằng PyTorch (`app/ml_core/xai.py`), không dùng `captum` hay `shap`.

### 1.2 Frontend (Node.js v22.21.1)

| Nhóm | Package | Phiên bản | Ghi chú |
|---|---|---|---|
| UI runtime | `react`, `react-dom` | 19.3.1 | `StrictMode` trong `main.tsx` |
| Routing | `react-router-dom` | 7.18.5 | `BrowserRouter` |
| Charts | `recharts` | 3.10.1 | Biểu đồ IG theo đặc trưng (trang XAI); heatmap/dải token vẽ bằng SVG tự viết |
| Icons | `lucide-react` | 1.70.0 | |
| Styling | `tailwindcss` | 3.4.19 | + `postcss`, `autoprefixer` |
| | `clsx`, `tailwind-merge` | | helper `cn()` trong `lib/utils.ts` |
| Build | `vite` | 8.x | dev server cổng 5173; alias `@ → ./src` |
| | `typescript` | 7.0.x | `tsc -b && vite build` |
| Lint | `oxlint` | 1.9x | `npm run lint` |

Các primitive UI (`button`, `card`, `badge`, `progress`, `switch`, `table`) là component tự viết theo phong cách shadcn/ui trong `src/components/ui/`.

### 1.3 Database

| Thành phần | Giá trị |
|---|---|
| Image | `pgvector/pgvector:pg16` (PostgreSQL 16 + extension `vector`) |
| Container | `rbnet_postgres` (tên container giữ từ phiên bản trước) |
| Database / user | `rbnet_genomics` / `rbnet_user` |
| Cổng | host `5433` → container `5432` |
| Volume | `postgres_data` (named volume) |
| Khởi tạo | `scripts/init_db.py`: `CREATE EXTENSION IF NOT EXISTS vector;` + tạo bảng + migration |

### 1.4 Bioinformatics tools, models & reference data

| Thành phần | Chi tiết |
|---|---|
| **Ensembl VEP** | Container `rbnet_vep`, image `ensemblorg/ensembl-vep:latest`, cache **release 116**, chạy `--offline` |
| VEP plugin | **dbNSFP 5.3.1a** (`/vep_resources/dbNSFP5.3.1a_grch38.gz`) |
| Genome FASTA | `Homo_sapiens.GRCh38.dna.primary_assembly.fa` (host `F:/VEP_RESOURCES`) |
| Proteome FASTA | `Homo_sapiens.GRCh38.pep.all.fa` (Ensembl peptide) |
| SpliceAI | `spliceai_scores.raw.snv.ensembl_mane_v1.4.grch38.vcf.gz` + `.tbi` (~28 GB, tra cứu theo tabix) |
| DNA foundation model | `InstaDeepAI/nucleotide-transformer-500m-human-ref` (NT-v1 500M, hidden 1280) |
| Protein foundation model | `facebook/esm1b_t33_650M_UR50S` (ESM-1b 650M, hidden 1280) |
| Classifier | **TriMVPP** (Tri-Modal Variant Pathogenicity Predictor), 7 checkpoint `artifacts/trimvpp/checkpoints/trimvpp_{111,110,101,011,100,010,001}.pth` |
| FAISS indexes | `artifacts/trimvpp/faiss_indexes/{nt_v1_500m_dna_1001_center, esm1b_650m_prot_501_center}.index` — `IndexFlatL2`, d = 1280, **ntotal = 23 208** (embedding displacement của tập train) |
| Scalers | 6 file `artifacts/bio_*.pkl` + 2 file `artifacts/trimvpp/geom_scalers/*_geom_v2_scalers.pkl` (bộ `geometry_v2_k64_step2_matched`, fit trên train3) |
| IG baseline | `artifacts/trimvpp/ig_baseline_train3_seed2026.pt` (của notebook 19 nhóm) |

### 1.5 Cây thư mục dự án (không gồm `.venv`, `node_modules`, `__pycache__`)

```
F:/Demo/
├── docker-compose.yml               # service "db" → container rbnet_postgres
├── start_system.bat                 # launcher 3 bước: DB → Backend → Frontend
├── stop_system.bat                  # dừng container DB
├── docs/
│   ├── chapter5_trimvpp.tex         # Chương 5 (bản hiện hành) + chapter5_trimvpp_shap_backup.tex (bản SHAP cũ)
│   ├── figures/                     # 5.1–5.3 (.svg + .png), 5.4a–e (ảnh chụp giao diện), 5.4 (sơ đồ DB cũ)
│   └── SYSTEM_ARCHITECTURE_OVERVIEW.md   # tài liệu này
├── backend/
│   ├── .env                         # cấu hình runtime (không có .env.example)
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py                  # FastAPI app (v2.0.0), lifespan preload, middleware, exception handlers
│   │   ├── api/v1/
│   │   │   ├── router.py            # gộp router: health, /predict, /xai
│   │   │   └── endpoints/
│   │   │       ├── health.py        # GET  /health
│   │   │       ├── predict.py       # POST /predict/single, GET /predict/batch-stream, POST /predict/batch-upload
│   │   │       └── xai.py           # GET  /xai/{variant_id}  (IG + attention)
│   │   ├── core/
│   │   │   ├── config.py            # Settings, registry 7 checkpoint, ngưỡng τ, MCC test, tham số IG
│   │   │   ├── constants.py         # cột đặc trưng, bitmask ablation, parse_variant_id, nhãn ClinVar
│   │   │   ├── errors.py            # PipelineError + 10 lớp lỗi nghiệp vụ → (HTTP status, error_code)
│   │   │   └── log.py               # logging UTF-8 + request_id + đo thời gian từng chặng
│   │   ├── db/session.py            # engine, SessionLocal, get_db()
│   │   ├── models/variant.py        # ORM: Variant, Prediction, InferenceFeature
│   │   ├── schemas/                 # common.py (IG/attention), single.py (+ XaiResponse), batch.py
│   │   ├── services/
│   │   │   ├── inference_service.py # điều phối: cache → VEP → gatekeeper → đặc trưng → TriMVPP + IG → upsert; run_xai
│   │   │   ├── vep_runner.py        # docker exec rbnet_vep vep …, parse TSV, Missense Gatekeeper
│   │   │   ├── sequence_service.py  # cửa sổ DNA 1001 bp / Protein 501 aa, kiểm tra REF
│   │   │   ├── feature_pipeline.py  # FoundationModelHub, SpliceAI, geometry, chuẩn hoá, preload, EXTRACTOR_TAG
│   │   │   ├── checkpoint_registry.py # khoá 3 bit → trimvpp_{key}.pth, nạp strict, cache model
│   │   │   ├── ml_engine.py         # xác suất + IG (mỗi lần dự đoán), attention (khi mở XAI)
│   │   │   └── upload_parser.py     # .vcf/.txt/.csv/.parquet/.xlsx → danh sách variant_id
│   │   └── ml_core/
│   │       ├── models.py            # BioModelManager (tải FM ở FP16, map token đột biến)
│   │       ├── fm_extractor.py      # FeatureExtractor: cửa sổ 33 token ref/alt của lớp ẩn cuối
│   │       ├── geometry_v2.py       # LVD_L2, LVD_Cosine, LVD_Relative, LID (Levina–Bickel)
│   │       ├── normalizer.py        # transform_bio / transform_geom_v2
│   │       ├── spliceai_lookup.py   # IndexedSpliceAILookup + NativeTabixReader
│   │       ├── trimvpp_model.py     # TriMVPP (cross-attention 3 nhánh), return_attention
│   │       └── xai.py               # integrated_gradients, explain_ig, attention_maps, load_baseline
│   ├── artifacts/
│   │   ├── bio_{freq,mean_imputer,phast,phylo,spliceai_fill_values,spliceai}_*.pkl
│   │   └── trimvpp/
│   │       ├── checkpoints/trimvpp_{111,110,101,011,100,010,001}.pth
│   │       ├── geom_scalers/{nt_v1_500m_dna_1001_center,esm1b_650m_prot_501_center}_geom_v2_scalers.pkl
│   │       ├── faiss_indexes/{nt_v1_500m_dna_1001_center,esm1b_650m_prot_501_center}.index
│   │       └── ig_baseline_train3_seed2026.pt
│   └── scripts/
│       ├── init_db.py               # extension + 3 bảng, migration (xoá bảng schema cũ)
│       ├── check_db.py              # kiểm tra sức khoẻ DB (chỉ đọc)
│       └── download_models.py       # tải FM về HF_HOME, chuyển .bin → safetensors
└── frontend/
    └── src/
        ├── main.tsx, App.tsx, index.css
        ├── pages/            Home.tsx, Single.tsx, Batch.tsx, Xai.tsx
        ├── components/
        │   ├── layout/MainLayout.tsx
        │   ├── ablation/     FeatureAblationPanel.tsx (3 công tắc + dòng MCC test)
        │   ├── xai/          IgModalityBars, IgTokenStrip, IgFeatureChart, AttentionPanel
        │   ├── variant/      FeatureInspectorTabs.tsx (chưa dùng)
        │   └── ui/           badge, button, card, progress, switch, table
        ├── lib/              ablation.ts (bitmask), xai.ts (thang màu, nhãn), utils.ts
        ├── services/         api.ts (client thật), api.mock.ts (VITE_USE_MOCK_API=true)
        ├── types/index.ts    # mirror schema Pydantic
        └── mock/mockData.json
```

---

## 2. Database Schema (PostgreSQL 16 + pgvector)

Nguồn: `backend/app/models/variant.py`, `backend/scripts/init_db.py`, `backend/app/services/inference_service.py`.

### 2.1 Sơ đồ quan hệ

```
variants (1) ──< predictions (N ≤ 7)        FK predictions.variant_id → variants.variant_id  ON DELETE CASCADE
variants (1) ──  inference_features (0..1)  FK/PK inference_features.variant_id → variants.variant_id ON DELETE CASCADE
```

- `variants` ↔ `predictions`: 1–N, tối đa **7 dòng** (một dòng cho mỗi tổ hợp đặc trưng hợp lệ).
- `variants` ↔ `inference_features`: 1–1; một dòng đặc trưng dùng chung cho cả 7 tổ hợp.

### 2.2 Bảng `variants`

| Cột | Kiểu PostgreSQL | Ràng buộc | Ý nghĩa |
|---|---|---|---|
| `variant_id` | `VARCHAR(64)` | **PRIMARY KEY**, index | `chr{CHROM}:{POS}:{REF}>{ALT}` (GRCh38; ty thể = `chrM`) |
| `gene` | `VARCHAR(32)` | index | SYMBOL từ VEP |
| `consequence` | `VARCHAR(128)` | `NOT NULL` | Consequence VEP (luôn chứa `missense_variant`) |
| `protein_change` | `VARCHAR(64)` | | HGVSp (vd `p.Val600Glu`), fallback `p.{ref_aa}{pos}{alt_aa}` |
| `clinvar_ground_truth` | `VARCHAR(16)` | `NOT NULL` | `Pathogenic` / `Benign` / `Uncertain` (`normalize_clinvar_label`) |
| `created_at` | `TIMESTAMPTZ` | | |

### 2.3 Bảng `predictions`

| Cột | Kiểu PostgreSQL | Ràng buộc | Ý nghĩa |
|---|---|---|---|
| `id` | `BIGSERIAL` | **PRIMARY KEY** | |
| `variant_id` | `VARCHAR(64)` | `NOT NULL`, **FK ON DELETE CASCADE** | |
| `ablation_mask` | `SMALLINT` | `NOT NULL` | Bitmask (bit3 DNA, bit2 Protein, bit1 Bio, bit0 Geometry); từ UI chỉ có 15, 12, 11, 7, 8, 4, 3 |
| `trimvpp_prob` | `FLOAT` | `NOT NULL` | Xác suất Pathogenic (làm tròn 4 chữ số) |
| `classification` | `VARCHAR(16)` | `NOT NULL` | `Pathogenic` nếu `p ≥ τ` của checkpoint, ngược lại `Benign` |
| `sota_alphamissense` | `FLOAT` | nullable | max(AlphaMissense_score) từ dbNSFP |
| `sota_revel` | `FLOAT` | nullable | max(REVEL_score) từ dbNSFP |
| `ig_values` | `JSONB` | `NOT NULL` | Kết quả Integrated Gradients (Mục 4.12): `{method, mode, baseline, steps, logit, baseline_logit, completeness_residual, relative_residual, residual_over_5pct, modalities[], tokens[], features[]}` |
| `model_tag` | `VARCHAR(128)` | nullable | Khoá phiên bản cache: `"{file}.pth@{mtime}\|{EXTRACTOR_TAG}"` |

**Index:** `ix_variant_mask` — **UNIQUE** trên `(variant_id, ablation_mask)`. Attention map **không** được lưu (tính khi mở trang XAI).

### 2.4 Bảng `inference_features`

| Cột | Kiểu PostgreSQL | Ràng buộc | Ý nghĩa |
|---|---|---|---|
| `variant_id` | `VARCHAR(64)` | **PK** + **FK ON DELETE CASCADE** | |
| `extractor_tag` | `VARCHAR(128)` | `NOT NULL` | `"fx-" + sha1(DNA_MODEL_ID\|PROT_MODEL_ID\|DNA_WINDOW\|PROT_WINDOW\|pad\|device\|dtype\|token-window-k16\|geom-step2-matched…)[:12]` |
| `emb_dna_ref`, `emb_dna_alt` | `vector(1280)` | `NOT NULL` | Hidden state lớp cuối NT-v1 tại token đột biến (token tâm), REF / ALT |
| `emb_pro_ref`, `emb_pro_alt` | `vector(1280)` | `NOT NULL` | Như trên cho ESM-1b |
| `tok_dna`, `tok_pro` | `BYTEA` | `NOT NULL` | Cửa sổ 33 token × 2 allele × 1280, **float16 little-endian** (~0,17 MB mỗi nhánh) |
| `tok_mask` | `BYTEA` | `NOT NULL` | uint8 `(2 nhánh, 2 allele, 33)`: vị trí token hợp lệ |
| `raw_features` | `JSONB` | `NOT NULL` | Giá trị **thô**: 11 bio core + 9 SpliceAI + `{dna,prot}_{LLR, LVD_L2, LVD_Cosine, LID, LVD_Relative}` (thiếu = `null`) |
| `annotation` | `JSONB` | `NOT NULL` | `{transcript, ensp, protein_position, hgvsp, clinvar_raw, alphamissense, revel, warnings[], lid_k}` |
| `created_at` | `TIMESTAMPTZ` | | |

Không có index ANN trên các cột vector: pgvector ở đây là **kho lưu trữ**, còn tìm láng giềng dùng FAISS trong bộ nhớ.

### 2.5 DDL tương đương (sinh bởi `Base.metadata.create_all`)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE variants (
    variant_id           VARCHAR(64)  PRIMARY KEY,
    gene                 VARCHAR(32),
    consequence          VARCHAR(128) NOT NULL,
    protein_change       VARCHAR(64),
    clinvar_ground_truth VARCHAR(16)  NOT NULL,
    created_at           TIMESTAMPTZ
);

CREATE TABLE predictions (
    id                 BIGSERIAL    PRIMARY KEY,
    variant_id         VARCHAR(64)  NOT NULL REFERENCES variants (variant_id) ON DELETE CASCADE,
    ablation_mask      SMALLINT     NOT NULL,
    trimvpp_prob       FLOAT        NOT NULL,
    classification     VARCHAR(16)  NOT NULL,
    sota_alphamissense FLOAT,
    sota_revel         FLOAT,
    ig_values          JSONB        NOT NULL,
    model_tag          VARCHAR(128)
);
CREATE UNIQUE INDEX ix_variant_mask ON predictions (variant_id, ablation_mask);

CREATE TABLE inference_features (
    variant_id    VARCHAR(64)  PRIMARY KEY REFERENCES variants (variant_id) ON DELETE CASCADE,
    extractor_tag VARCHAR(128) NOT NULL,
    emb_dna_ref   vector(1280) NOT NULL,
    emb_dna_alt   vector(1280) NOT NULL,
    emb_pro_ref   vector(1280) NOT NULL,
    emb_pro_alt   vector(1280) NOT NULL,
    tok_dna       BYTEA        NOT NULL,
    tok_pro       BYTEA        NOT NULL,
    tok_mask      BYTEA        NOT NULL,
    raw_features  JSONB        NOT NULL,
    annotation    JSONB        NOT NULL,
    created_at    TIMESTAMPTZ
);
```

**Migration trong `init_db.py`:** (1) nếu `inference_features` thiếu cột `tok_dna` (schema trước TriMVPP) → xoá `predictions` + `inference_features`; (2) nếu `predictions` còn cột `shap_values` → xoá `predictions` để tạo lại với `ig_values` (đặc trưng được giữ); (3) `ADD COLUMN IF NOT EXISTS model_tag`; (4) `DROP TABLE IF EXISTS extracted_features`.

### 2.6 Cơ chế Upsert chống trùng lặp

`_upsert()` (`inference_service.py`) sinh `INSERT … ON CONFLICT (<conflict_columns>) DO UPDATE SET <cột còn lại> = EXCLUDED.<cột>`.

| Bảng | `conflict_columns` | Thời điểm ghi |
|---|---|---|
| `variants` | `variant_id` | Sau khi trích xuất đặc trưng (`_store_features`) |
| `inference_features` | `variant_id` | Cùng giao dịch với `variants`; **commit ngay** để nếu bước sau lỗi thì không phải chạy lại VEP + FM |
| `predictions` | `(variant_id, ablation_mask)` | Sau khi tính xác suất + IG |

Chạy lại (Live Bypass), dữ liệu khác `model_tag`, hay hai request đồng thời đều **ghi đè đúng một dòng**.

---

## 3. Backend API Contracts & Communication Protocols

### 3.1 Tổng quan

- Base URL: `http://localhost:8000/api/v1`. OpenAPI: `GET /docs`. Phiên bản app: `2.0.0`.
- Middleware `request_context`: gán `X-Request-ID`, ghi log `METHOD path -> status (ms)`, biến exception chưa xử lý thành JSON 500. `CORSMiddleware` là lớp ngoài cùng.

| # | Method | Path | Response | Mô tả |
|---|---|---|---|---|
| 1 | GET | `/` | JSON | Thông tin dịch vụ |
| 2 | GET | `/api/v1/health` | JSON | DB, thiết bị, 7 checkpoint (kèm τ và MCC test), artifact |
| 3 | POST | `/api/v1/predict/single` | `SingleVariantResponse` | Dự đoán + IG cho một biến thể |
| 4 | GET | `/api/v1/predict/batch-stream` | `text/event-stream` | Sàng lọc cohort demo 10 biến thể |
| 5 | POST | `/api/v1/predict/batch-upload` | `text/event-stream` | Sàng lọc file upload |
| 6 | GET | `/api/v1/xai/{variant_id}?ablation_mask=` | `XaiResponse` | Dự đoán + IG (từ cache) + attention map |

### 3.2 `GET /api/v1/health`

```json
{
  "status": "healthy | degraded",
  "database": "connected | error",
  "model_mode": "pure_dl_inference",
  "device": "cuda | cpu",
  "checkpoints": [
    {"combination": "111", "active_modalities": ["dna","prot","bio","geom"],
     "checkpoint": "<abs path>/trimvpp_111.pth", "status": "available | checkpoint_missing",
     "decision_threshold": 0.4791317582130432, "test_mcc": 0.8424}
  ],
  "artifacts": { "bio_scalers_missing": [], "geom_scalers_missing": [], "faiss_indexes": {...},
                 "foundation_models_loaded": ["dna","prot"], "fm_device": "cuda", "fm_dtype": "fp16",
                 "extractor_tag": "fx-xxxxxxxxxxxx", "torch_cuda_available": true }
}
```

Frontend dùng `checkpoints[].test_mcc` để hiển thị dòng *Test-set MCC of this configuration* dưới bảng công tắc.

### 3.3 `POST /api/v1/predict/single`

**Request — `SingleVariantRequest`:** `variant_id` (str), `dna_enabled`, `protein_enabled`, `bio_enabled`, `geo_enabled` (bool, mặc định `true`); `use_dna`, `use_prot`, `use_biogeom` (bool | null, nếu gửi thì ghi đè; `use_biogeom` gán đồng thời bio + geo); `use_cache` (bool, `false` = Live Bypass). `variant_id` chấp nhận `chr7:140753336:A>T`, `7-140753336-A-T`, `chr7_140753336_A/T`…

**Response 200 — `SingleVariantResponse`:**

```json
{
  "variant_id": "chr17:43063930:C>T",
  "gene": "BRCA1",
  "consequence": "missense_variant (p.Arg1699Gln)",
  "clinvar_ground_truth": "Pathogenic | Benign | Uncertain",
  "results": { "trimvpp": { "probability": 0.7769, "label": "Pathogenic" } },
  "sota":    { "alphamissense": 0.0, "revel": 0.0 },
  "ig_analysis": {
    "method": "integrated_gradients", "mode": "full_prediction", "baseline": "train3_background_256_seed2026",
    "steps": 64, "logit": 1.249, "baseline_logit": -0.201,
    "completeness_residual": 0.0007, "relative_residual": 0.0005, "residual_over_5pct": false,
    "modalities": [ {"modality": "DNA | Protein | Bio_SpliceAI | Geometry", "signed": 0.0, "absolute": 0.0} ],
    "tokens":     [ {"modality": "DNA | Protein", "allele": "Ref | Alt", "position": -16, "signed": 0.0, "absolute": 0.0} ],
    "features":   [ {"feature": "gnomADe_AF", "name": "gnomAD Exomes AF", "group": "Bio_SpliceAI | Geometry",
                     "description": "...", "raw_value": 1.2e-5, "raw_note": null, "input_value": -1.3, "signed": 0.0} ]
  },
  "meta": {
    "is_cache_hit": false, "computation_time_ms": 0, "ablation_mask": 15,
    "active_modalities": ["dna","prot","bio","geom"],
    "checkpoint": "trimvpp_111.pth", "architecture": "TriMVPP",
    "decision_threshold": 0.4791317582130432, "test_mcc": 0.8424,
    "warnings": ["spliceai_not_found: ...", "alphamissense_unavailable: ..."],
    "timings_ms": { "vep": 0, "spliceai": 0, "fm_dna": 0, "fm_prot": 0, "geom": 0, "norm": 0, "ig": 0, "db": 0 }
  }
}
```

- `tokens`: 66 dòng cho mỗi nhánh chuỗi đang bật (2 allele × 33 vị trí). `features`: 28 dòng khi bật Bio + Geom, sắp theo |IG| giảm dần.
- `sota.*` = `null` khi dbNSFP không có điểm (frontend hiển thị "N/A").

### 3.4 `GET /api/v1/xai/{variant_id}?ablation_mask={0..15}`

- Nội bộ (`run_xai`): `run_single_variant_inference(use_cache=True)` (dự đoán + IG, thường trúng cache) → đọc `inference_features` → `ml_engine.attention()`.
- Response `XaiResponse` = `SingleVariantResponse` + trường `attention`:

```json
"attention": {
  "num_layers": 3, "num_heads": 8,
  "layers": [ { "layer": 1, "maps": [ {
      "query": "dna | protein | bio_geometry",
      "query_labels": ["Protein:-16", "...", "Protein:+16"],
      "context_segments": [ {"name": "dna", "labels": ["DNA:-16", "..."]}, {"name": "bio_geometry", "labels": ["AF", "..."]} ],
      "key_labels": ["DNA:-16", "...", "prot_LVD_Relative"],
      "weights": [[[0.0]]]
  } ] } ]
}
```

- `weights` có shape `(heads, query, key)`, làm tròn 5 chữ số, mỗi hàng cộng lại bằng 1. Tổ hợp **một nhánh** (100, 010, 001) → `attention = null`.
- Kích thước: ~1,0 MB (111), ~0,35 MB (tổ hợp 2 nhánh), ~0,01 MB (1 nhánh). `meta.timings_ms.attn` ghi thời gian tính attention.

### 3.5 Batch Screening qua Server-Sent Events

#### 3.5.1 `GET /api/v1/predict/batch-stream`

Query: `stream_delay_ms` (≥ 0, mặc định 0), `dna_enabled`, `protein_enabled`, `bio_enabled`, `geo_enabled`. Hàng đợi là **cohort demo 10 biến thể GRCh38** (`_get_demo_cohort_variants`):

| # | variant_id | Gene / protein |
|---|---|---|
| 1 | `chr7:140753336:A>T` | BRAF p.Val600Glu |
| 2 | `chr17:7674220:C>T` | TP53 p.Arg248Gln |
| 3 | `chr17:7673802:C>T` | TP53 p.Arg273His |
| 4 | `chr17:7675088:C>T` | TP53 p.Arg175His |
| 5 | `chr12:25245350:C>T` | KRAS p.Gly12Asp |
| 6 | `chr3:179234297:A>G` | PIK3CA p.His1047Arg |
| 7 | `chr3:179218303:G>A` | PIK3CA p.Glu545Lys |
| 8 | `chr7:55191822:T>G` | EGFR p.Leu858Arg |
| 9 | `chr2:208248388:C>T` | IDH1 p.Arg132His |
| 10 | `chr1:114713908:T>C` | NRAS p.Gln61Arg |

#### 3.5.2 `POST /api/v1/predict/batch-upload` (multipart/form-data)

Trường form: `file` (bắt buộc), `stream_delay_ms`, 4 cờ modality. Định dạng (`upload_parser.py`):

| Đuôi | Cách đọc |
|---|---|
| `.vcf`, `.txt` | Bỏ dòng `#`; dòng ≥ 5 cột với cột 2 là số → `CHROM, POS, REF, ALT`; ngược lại coi cả dòng là mã `chrN:POS:REF>ALT` |
| `.csv` | Header có đủ 4 cột (alias không phân biệt hoa thường) → đọc theo cột; nếu không, tìm ô khớp regex variant ID |
| `.parquet` | `pd.read_parquet` |
| `.xlsx` | `pd.read_excel(sheet_name=0, engine="openpyxl")` |

Alias cột: `chrom ∈ {chrom, #chrom, chr, chromosome, contig, seqname}`, `pos ∈ {pos, position, start, pos_hg38, pos_grch38}`, `ref ∈ {ref, ref_allele, reference}`, `alt ∈ {alt, alt_allele, alternate}`. ALT đa alen chỉ lấy alen đầu. Kết quả được chuẩn hoá + khử trùng lặp, giữ thứ tự. File rỗng/không hợp lệ → HTTP 400 `INVALID_UPLOAD` (trước khi mở stream).

#### 3.5.3 Giao thức SSE

- Mỗi sự kiện: một dòng `data: <JSON BatchStreamEvent>`. Lỗi tổ hợp/checkpoint được kiểm tra **trước** khi mở stream (trả HTTP 400/409 thường).
- `BatchStreamEvent`: `type` (`variant_stream | variant_skipped | stream_completed | stream_error`), `progress_percent`, `variant` (`BatchVariant`), `variant_id`, `detected_consequence`, `reason`, `error_code`, `summary {total_processed, missense_accepted, non_missense_rejected}`.
- `BatchVariant`: `id`, `variant_id`, `gene`, `consequence`, `trimvpp_probability`, `classification`, `clinvar_ground_truth`, `sota_alphamissense`, `danger_rank` (= thứ tự đến).

| Tình huống | Sự kiện | Stream |
|---|---|---|
| Không phải missense | `variant_skipped`, `NON_MISSENSE_VARIANT` | tiếp tục |
| Lỗi riêng biến thể (REF sai build, không phải SNV, thiếu ENSP…) | `variant_skipped`, `INVALID_VARIANT` | tiếp tục |
| Lỗi hạ tầng (VEP/scaler/FM…) | `stream_error` | dừng |
| Mất kết nối PostgreSQL | `stream_error`, `DATABASE_UNAVAILABLE` | dừng |
| Chạy xong | `stream_completed` + `summary` | kết thúc |

Client: `batch-stream` dùng `EventSource`; `batch-upload` dùng `fetch` + `ReadableStream` (`parseSseLines` trong `services/api.ts`).

### 3.6 Định dạng lỗi & bảng `error_code`

Mọi lỗi (ngoài SSE): `{ "detail": { "error_code": "…", "message": "…", "request_id": "…", ...extra } }`.

| `error_code` | HTTP | Nguồn |
|---|---|---|
| `INVALID_VARIANT` | 400 | Sai định dạng, không phải SNV, ngoài hg38, **REF ≠ base hg38**, residue protein không khớp |
| `INVALID_UPLOAD` | 400 | Đuôi không hỗ trợ, file hỏng, không có biến thể |
| `NO_MODALITY_SELECTED` | 400 | Tắt cả các nhóm (mask 0) |
| `UNSUPPORTED_MODALITY_COMBINATION` | 400 | Tổ hợp không có trong registry (vd Bio bật nhưng Geometry tắt qua API 4 cờ) |
| `CHECKPOINT_MISSING` | 409 | Tổ hợp hợp lệ nhưng thiếu file `.pth` |
| `NON_MISSENSE_VARIANT` | 422 | Missense Gatekeeper |
| `VALIDATION_ERROR` | 422 | Pydantic/Query |
| `CHECKPOINT_INCOMPATIBLE` | 500 | state_dict không khớp kiến trúc (nạp strict) |
| `PIPELINE_ERROR` / `INTERNAL_ERROR` | 500 | Lớp cơ sở / exception không lường trước |
| `FEATURE_ARTIFACT_MISSING` | 503 | Thiếu FASTA, SpliceAI, scaler `.pkl` hoặc **IG baseline** (`status = ig_baseline_missing`) |
| `FOUNDATION_MODEL_UNAVAILABLE` | 503 | Không nạp được NT-v1/ESM-1b |
| `VEP_UNAVAILABLE` | 503 | Docker CLI/container lỗi, timeout 900 s |
| `DATABASE_UNAVAILABLE` | 503 | `OperationalError` |

---

## 4. End-to-End Inference & Hybrid Lazy-Caching Pipeline

### 4.1 Khởi động: `lifespan` preloading (`app/main.py`)

Khi `PRELOAD_ON_STARTUP=True`, `_preload()` chạy trong `asyncio.to_thread` **trước khi nhận request**:

1. `feature_pipeline.preload()`: mở FASTA hg38 + proteome (`pyfaidx`), nạp 2 FAISS index, parse `.tbi` của SpliceAI, nạp 6 scaler Bio + 2 scaler Geometry, tải NT-v1 500M và ESM-1b 650M lên GPU ở **FP16** (không autocast, giống pipeline huấn luyện).
2. `checkpoint_registry.get(("dna","prot","bio","geom"))`: nạp sẵn `trimvpp_111.pth`. Các checkpoint khác và baseline IG được nạp ở request đầu tiên cần đến (cache theo `(path, mtime)`).

Lỗi preload chỉ được ghi log; request sau sẽ trả lỗi nghiệp vụ rõ ràng. Khởi động mất ~10 s và ~2,3 GiB VRAM. Không dùng `--reload`.

### 4.2 Sơ đồ luồng tổng quát (Single Variant)

```
POST /predict/single
  │ parse_variant_id → variant_id chuẩn; resolve_request_mask → bitmask → resolve_model (400/409 sớm)
  ▼
[DB] inference_features có extractor_tag hiện tại?
  ├─ có → _refresh_stale_geometry (tính lại LID nếu cần) ─┐
  │                                                     ▼
  │       use_cache && predictions(variant_id, mask, model_tag hiện tại) tồn tại?
  │         ├─ có  ─────────────────────────────► CACHE HIT → trả response (is_cache_hit=true)
  │         └─ không → nạp cửa sổ token + raw_features ─► FEATURE REUSE (đổi tổ hợp / Live Bypass)
  └─ không → CACHE MISS: _extract_and_store
              ├─ Thread VEP (docker exec)  ║ song song ║  NT-v1 (cửa sổ DNA) + SpliceAI lookup
              ├─ Missense Gatekeeper (422 nếu không phải missense)
              ├─ Cửa sổ Protein 501 aa → ESM-1b
              ├─ Geometry V2 từ token tâm (LVD + LID qua FAISS k=64)
              └─ UPSERT variants + inference_features → COMMIT
  ▼
ml_engine.predict: chuẩn hoá .pkl → TriMVPP (FP32) → Integrated Gradients (logit) → sigmoid → so với τ
  ▼
UPSERT predictions (variant_id, ablation_mask) → COMMIT → response
```

### 4.3 Bước 1 — Chuẩn hoá đầu vào & chọn checkpoint

- Bitmask: `mask = dna<<3 | prot<<2 | bio<<1 | geo`. Giao diện có 3 công tắc; nhóm **Bio + Geom + SpliceAI** gán đồng thời bit 1 và bit 0.
- `CheckpointRegistry` (`services/checkpoint_registry.py`) ánh xạ khoá 3 bit → file, dựng `TriMVPP(active_branches)` đúng các nhánh, nạp `load_state_dict(strict=True)`; tên/shape tham số lệch → 500 `CHECKPOINT_INCOMPATIBLE`.

| Khoá | Nhánh | mask | File | Tham số | τ (MCC tối ưu trên validation) | MCC test |
|---|---|---|---|---|---|---|
| `111` | DNA, Protein, Bio+Geom+SpliceAI | 15 | `trimvpp_111.pth` | 33 273 860 | 0.4791 | **0.8424** |
| `110` | DNA, Protein | 12 | `trimvpp_110.pth` | 23 257 091 | 0.6340 | 0.7259 |
| `101` | DNA, Bio+Geom+SpliceAI | 11 | `trimvpp_101.pth` | 21 908 483 | 0.7515 | 0.7676 |
| `011` | Protein, Bio+Geom+SpliceAI | 7 | `trimvpp_011.pth` | 21 908 483 | 0.5660 | 0.8390 |
| `100` | DNA | 8 | `trimvpp_100.pth` | 2 431 490 | 0.4819 | 0.2569 |
| `010` | Protein | 4 | `trimvpp_010.pth` | 2 431 490 | 0.7383 | 0.5873 |
| `001` | Bio+Geom+SpliceAI | 3 | `trimvpp_001.pth` | 1 082 882 | 0.6309 | 0.7615 |

Ngưỡng và MCC nằm trong `DEFAULT_DECISION_THRESHOLDS` / `DEFAULT_TEST_MCC` (`core/config.py`); có thể ghi đè bằng `.env`.

### 4.4 Bước 2 — Ensembl VEP 116 (`services/vep_runner.py`)

- Ghi VCF tạm vào `VEP_TMP_DIR` (mount `/tmp_io`), chạy:

```bash
docker exec rbnet_vep vep \
  -i /tmp_io/<job>.vcf -o /tmp_io/<job>.tsv \
  --format vcf --tab --force_overwrite --no_stats \
  --offline --cache --cache_version 116 --assembly GRCh38 \
  --dir_cache /vep_cache --fasta /vep_resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
  --dir_plugins /vep_cache/Plugins \
  --mane_select --canonical --pick --symbol --protein --hgvs \
  --af --af_gnomade --check_existing \
  --plugin dbNSFP,/vep_resources/dbNSFP5.3.1a_grch38.gz,GERP++_NR,GERP++_RS,GERP_92_mammals,phastCons100way_vertebrate,phastCons17way_primate,phastCons470way_mammalian,phyloP100way_vertebrate,phyloP17way_primate,phyloP470way_mammalian,AlphaMissense_score,REVEL_score,clinvar_clnsig \
  [--fork 4]   # khi số biến thể > VEP_FORK_THRESHOLD (50)
```

- `--pick` trả đúng một dòng/biến thể, ưu tiên MANE Select hoặc canonical. Lấy consequence, SYMBOL, transcript, ENSP, vị trí protein, amino acid, HGVSp, ClinVar thô; AlphaMissense/REVEL lấy **max**; 11 cột bio core lấy giá trị số đầu tiên (thiếu = `NaN`). Timeout 900 s.

### 4.5 Bước 3 — Missense Gatekeeper & kiểm tra REF build

- `ensure_missense`: consequence không chứa `missense_variant` → 422 / SSE `variant_skipped`.
- `dna_windows`: chỉ nhận SNV; **base hg38 tại tâm phải bằng REF** (nếu không → `INVALID_VARIANT` kèm `genome_base`, gợi ý kiểm tra genome build).
- `protein_windows`: ENSP phải có trong proteome; residue tham chiếu phải bằng `ref_aa` của VEP.

### 4.6 Bước 4 — SpliceAI lookup (`ml_core/spliceai_lookup.py`)

`NativeTabixReader` tự parse `.tbi`, đọc các block BGZF quanh vị trí (cửa sổ 1024 bp), không giải nén toàn bộ 28 GB. Khớp `(chrom, pos, ref, alt)`; nhiều annotation → giữ annotation có delta lớn nhất. 9 cột ra theo thứ tự `DS_AG, DS_AL, DS_DG, DS_DL, DP_AG, DP_AL, DP_DG, DP_DL, DS_max`. Không tìm thấy → `NaN` + cảnh báo `spliceai_not_found`.

### 4.7 Bước 5 — Cắt chuỗi GRCh38

| Nhánh | Cửa sổ | Tâm | Đệm | Nguồn |
|---|---|---|---|---|
| DNA | 1001 bp `[pos-500, pos+500]` | index 500 | `N` | FASTA hg38 |
| Protein | 501 aa `[p-250, p+250]` | index 250 | `X` | proteome Ensembl, ENSP do VEP chọn |

Chuỗi ALT = chuỗi REF thay ký tự ở tâm.

### 4.8 Bước 6 — Cửa sổ token NT-v1 500M + ESM-1b 650M (`ml_core/fm_extractor.py`)

- Mỗi chuỗi REF/ALT được tokenize riêng; vị trí token đột biến được ánh xạ **riêng cho từng allele** (offset mapping hoặc so khác biệt token). Khi REF và ALT có cùng độ dài token, cả hai được ghép vào một forward.
- Từ hidden state lớp cuối, lấy **33 token** quanh token đột biến (±16) cho REF và ALT → `TokenWindow.hidden (2, 33, 1280)` float16 và `mask (2, 33)` (vị trí ngoài chuỗi = `False`). Đây là đầu vào chuỗi của TriMVPP.
- Token tâm (`hidden[:, 16]`) là embedding REF/ALT dùng cho geometry và lưu vào cột `vector(1280)`.
- FM nạp ở FP16 (`FM_CUDA_DTYPE=fp16`), không autocast. Batch: tối đa `FM_BATCH_VARIANTS = 8` biến thể/forward; khi CUDA OOM thì chia đôi lô đệ quy.
- **Chồng lấp ở luồng Single:** VEP chạy trong `ThreadPoolExecutor(max_workers=4)` song song với NT-v1 + SpliceAI; ESM-1b chờ VEP trả ENSP + vị trí protein.

### 4.9 Bước 7 — Geometry V2: LVD + LID (`ml_core/geometry_v2.py`)

Với embedding token tâm (ép về float16) và δ = E_alt − E_ref:

| Đặc trưng | Công thức |
|---|---|
| `LVD_L2` | ‖δ‖₂ |
| `LVD_Cosine` | 1 − cos(E_ref, E_alt) |
| `LVD_Relative` | ‖δ‖₂ / (‖E_ref‖ + 10⁻⁸) |
| `LID` | MLE Levina–Bickel trên k = **64** láng giềng của δ trong FAISS (23 208 displacement của tập train) |
| `LLR` | **= 0** (bộ geometry `step2_matched` không dùng LLR; chỉ lưu để tương thích) |

Model dùng 8 giá trị `{dna,prot}_{LVD_L2, LVD_Cosine, LID, LVD_Relative}`. Thiếu FAISS → `LID = NaN` + cảnh báo (điền trung vị train).

### 4.10 Bước 8 — Chuẩn hoá (`ml_core/normalizer.py`)

| Nhóm cột | Biến đổi |
|---|---|
| `AF`, `gnomADe_AF` | fillna(0) → `log10(x + 1e-8)` → `bio_freq_scaler` |
| 3 phastCons + 3 phyloP + 3 GERP | `bio_mean_imputer` → `bio_phast_scaler` / `bio_phylo_scaler` |
| 9 cột SpliceAI | fillna(`bio_spliceai_fill_values`) → `bio_spliceai_scaler` |
| Geometry (mỗi nhánh) | fillna → `log1p(clip(x, 0))` → scaler riêng từng cột (`*_geom_v2_scalers.pkl`) |

Vector cuối: `bio (20) + geom (8)`; giá trị còn `NaN` sau chuẩn hoá → 0. Thiếu scaler → 503 `FEATURE_ARTIFACT_MISSING`.

### 4.11 Bước 9 — Suy luận TriMVPP (`ml_core/trimvpp_model.py`)

Siêu tham số chung (`TriMVPPConfig`): `model_dim = 512`, 8 head, 3 lớp cross-attention, FFN ×4, `k_radius = 16`, pooling trung bình.

```
Nhánh DNA / Protein:   [ref_token ⊕ alt_token] (2560) → Linear → 512  + relative position + variant embedding (vị trí 0)
                       + modality embedding → 33 token; mask = ref_mask ∨ alt_mask
Nhánh Bio+Geom:        mỗi đặc trưng vô hướng x_j → token  x_j·W_j + e_j  (20 bio + 8 geom = 28 token) + modality embedding
Cross-attention (×3):  mỗi nhánh làm query, key/value = token ghép của các nhánh còn lại (theo thứ tự DNA, Protein, Bio);
                       pre-LayerNorm → MultiheadAttention (8 head) → residual → FFN (512→2048→512) → residual;
                       mọi nhánh đọc cùng trạng thái đầu vào của lớp
Pooling:               trung bình theo mask của từng nhánh → ghép → LayerNorm
Classifier:            Linear(512·n → 1024) → GELU → Linear(1024 → 512) → GELU → Linear(512 → 1) = logit
p = sigmoid(logit);    "Pathogenic" nếu p ≥ τ, ngược lại "Benign"
```

- Nhánh bị tắt **không tồn tại** trong mạng (không phải điền 0). Khi chỉ có một nhánh, mạng **không có lớp cross-attention** (pooling trực tiếp).
- `forward(..., return_attention=True)` trả thêm `{"layer_{l}_{branch}": (B, heads, Lq, Lk)}` (`average_attn_weights=False`).
- Mô hình chạy FP32 trên GPU; đầu vào token được chuyển từ float16 lên float32.

### 4.12 Bước 10 — Integrated Gradients (`ml_core/xai.py`, `services/ml_engine.py`)

Giữ nguyên phương pháp notebook 19 của nhóm (`cross_attention_explainability.integrated_gradients`):

- **Đầu ra được giải thích:** logit Pathogenic (trước sigmoid), chế độ `full_prediction`.
- **Đầu vào được giải thích:** `token_dna`, `token_protein` (2 × 33 × 1280), `bio` (20), `geom` (8) của các nhánh đang bật; mask giữ cố định.
- **Baseline:** token trung bình (theo mask) + trung vị đặc trưng của 256 biến thể train3, seed 2026 (`ig_baseline_train3_seed2026.pt`). Baseline **cố định cho mọi request**, không phụ thuộc biến thể hay số lượng biến thể người dùng nhập.
- **Tích phân:** Riemann-trapezoid 64 bước, tính theo nhóm 8 điểm nội suy (`IG_INTERNAL_BATCH`), FP32. Nếu `|residual| / |logit − logit_baseline| > 5%` → chạy lại 128 bước; vẫn vượt thì `residual_over_5pct = true` (UI cảnh báo).
- **Completeness:** Σ IG + residual = logit − logit_baseline (kiểm chứng: lệch < 10⁻⁶).
- **Gộp kết quả:** theo nhánh (`modalities`: tổng có dấu + tổng |IG|), theo token (`tokens`: tổng trên 1280 chiều cho mỗi allele × vị trí), theo đặc trưng (`features`: kèm tên hiển thị, mô tả, giá trị thô, giá trị đã chuẩn hoá).
- Xác suất trả về = sigmoid(logit) của chính lần forward FP32 trong IG.
- IG được tính **trong mỗi lần dự đoán** (Single và Batch) và lưu vào `predictions.ig_values`.

**Attention map** (`ml_engine.attention`, `xai.attention_maps`): tính khi gọi `/xai`, từ đặc trưng đã lưu; trả 3 lớp × 8 head cho mọi nhánh query; nhãn token theo dạng `DNA:-16 … DNA:+16`, `Protein:+0`, tên đặc trưng. Không lưu DB.

**Đối chiếu với code nhóm (bước A, Phase 2):** với cùng đầu vào, logit lệch ~10⁻⁷, attention lệch ~6·10⁻⁷. Trên 200 biến thể test (tổ hợp 111, so với notebook seed 37 = checkpoint đang dùng): logit lệch trung vị 0,001 (max 0,038, do embedding trích lại ở FP16); Spearman ≥ 0,9998 (nhánh), 0,9999 (đặc trưng), 1,000 (token); đặc trưng quan trọng nhất trùng 200/200; attention (48 biến thể) lệch tối đa 0,0014.

### 4.13 Bước 11 — Hybrid Lazy-Caching

Trạng thái ghi ở trường `cache=` của log `inference variant=… mask=… cache=… total=…ms <timings>`:

| Trạng thái | Điều kiện | Công việc | Ghi DB |
|---|---|---|---|
| **`hit`** | `use_cache` ∧ có đặc trưng hợp lệ ∧ có prediction khớp `(variant_id, mask, model_tag)` | Chỉ đọc DB | không |
| **`features`** | Có đặc trưng nhưng chưa có prediction cho mask/model hiện tại, **hoặc** `use_cache=false` | Đọc cửa sổ token + `raw_features` → chuẩn hoá → TriMVPP + IG. Không chạy lại VEP/SpliceAI/FM | upsert `predictions` |
| **`miss`** | Chưa có đặc trưng (hoặc `extractor_tag` cũ) | Toàn bộ pipeline | upsert `variants` + `inference_features` (commit), rồi `predictions` |

**Vô hiệu hoá cache tự động:** `EXTRACTOR_TAG` đổi khi đổi FM, cửa sổ, ký tự đệm, thiết bị, dtype hoặc phiên bản cửa sổ token/geometry; `model_tag` đổi khi file `.pth` đổi mtime; `_refresh_stale_geometry` tính lại LID từ embedding đã lưu nếu `LID = null` hoặc `lid_k` khác cấu hình, rồi xoá predictions cũ của biến thể.

### 4.14 Luồng Batch (SSE)

1. `prefetch_annotations`: lọc biến thể chưa có đặc trưng → **VEP một lần cho cả file** (`--fork 4` nếu > 50) → SpliceAI cho các SNV missense.
2. Theo khối 8 biến thể: `prepare_features_batch` dựng cửa sổ, chạy NT-v1 rồi ESM-1b (một forward/khối), tính geometry, upsert + commit một lần. Lỗi riêng biến thể trả về dạng `failures`.
3. Từng biến thể: `run_single_variant_inference(use_cache=True, prefetched)` (dự đoán + IG) → phát sự kiện SSE ngay khi xong.
4. Mọi lời gọi nặng chạy trong `asyncio.to_thread`.

---

## 5. Frontend Architecture & User Workflows

### 5.1 Routing (`src/App.tsx`)

| Path | Component | Chức năng |
|---|---|---|
| `/` | `Home` | Landing: 3 thẻ chức năng |
| `/single` | `Single` | Dự đoán một biến thể, so sánh ClinVar + SOTA |
| `/batch` | `Batch` | Sàng lọc theo lô qua SSE |
| `/xai` | `Xai` | Integrated Gradients + attention map |
| `*` | `<Navigate to="/" replace />` | Fallback |

`MainLayout`: logo + tiêu đề "Missense Variant Pathogenicity Classification System", badge **TriMVPP**, nav (Single Variant / Batch Screening / XAI Attribution), nút Back to Home (trừ `/`), footer "© 2026 TriMVPP Variant Classifier • … • AIP491 Project".

### 5.2 Component chính

| Component | Vai trò |
|---|---|
| `ablation/FeatureAblationPanel` | 3 công tắc: DNA Sequence (NT-v1 500M, 1001 bp, 33 ref/alt token), Protein Sequence (ESM-1b 650M, 501 aa), Bio + Geom + SpliceAI (28 đặc trưng); badge `n/3 Modules Active`; dòng nhỏ *Test-set MCC of this configuration* (lấy từ `/health`) |
| `xai/IgModalityBars` | Thanh lệch hai phía theo nhánh (hồng = Pathogenic, xanh lục = Benign) + % trên tổng \|IG\| |
| `xai/IgTokenStrip` | Dải SVG Ref/Alt × 33 vị trí, thang màu phân kỳ, cột +0 được viền, tooltip khi rê chuột |
| `xai/IgFeatureChart` | Recharts: 28 đặc trưng theo \|IG\|; tooltip có giá trị thô, giá trị chuẩn hoá, mô tả |
| `xai/AttentionPanel` | Chọn Query (DNA/Protein/Bio+Geom+SpliceAI), Layer (1–3), Head (Mean / H1–H8); heatmap SVG (thang tím tuần tự); bấm một hàng → bảng 8 key được chú ý nhiều nhất |
| `lib/xai.ts` | Thang màu, nhãn rút gọn, `headMatrix` (trung bình head), `defaultFocus` |
| `services/api.ts` | `predictSingleVariant`, `fetchVariantXai` (→ `XaiResponse`), `createVcfStreamService`, `fetchTestMcc` |
| `services/api.mock.ts` | Giả lập backend (`VITE_USE_MOCK_API="true"`), sinh IG/attention giả có cùng cấu trúc |

### 5.3 Quản lý state

Không dùng store toàn cục. Ba tầng: state cục bộ (`useState`/`useRef`); URL query (`variant`, `mask`, `from`, `origin`); `sessionStorage`:

| Khoá | Trang | Nội dung |
|---|---|---|
| `trimvpp_batch_state` | Batch | `{progress, stats, streamedVariants, dangerQueue, ablation, runMask}` |
| `trimvpp_single_state_v1` | Single | `{variantInput, activePresetId, useCache, ablation, submittedVariant, result, meta}` |

Mọi đọc/ghi `sessionStorage` được bọc `try/catch`. Chống race condition ở Single: `requestSeq` bỏ qua phản hồi cũ.

### 5.4 Trang Single Variant (`/single`)

- **Nhập + Run:** gõ phím không gọi API; chỉ nút **Run Inference** hoặc Enter mới chạy. Mặc định ô nhập `chr7:140753336:A>T`.
- **Preset** (chỉ điền ô nhập): BRAF p.Val600Glu (`chr7:140753336:A>T`), BRCA1 p.Arg1699Gln (`chr17:43063930:C>T`), BRCA2 p.Asn372His (`chr13:32332592:A>C`) — toạ độ GRCh38.
- **Đổi tổ hợp đặc trưng** sau khi đã chạy → tự chạy lại cho biến thể đã submit.
- **Live Bypass Mode** → `use_cache=false`.
- **Giữ kết quả khi chuyển trang:** trạng thái lưu trong `trimvpp_single_state_v1`. Quay lại từ menu hoặc nút *Back to Single Variant* của XAI → khôi phục ngay, **không gọi API**. URL trỏ tới biến thể/tổ hợp khác (vd bấm một dòng ở Batch) → chạy biến thể đó. Kết quả cũ chỉ bị thay khi tra biến thể mới hoặc đổi tổ hợp.
- **Thẻ kết quả:** badge trạng thái (`Computing…` / `Cached Result (Xms)` / `Live Inference (Xms)`); toạ độ, gene, protein change; nút **Inspect IG & Attention**; nhãn TriMVPP (badge lớn, kèm biểu tượng ⚠/✓); **xác suất và ngưỡng τ là hai số lớn đặt cạnh nhau** (xác suất đỏ/xanh theo nhãn, ngưỡng màu đen); thanh xác suất có vạch ngưỡng; dòng quy tắc `≥ τ → Pathogenic · < τ → Benign · MCC-optimal threshold of trimvpp_xxx.pth`; khối ClinVar Ground Truth (Consistent / Discrepancy / Uncertain) ngay bên dưới.
- **Bảng SOTA:** TriMVPP, AlphaMissense (ngưỡng hiển thị 0,56), REVEL (0,5); `null` → "N/A".

### 5.5 Trang Batch Screening (`/batch`)

- Kéo-thả/chọn file (`.vcf, .csv, .txt, .parquet, .xlsx`); không chọn file → cohort demo.
- **Start Screening / Stop Process**; panel ablation khoá khi đang stream; `runMask` lưu tổ hợp của lần chạy.
- Thanh tiến độ + 3 thống kê (Processed, Missense Valid, Gatekeeper Rejected).
- Bảng Accepted Missense Variants (tối đa 500 dòng): bấm một dòng → `/single?variant=…&mask=<runMask>&from=batch`.
- **Top Pathogenic Triage:** tối đa 5 biến thể Pathogenic có xác suất cao nhất; nút **View Explainability** → `/xai?…&from=batch`.
- **Export CSV** → `TriMVPP_Batch_Results_<timestamp>.csv`.

### 5.6 Trang XAI (`/xai`)

- Đọc `variant` và `mask` từ URL, gọi `GET /xai/{variant}?ablation_mask=<mask>`.
- Nút Back theo ngữ cảnh: `from=batch` → Batch; `from=single` → Single (giữ `origin=batch` nếu Single được mở từ Batch).
- **Cột trái:** Target Variant Context (nhãn, xác suất, τ); Active Modules; **Attribution Quality** (logit biến thể, logit baseline, Σ IG, residual + %, số bước; thông báo xanh nếu < 5%, vàng nếu vượt); **Top Drivers** (3 đầu vào mạnh nhất mỗi phía, xếp hạng trên cả token và đặc trưng).
- **Cột phải:** IG by Branch → IG by Token Position (DNA, Protein) → IG by Feature (28) → Cross-Attention Maps (hoặc ghi chú "single branch, no cross-attention" cho 100/010/001) → How to Read This Page.

### 5.7 Biến môi trường frontend

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000/api/v1` | Base URL backend |
| `VITE_USE_MOCK_API` | `false` | `true` = demo offline |

---

## 6. Deployment Topology & Performance Benchmarks

### 6.1 Topology

```
┌──────────────────────── Windows 11 host (RTX 4060 Laptop, 8 GB) ───────────────┐
│  Browser ──HTTP──► Vite dev server :5173 (React SPA)                            │
│     │                                                                           │
│     └──REST / SSE──► Uvicorn :8000  FastAPI (backend/.venv, Python 3.10)        │
│                        │  GPU: NT-v1 500M + ESM-1b 650M (FP16), 7 TriMVPP (FP32)│
│                        │  RAM: FAISS IndexFlatL2 ×2, scalers, pyfaidx            │
│                        │  Disk: SpliceAI VCF.gz + .tbi, hg38/pep FASTA          │
│                        ├──psycopg2──► localhost:5433 ─► [rbnet_postgres]        │
│                        │                               pgvector/pgvector:pg16   │
│                        └──`docker exec`──────────────► [rbnet_vep]              │
│                              (VCF/TSV qua bind mount)  ensemblorg/ensembl-vep   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Docker

- **`docker-compose.yml`** chỉ có service `db` (`pgvector/pgvector:pg16`, container `rbnet_postgres`, cổng `5433:5432`, volume `postgres_data`, healthcheck `pg_isready`). Mật khẩu DB nằm trong file này và trong `backend/.env` — không đưa vào tài liệu.
- **`rbnet_vep`** là container độc lập (không thuộc compose): image `ensemblorg/ensembl-vep:latest`; bind mounts `F:/VEP_RESOURCES → /vep_resources`, `F:/VEP_CACHE → /vep_cache`, `F:/BioData/vep_data/tmp_io → /tmp_io`. Backend gọi `docker exec rbnet_vep vep …`.

### 6.3 Môi trường Python GPU (`backend/.venv`)

```powershell
$env:PIP_CACHE_DIR="F:\pip-cache"; $env:TMP="F:\tmp"; $env:TEMP="F:\tmp"
python -m venv .venv
.venv\Scripts\pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\pip install -r requirements.txt
python scripts/download_models.py   # tải NT-v1 + ESM-1b về HF_HOME, chuyển .bin → safetensors (clone tensor dùng chung bộ nhớ)
python scripts/init_db.py           # pgvector + 3 bảng + migration
python scripts/check_db.py          # (tuỳ chọn) kiểm tra DB
```

### 6.4 Biến môi trường backend (`Settings`, `core/config.py`)

Repo không có `backend/.env.example`. Các khoá bắt buộc: `PROJECT_NAME`, `API_V1_STR`, `DATABASE_URL`, `BACKEND_CORS_ORIGINS`. Thiết lập chính và mặc định trong code:

| Khoá | Mặc định | Ý nghĩa |
|---|---|---|
| `LOG_LEVEL` / `PRELOAD_ON_STARTUP` | `INFO` / `True` | |
| `VEP_CONTAINER_NAME`, `VEP_TMP_DIR`, `VEP_CACHE_VERSION`, `VEP_TIMEOUT_SECONDS`, `VEP_FORK_THRESHOLD`/`WORKERS` | `rbnet_vep`, `F:/BioData/vep_data/tmp_io`, `116`, `900`, `50`/`4` | VEP |
| `HG38_FASTA_PATH`, `HG38_PEP_FASTA_PATH`, `SPLICEAI_VCF_PATH` | `F:/VEP_RESOURCES/…` | Tài nguyên tham chiếu |
| `DNA_MODEL_ID` / `PROT_MODEL_ID` | `InstaDeepAI/nucleotide-transformer-500m-human-ref` / `facebook/esm1b_t33_650M_UR50S` | Foundation models |
| `DNA_CONFIG_ID` / `PROT_CONFIG_ID` | `nt_v1_500m_dna_1001_center` / `esm1b_650m_prot_501_center` | Tên FAISS index + geom scaler |
| `DNA_WINDOW` / `PROT_WINDOW` | `1001` / `501` | |
| `INFERENCE_DEVICE` / `FM_CUDA_DTYPE` | `auto` / `fp16` | |
| `FM_BATCH_VARIANTS` | `8` | Biến thể/forward FM |
| `HF_HOME` | `None` | Thư mục trọng số HuggingFace |
| `MODEL_ARTIFACTS_DIR` | `artifacts/trimvpp/geom_scalers` | Geom scalers |
| `BIO_SCALER_DIR` | `artifacts` | 6 bio scalers |
| `FAISS_INDEX_DIR` | `artifacts/trimvpp/faiss_indexes` | |
| `GEOM_K_NEIGHBORS` | `64` | k cho LID |
| `MODEL_CHECKPOINT_REGISTRY` / `MODEL_DECISION_THRESHOLDS` / `MODEL_TEST_MCC` | Mục 4.3 | JSON, merge với mặc định |
| `IG_BASELINE_PATH` | `artifacts/trimvpp/ig_baseline_train3_seed2026.pt` | Baseline IG |
| `IG_STEPS` / `IG_RETRY_STEPS` / `IG_RETRY_THRESHOLD` | `64` / `128` / `0.05` | Tích phân IG |
| `IG_INTERNAL_BATCH` | `8` | Điểm nội suy mỗi forward/backward |
| `SPLICEAI_WINDOW_SIZE` | `1024` | |

### 6.5 Launcher `start_system.bat` / `stop_system.bat`

`start_system.bat`: kiểm tra `.venv` → **[1/3]** `docker compose up -d db` → **[2/3]** cửa sổ "TriMVPP Backend (FastAPI)": `uvicorn app.main:app --port 8000` (không `--reload`) → **[3/3]** cửa sổ "TriMVPP Frontend (React/Vite)": `npm run dev` → mở `http://localhost:5173`. Launcher **không** khởi động `rbnet_vep` (phải chạy sẵn, nếu không cache-miss trả 503 `VEP_UNAVAILABLE`). `stop_system.bat`: `docker compose stop db` và hướng dẫn đóng 2 cửa sổ.

### 6.6 Số liệu hiệu năng (đo ngày 26/11/2026)

**Máy đo:** AMD Ryzen 7 8745H, 32 GB RAM, RTX 4060 Laptop 8 GB, Windows 11; FM ở FP16. Cohort demo 10 biến thể, một người dùng gửi tuần tự; mỗi kịch bản lặp 3 lần (cache hit: 10 lần) sau một lần khởi động, lấy **trung vị**. Thời gian đo ở backend (`meta.computation_time_ms`), trừ Batch và mở trang XAI đo ở client.

| Kịch bản | Đường thực thi | Độ trễ |
|---|---|---|
| Biến thể mới (Single) | cache miss | **3,54 s** |
| Batch 10 biến thể mới | cache miss | **10,9 s** tổng (1,09 s/biến thể) |
| Đổi tổ hợp / Live Bypass — 111 | feature reuse | **0,33 s** |
| Feature reuse — 110 / 101 / 011 | | 0,24 / 0,25 / 0,25 s |
| Feature reuse — 100 / 010 / 001 | | 0,06 / 0,06 / 0,07 s |
| Truy vấn lặp lại | cache hit | **18 ms** |
| Mở trang XAI (đã có dự đoán) — 111 / 2 nhánh / 1 nhánh | cache + attention | 0,15 s / 0,08–0,10 s / 0,02–0,03 s |
| Batch 10 biến thể chạy lại | cache hit ×10 | 0,19 s |

**Từng chặng khi cache miss (trung vị):** VEP 2 789 ms (~80%), ESM-1b 301 ms, IG 264 ms, SpliceAI 146 ms, DB 79 ms, NT-v1 28 ms (phần lớn chạy song song với VEP), chuẩn hoá 22 ms, geometry 7 ms.

**So với phiên bản dùng SHAP (permutation Shapley, 128 hoán vị):** biến thể mới 4,78 s → 3,54 s; Batch 10 biến thể 23,4 s → 10,9 s; feature reuse 111: 1,59 s → 0,33 s; cache hit không đổi (18 ms).

**Kiểm tra chức năng (bước D):** 10 biến thể × 7 tổ hợp: completeness đúng, residual tương đối lớn nhất 0,88%; attention có đúng ở 4 tổ hợp nhiều nhánh, mỗi hàng cộng bằng 1; xác suất Batch trùng tuyệt đối với Single; tắt hết nhánh → 400.

**Đối chiếu dự đoán với kết quả offline (Phase 1):** 440 biến thể test qua toàn bộ pipeline web: |Δp| trung bình ≤ 0,0011 ở mọi tổ hợp; nhãn trùng hoàn toàn ở 5 tổ hợp (gồm 111), lệch 2 biến thể (010) và 1 biến thể (001) có xác suất sát ngưỡng.

---

## 7. Ghi chú đối chiếu & các điểm cần lưu ý khi viết luận văn

1. **Xác suất không so được giữa các tổ hợp.** Mỗi tổ hợp là một model huấn luyện riêng với ngưỡng τ riêng (0,48–0,75). Tắt một nhóm có thể làm xác suất **tăng** nếu nhóm đó đang kéo về Benign (ví dụ BRAF V600E: 111 cho 51,2%, IG cho thấy token Protein kéo −2,79; 101 (tắt Protein) cho 97,0%). Cần so xác suất với τ của chính tổ hợp đó.
2. **IG giải thích logit, không phải xác suất**, so với một baseline cố định (trung bình/trung vị của 256 biến thể train). Đóng góp của token REF cũng khác 0 vì baseline là token trung bình, không phải "alt = ref".
3. **Attention không phải độ quan trọng.** Trọng số attention cho biết nhánh query đọc thông tin từ đâu; ở nhiều hàng trọng số gần đều (≈ 1/số key). Attention không được lưu DB.
4. **Đối chiếu IG ở bước A** dùng 11 đặc trưng bio lấy từ annotation đã lưu của tập test (Docker tắt lúc đó); các đầu vào khác do pipeline web trích xuất. Logit lệch tối đa 0,038 đến từ geometry/embedding trích lại ở FP16.
5. **`.env.example` không tồn tại**; `rbnet_vep` không thuộc `docker-compose.yml` và không được launcher khởi động.
6. **Tên container/DB** (`rbnet_postgres`, `rbnet_vep`, `rbnet_genomics`) giữ từ phiên bản cũ; không ảnh hưởng chức năng.
7. **Comment trong `models/variant.py`** ghi `clinvar_ground_truth` là nhị phân, nhưng code lưu cả `Uncertain`.
8. **`danger_rank`** trong SSE bằng thứ tự đến; xếp hạng Top-5 do frontend làm.
9. **Badge "PostgreSQL Connected"** ở trang Single/Batch là nhãn tĩnh, không phản ánh `/health`.
10. **Live Bypass** chỉ bỏ qua bảng `predictions`; không chạy lại VEP/FM nếu đặc trưng đã có.
11. **Upload** đọc toàn bộ file vào bộ nhớ (`await file.read()`), không giới hạn kích thước phía server.
12. **Component chưa dùng:** `variant/FeatureInspectorTabs.tsx` (cùng kiểu `BioFeatures`, `GeometryFeatures`) không được trang nào import.
13. **Hình trong luận văn:** 5.1–5.3 (`.svg` → `.png` 2×), 5.4a–e (ảnh chụp giao diện 1440×1000, tỉ lệ 1,5×). `figures/5.4.svg/png` là sơ đồ DB cũ, Chương 5 không dùng.
