# RB-Net Frontend

React 19 + Vite + Tailwind UI for the RB-Net missense variant pathogenicity system.

## Run

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # type-check + production build into dist/
npm run lint     # oxlint
```

## Environment variables (`.env.local`)

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000/api/v1` | FastAPI backend base URL |
| `VITE_USE_MOCK_API` | `false` | `true` = offline demo using `src/services/api.mock.ts` |

## Layout

```
src/
  pages/            Home, Single (single variant), Batch (VCF screening, SSE), Xai (SHAP)
  services/api.ts   FastAPI client (single, xai, batch-upload / batch-stream SSE)
  services/api.mock.ts  offline simulation used only when VITE_USE_MOCK_API=true
  lib/ablation.ts   ablation mask helpers (bit 3 DNA, 2 Protein, 1 Bio, 0 Geometry)
  types/index.ts    all shared types (mirrors backend/app/schemas)
  mock/mockData.json presets for the Single page + mock-mode data
```

API errors have one shape: `{"detail": {"error_code", "message", "request_id"}}`.
SOTA scores (`alphamissense`, `revel`) are `null` when dbNSFP has no score and are shown as "N/A".
