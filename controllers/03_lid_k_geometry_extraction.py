"""Extract train3-context geometry for the LID-k ablation."""

from core.module03_geometry_extraction.geometry_v2 import GeometryV2Config, run_geometry_v2


BASE_DIR = "D:/variant_data"
CONTEXT = "train3"
SPLITS = ["train3", "val", "test", "clinvarhq", "uniprot", "proteingym"]
MODELS = ["nt_v2_500m", "esm1b_650m"]
POOLINGS = ["mean"]
K_VALUES = [8, 16, 32, 64]


def main():
    for k in K_VALUES:
        geometry_root = "geometry_v2" if k == 32 else f"geometry_v2_k{k}"
        normalization_root = (
            "normalization_artifacts_v2"
            if k == 32
            else f"normalization_artifacts_v2_k{k}"
        )
        print(f"\n{'=' * 80}\n[LID] Extracting k={k}\n{'=' * 80}")
        result = run_geometry_v2(
            GeometryV2Config(
                base_dir=BASE_DIR,
                contexts=[CONTEXT],
                splits=SPLITS,
                models=MODELS,
                poolings=POOLINGS,
                k_neighbors=k,
                resume=True,
                geometry_root=geometry_root,
                normalization_root=normalization_root,
            )
        )
        if not result.empty:
            print(result.to_string(index=False))


if __name__ == "__main__":
    main()
