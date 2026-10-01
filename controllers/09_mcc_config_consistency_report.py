"""Compare OURS configurations across datasets using MCC.

The report is intentionally dataset-aware: each dataset gets its own OURS
top-1, and a configuration is considered robust only when it is within the
requested tolerance of every dataset-specific top-1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONFIG_COLUMNS = [
    "Pooling",
    "Ablation",
    "DNA_Model",
    "Prot_Model",
    "Network",
    "Geom_Features",
]
METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "Specificity", "F1_Score", "MCC", "AUROC", "AUPRC"]


def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().replace({"": "None", "nan": "None"})


def _display_config(row: pd.Series) -> str:
    return " | ".join(f"{column}={row[column]}" for column in CONFIG_COLUMNS)


def _format_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_Không có dữ liệu._"
    output = frame[columns].copy()
    for column in output.columns:
        if pd.api.types.is_numeric_dtype(output[column]):
            if pd.api.types.is_integer_dtype(output[column]):
                output[column] = output[column].map(lambda value: f"{int(value)}")
            else:
                output[column] = output[column].map(lambda value: f"{float(value):.4f}")
        elif column in {"MCC", "Delta_to_top1", "Required_tolerance"}:
            output[column] = output[column].map(lambda value: f"{float(value):.4f}")
    return output.to_markdown(index=False)


def analyze(input_csv: Path, output_md: Path, tolerance: float = 0.02) -> None:
    raw = pd.read_csv(input_csv)
    required = {"Dataset", "Source", "MCC", *CONFIG_COLUMNS}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = raw.copy()
    data["Source"] = _clean_text(data["Source"]).str.upper()
    data = data.loc[data["Source"].eq("OURS")].copy()
    data["Dataset"] = _clean_text(data["Dataset"])
    for column in CONFIG_COLUMNS:
        data[column] = _clean_text(data[column])
    data["MCC"] = pd.to_numeric(data["MCC"], errors="coerce")
    data = data.dropna(subset=["MCC"])

    # A compare CSV can contain duplicate rows for the same dataset/config.
    # Keep the strongest MCC so the comparison is about the config itself.
    grouped = (
        data.groupby(["Dataset", *CONFIG_COLUMNS], dropna=False, as_index=False)["MCC"]
        .max()
    )
    top_by_dataset = grouped.groupby("Dataset", as_index=False)["MCC"].max().rename(columns={"MCC": "Top1_MCC"})
    scored = grouped.merge(top_by_dataset, on="Dataset", how="left")
    scored["Delta_to_top1"] = scored["Top1_MCC"] - scored["MCC"]
    scored["Within_0.02"] = scored["Delta_to_top1"].le(tolerance + 1e-12)

    dataset_order = list(dict.fromkeys(grouped["Dataset"].tolist()))
    top_rows = []
    for dataset in dataset_order:
        row = scored.loc[scored["Dataset"].eq(dataset)].sort_values("MCC", ascending=False).iloc[0]
        top_rows.append({
            "Dataset": dataset,
            "Top1_MCC": row["MCC"],
            "Top1_config": _display_config(row),
        })
    top_table = pd.DataFrame(top_rows)

    near_counts = (
        scored.groupby("Dataset")["Within_0.02"].sum()
        .reindex(dataset_order)
        .fillna(0)
        .astype(int)
    )
    near_table = pd.DataFrame({"Dataset": dataset_order, "Configs_within_0.02": near_counts.to_numpy()})

    coverage = (
        scored.pivot_table(
            index=CONFIG_COLUMNS,
            columns="Dataset",
            values="Within_0.02",
            aggfunc="max",
            fill_value=False,
        )
        .reset_index()
    )
    dataset_flags = [column for column in coverage.columns if column not in CONFIG_COLUMNS]
    present = (
        scored.groupby(CONFIG_COLUMNS, dropna=False, as_index=False)["Dataset"]
        .nunique()
        .rename(columns={"Dataset": "Datasets_present"})
    )
    coverage = coverage.merge(present, on=CONFIG_COLUMNS, how="left")
    coverage["Datasets_within_0.02"] = coverage[dataset_flags].sum(axis=1).astype(int)
    full_dataset_count = len(dataset_order)
    coverage["All_datasets_within_0.02"] = coverage["Datasets_within_0.02"].eq(full_dataset_count) & coverage["Datasets_present"].eq(full_dataset_count)
    robust = coverage.loc[coverage["All_datasets_within_0.02"]].copy()
    delta_matrix = scored.pivot_table(
        index=CONFIG_COLUMNS,
        columns="Dataset",
        values="Delta_to_top1",
        aggfunc="max",
    ).dropna()
    if delta_matrix.empty:
        required_tolerance = None
        required_config_values = None
        required_config = None
    else:
        delta_matrix["Required_tolerance"] = delta_matrix.max(axis=1)
        required_tolerance = float(delta_matrix["Required_tolerance"].min())
        required_idx = delta_matrix["Required_tolerance"].idxmin()
        required_config_values = dict(zip(CONFIG_COLUMNS, required_idx))
        required_config = delta_matrix.loc[required_idx]
    max_coverage = int(coverage["Datasets_within_0.02"].max()) if not coverage.empty else 0
    best_coverage = coverage.loc[coverage["Datasets_within_0.02"].eq(max_coverage)].copy()
    best_coverage = best_coverage.sort_values(["Datasets_present", *CONFIG_COLUMNS], ascending=[False, *([True] * len(CONFIG_COLUMNS))])

    lines = [
        f"# MCC config consistency: `{input_csv.parent.name}`",
        "",
        "## Phạm vi và quy ước",
        "",
        "- Chỉ dùng các dòng có `Source=OURS`; không dùng các model SOTA để xác định top 1.",
        "- Metric duy nhất: `MCC`.",
        f"- Top 1 được tính riêng theo từng dataset; ngưỡng đạt là `top1_MCC - MCC <= {tolerance:.2f}`.",
        "- Một config được coi là giống nhau khi toàn bộ 6 trường sau giống nhau: " + ", ".join(f"`{column}`" for column in CONFIG_COLUMNS) + ".",
        "- Các dòng trùng dataset/config được gộp bằng MCC lớn nhất; giá trị rỗng được chuẩn hóa thành `None`.",
        "",
        "## Tổng quan",
        "",
        f"- OURS rows ban đầu: **{len(data):,}**.",
        f"- Dataset/config duy nhất sau khi gộp: **{len(grouped):,}**.",
        f"- Config duy nhất xuất hiện trên toàn bộ {full_dataset_count} dataset: **{int((coverage['Datasets_present'] == full_dataset_count).sum()):,}**.",
        f"- Config đạt ngưỡng trên ít nhất một dataset: **{int((coverage['Datasets_within_0.02'] > 0).sum()):,}**.",
        f"- Config đạt ngưỡng trên cả {full_dataset_count} dataset: **{len(robust):,}**.",
        "",
        "## Top 1 OURS theo dataset",
        "",
        _format_table(top_table, ["Dataset", "Top1_MCC", "Top1_config"]),
        "",
        "## Số config nằm trong ngưỡng 0.02",
        "",
        _format_table(near_table, ["Dataset", "Configs_within_0.02"]),
        "",
        "## Kiểm tra tính ổn định xuyên dataset",
        "",
    ]

    if robust.empty:
        lines.extend([
            f"**Không có config nào** đồng thời nằm trong ngưỡng 0.02 của top 1 trên cả {full_dataset_count} dataset.",
            "",
            f"Mức bao phủ tốt nhất chỉ là **{max_coverage}/{full_dataset_count} dataset**.",
        ])
    else:
        lines.extend([
            f"Có **{len(robust)} config** đạt đồng thời trên cả {full_dataset_count} dataset:",
            "",
            _format_table(robust, [*CONFIG_COLUMNS, "Datasets_present", "Datasets_within_0.02"]),
        ])
    if required_config is not None:
        lines.extend([
            "",
            "### Ngưỡng nhỏ nhất để có config phủ đủ tất cả dataset",
            "",
            f"Cần nới ngưỡng tối thiểu lên **{required_tolerance:.4f}**. Đây là giá trị nhỏ nhất của `max(delta_to_top1)` trên một config xuất hiện ở toàn bộ dataset.",
            "",
            _format_table(
                pd.DataFrame([{
                    **required_config_values,
                    **{dataset: required_config[dataset] for dataset in dataset_order},
                    "Required_tolerance": required_tolerance,
                }]),
                [*CONFIG_COLUMNS, *dataset_order, "Required_tolerance"],
            ),
        ])
    lines.extend(["", "### Các config có mức bao phủ tốt nhất", ""])
    best_columns = [*CONFIG_COLUMNS, "Datasets_present", "Datasets_within_0.02"]
    lines.append(_format_table(best_coverage.head(10), best_columns))

    for dataset in dataset_order:
        near = scored.loc[scored["Dataset"].eq(dataset) & scored["Within_0.02"]].sort_values(
            ["MCC", *CONFIG_COLUMNS], ascending=[False, *([True] * len(CONFIG_COLUMNS))]
        )
        lines.extend([
            "",
            f"## Top 10 config gần top 1: `{dataset}`",
            "",
            _format_table(near.head(10), [*CONFIG_COLUMNS, "MCC", "Delta_to_top1"]),
        ])

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_md", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
    analyze(args.input_csv, args.output_md, args.tolerance)
    print(f"[OK] Wrote {args.output_md}")


if __name__ == "__main__":
    main()
