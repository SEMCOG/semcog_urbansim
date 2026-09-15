"""Create a standalone HTML summary of the priority and standard REPM diagnostics."""
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path("/home/da/RDF2055/d_drive/estimation/REPM")
RUNS = {
    "Priority": ROOT / "repm_20260904_090443_compact_priority",
    "Standard": ROOT / "repm_20260904_091119_compact_standard",
}
# Decisions use the best compact result only where it was at least as useful as
# the reviewed broad specification and materially reduces model size.
DECISIONS = {94: "compact_30", 14731: "compact_50", 11533: "compact_50", 14733: "compact_30"}


def fmt(value):
    return f"{value:.3f}"


def source_link(run, name):
    return f'../{run.name}/{name}'


def load_batch(label, run):
    summary = pd.read_csv(run / "compact_priority_summary.csv")
    report = json.loads((run / "compact_priority_report.json").read_text())
    samples = {item["hedonic_id"]: item["sample_size"] for item in report["segments"]}
    rows = []
    for hid, group in summary.groupby("hedonic_id", sort=True):
        broad = group.loc[group.specification == "reviewed_broad"].iloc[0]
        compact = group.loc[group.specification != "reviewed_broad"].sort_values(
            ["r2_test_mean", "rmse_test_log_mean"], ascending=[False, True]
        ).iloc[0]
        selected = DECISIONS.get(hid, "reviewed_broad")
        selected_row = broad if selected == "reviewed_broad" else group.loc[
            group.specification == selected
        ].iloc[0]
        rows.append({
            "batch": label,
            "hedonic_id": int(hid),
            "market": broad.market.replace("_", " "),
            "sample_size": samples[int(hid)],
            "broad": broad,
            "compact": compact,
            "selected": selected,
            "selected_row": selected_row,
        })
    return rows


def row_html(row):
    broad, compact = row["broad"], row["compact"]
    selected = row["selected"]
    decision = "Compact" if selected != "reviewed_broad" else "Reviewed broad"
    badge = "compact" if selected != "reviewed_broad" else "broad"
    compact_name = compact.specification.replace("_", " ")
    return f"""<tr>
      <td>{row['hedonic_id']}</td><td>{html.escape(row['market'].title())}</td><td>{row['sample_size']:,}</td>
      <td>{int(broad.n_features)}</td><td>{fmt(broad.r2_test_mean)}</td><td>{fmt(broad.rmse_test_log_mean)}</td>
      <td>{html.escape(compact_name)}</td><td>{int(compact.n_features)}</td><td>{fmt(compact.r2_test_mean)}</td><td>{fmt(compact.rmse_test_log_mean)}</td>
      <td><span class=\"badge {badge}\">{decision}</span></td>
    </tr>"""


def table_html(label, rows):
    body = "\n".join(row_html(row) for row in rows)
    run = RUNS[label]
    return f"""<section><h2>{label} segments</h2>
    <p class=\"source\">Source: <a href=\"{source_link(run, 'compact_priority_summary.csv')}\">summary CSV</a> ·
    <a href=\"{source_link(run, 'compact_priority_report.json')}\">run report</a></p>
    <div class=\"table-wrap\"><table><thead><tr>
    <th>Hedonic ID</th><th>Market</th><th>Sample</th><th>Broad features</th><th>Broad R²</th><th>Broad RMSE*</th>
    <th>Best compact</th><th>Compact features</th><th>Compact R²</th><th>Compact RMSE*</th><th>Recommendation</th>
    </tr></thead><tbody>{body}</tbody></table></div></section>"""


def main():
    priority = load_batch("Priority", RUNS["Priority"])
    standard = load_batch("Standard", RUNS["Standard"])
    all_rows = priority + standard
    compact_rows = [row for row in all_rows if row["selected"] != "reviewed_broad"]
    generated = datetime.now(ZoneInfo("America/Detroit")).strftime("%Y-%m-%d %H:%M %Z")
    recommendation_rows = "\n".join(
        f"<tr><td>{row['hedonic_id']}</td><td>{html.escape(row['market'].title())}</td>"
        f"<td>{row['selected'].replace('_', ' ')}</td><td>{int(row['selected_row'].n_features)}</td>"
        f"<td>{fmt(row['selected_row'].r2_test_mean)}</td><td>{fmt(row['selected_row'].rmse_test_log_mean)}</td></tr>"
        for row in compact_rows
    )
    page = f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>REPM Compact Model Review</title>
    <style>
    body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#172033;max-width:1500px;margin:0 auto;padding:28px;line-height:1.45;background:#fafbfc}}
    h1{{margin-bottom:0}} h2{{margin-top:36px}} .muted,.source{{color:#596579}} .cards{{display:flex;gap:14px;flex-wrap:wrap;margin:24px 0}}
    .card{{background:white;border:1px solid #dce2ea;border-radius:8px;padding:16px;min-width:175px}}.num{{font-size:1.8rem;font-weight:700;color:#145b8c}}
    table{{border-collapse:collapse;width:100%;background:#fff;font-size:.91rem}}th,td{{border:1px solid #dce2ea;padding:8px;text-align:right;white-space:nowrap}}th{{background:#eaf3f8;text-align:center}}td:nth-child(2){{text-align:left}}
    .table-wrap{{overflow-x:auto}}.badge{{padding:3px 7px;border-radius:999px;font-weight:650;font-size:.83rem}}.compact{{background:#d9f4e4;color:#135c32}}.broad{{background:#e7eef7;color:#294d72}}
    .note{{background:#fff7df;border-left:4px solid #d99400;padding:12px 16px}} a{{color:#075d94}} footer{{margin-top:32px;color:#596579;font-size:.9rem}}
    </style></head><body>
    <h1>REPM compact model review</h1><p class=\"muted\">Priority and standard non-residential segments · generated {generated} (Eastern Time)</p>
    <div class=\"cards\"><div class=\"card\"><div class=\"num\">15</div>segments reviewed</div><div class=\"card\"><div class=\"num\">4</div>compact models recommended</div><div class=\"card\"><div class=\"num\">11</div>retain reviewed broad specification</div><div class=\"card\"><div class=\"num\">25</div>validation folds per specification</div></div>
    <p>The preferred specification is based on repeated 5×5 cross-validation. “Reviewed broad” means the theory- and data-screened candidate set after fold-specific variance and correlation screening; it is not the original 320-variable estimation. Compact models use mutual-information selection inside each outer training fold.</p>
    <section><h2>Recommended compact specifications</h2><div class=\"table-wrap\"><table><thead><tr><th>Hedonic ID</th><th>Market</th><th>Specification</th><th>Features</th><th>Validation R²</th><th>Validation RMSE*</th></tr></thead><tbody>{recommendation_rows}</tbody></table></div></section>
    {table_html('Priority', priority)}
    {table_html('Standard', standard)}
    <section><h2>Interpretation and guardrails</h2><ul><li>Higher validation R² and lower validation RMSE indicate better out-of-sample fit. RMSE is on the log price-per-square-foot target scale.</li><li>Compact specifications are recommended only where they match or improve the reviewed broad result while using substantially fewer features.</li><li>The variable screen retains physical/site, land-use, employment-market, and self-excluded peer-price context. It excludes bike indicators, demographic/travel-behavior proxies, and the currently unreliable zoning capacity/FAR fields.</li><li>Peer-price fields exclude the focal building itself, but validation buildings may still contribute to one another’s local peer context. Treat these values as a current practical specification, not a fully isolated holdout design.</li></ul>
    <div class=\"note\"><strong>Do not compare these validation R² values directly with the full-estimation training summary.</strong> This report uses a smaller, repeated-CV diagnostic with feature selection repeated inside each training fold. Its main use is comparison of specifications within the same hedonic segment.</div></section>
    <footer>Evidence basis: <a href=\"https://www.pbl.nl/en/publications/a-hedonic-price-analysis-of-the-value-of-industrial-sites\">PBL industrial-site hedonic analysis</a>, <a href=\"https://ec.europa.eu/eurostat/documents/7870049/8545612/KS-FT-16-001-EN-N.pdf\">Eurostat commercial-property hedonic guidance</a>, and <a href=\"https://scikit-learn.org/stable/common_pitfalls.html\">scikit-learn validation guidance</a>.</footer>
    </body></html>"""
    out_dir = ROOT / f"repm_{datetime.now(ZoneInfo('America/Detroit')).strftime('%Y%m%d_%H%M%S')}_compact_summary"
    out_dir.mkdir(parents=True, exist_ok=False)
    out_path = out_dir / "compact_model_summary.html"
    out_path.write_text(page)
    print(out_path)


if __name__ == "__main__":
    main()
