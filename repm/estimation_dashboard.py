"""Build a standalone, local-file dashboard for a completed REPM estimation."""
import html
import json
import os
from pathlib import Path

import pandas as pd
import yaml


RUN = Path("/home/da/RDF2055/d_drive/estimation/REPM/repm_20260904_102106")
INPUT_HDF = Path(os.environ.get(
    "SEMCOG_INPUT_HDF", "/home/da/RDF2055/d_drive/forecast_inputs/base_year/main_082426.h5"
))
COMPACT_DECISIONS = {
    94: {"decision": "Reviewed broad", "reason": "Compact fit was lower; retain the broader screened model."},
    14731: {"decision": "Compact 50", "reason": "Near-tied fit with 50 instead of 91 reviewed-broad features."},
    11533: {"decision": "Compact 50", "reason": "Higher repeated-CV R² and lower RMSE with 50 features."},
    14733: {"decision": "Compact 30", "reason": "Near-tied/slightly better repeated-CV fit with 30 instead of 95 features."},
}
COMPACT_CV = {
    94: (0.2347, 0.2242, 167, 30),
    14731: (0.3489, 0.3484, 91, 50),
    11533: (0.6707, 0.6772, 91, 50),
    14733: (0.5978, 0.5989, 95, 30),
}


def friendly(name):
    return name.replace("_excl_self", " (peer, self excluded)").replace("_", " ")


def load_hedonic_labels():
    building_types = pd.read_hdf(INPUT_HDF, "building_types")
    large_areas = pd.read_hdf(INPUT_HDF, "large_areas")
    regional = {
        11: "Institutional (aggregated)", 32: "Wholesale Trade", 41: "Transportation & Utility (aggregated)",
        51: "Medical (aggregated)", 61: "Leisure & Hospitality (aggregated)", 71: "Agricultural",
        84: "Mobile Home", 94: "Death Care Services", 96: "Data Center",
    }
    def label(hedonic_id):
        if hedonic_id in regional:
            return "Regional", regional[hedonic_id], f"Regional · {regional[hedonic_id]}"
        area_id, type_id = divmod(hedonic_id, 100)
        area = large_areas.loc[area_id, "large_area_name"].replace(
            "Wayne County, excluding Detroit", "Wayne excl. Detroit"
        ).replace(" County", "")
        building_type = building_types.loc[type_id, "building_type_name"]
        return area, building_type, f"{area} · {building_type}"
    return label


def load_models():
    rows = []
    hedonic_label = load_hedonic_labels()
    for path in sorted(RUN.iterdir()):
        summary_path = path / "summary.yaml"
        if not summary_path.exists():
            continue
        summary = yaml.safe_load(summary_path.read_text())
        perf = summary["performance"]
        hid = int(summary["hedonic_id"])
        area, building_type, label = hedonic_label(hid)
        rows.append({
            "name": summary["model_name"], "hedonic_id": hid, "type": summary["type"],
            "model_type": summary["model_type"], "sample_size": int(summary["sample_size"]),
            "n_features": int(summary["n_features"]), "r2_train": float(perf["r2_train"]) if perf["r2_train"] != "not_available" else None,
            "r2_val": float(perf["r2_val"]) if perf["r2_val"] != "not_available" else None,
            "rmse_val": float(perf["rmse_val"]) if perf["rmse_val"] != "not_available" else None,
            "mae_val": float(perf["mae_val"]) if perf["mae_val"] != "not_available" else None,
            "fixed_compact": bool(summary.get("fixed_feature_specification", False)),
            "large_area": area, "building_type": building_type, "label": label,
            "short_id": f"repm{hid}",
            "decision": COMPACT_DECISIONS.get(hid, {}).get("decision", "Standard specification"),
            "top_features": [{"name": name, "label": friendly(name), "importance": value}
                             for name, value in summary.get("top_features", {}).items()],
        })
    return rows


def main():
    training = yaml.safe_load((RUN / "training_summary.yaml").read_text())
    models = load_models()
    residential = [row for row in models if row["type"] == "residential" and row["r2_val"] is not None]
    nonresidential = [row for row in models if row["type"] != "residential" and row["r2_val"] is not None]
    overview = {
        "models": len(models), "successful": training["successful"], "variables": training["n_variables"],
        "res_r2": sum(row["r2_val"] for row in residential) / len(residential),
        "nonres_r2": sum(row["r2_val"] for row in nonresidential) / len(nonresidential),
        "timestamp": training["timestamp"], "seconds": training["training_time_seconds"],
    }
    compact = [{"hedonic_id": hid, **COMPACT_DECISIONS[hid],
                "broad_r2": COMPACT_CV[hid][0], "compact_r2": COMPACT_CV[hid][1],
                "broad_features": COMPACT_CV[hid][2], "compact_features": COMPACT_CV[hid][3]}
               for hid in sorted(COMPACT_DECISIONS)]
    payload = json.dumps({"models": models, "overview": overview, "compact": compact}).replace("</", "<\\/")
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>REPM Estimation Results</title><style>
:root{{--ink:#172238;--muted:#607085;--line:#dbe4ec;--paper:#fff;--bg:#f5f8fb;--res:#2e72a8;--nonres:#26866d;--accent:#d98525}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1450px;margin:auto;padding:34px}} h1{{margin:0;font-size:2rem}} h2{{margin:38px 0 8px}} h3{{margin:0 0 8px}} .sub,.muted{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(5,minmax(125px,1fr));gap:14px;margin:25px 0}} .card,section{{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:18px}}
.value{{font-size:1.75rem;font-weight:750;color:#124f79}} .label{{font-size:.86rem;color:var(--muted)}} .grid{{display:grid;grid-template-columns:1.2fr .8fr;gap:18px}} .chart{{min-height:560px;overflow:auto}}
svg{{width:100%;min-width:760px}} .axis{{fill:var(--muted);font-size:11px}} .barlabel{{fill:#233148;font-size:11px}} .barvalue{{fill:#233148;font-size:11px;font-weight:650}}
.legend{{display:flex;gap:16px;color:var(--muted);font-size:.9rem}} .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}} .res{{background:var(--res)}} .nonres{{background:var(--nonres)}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}} th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right}} th{{color:var(--muted);font-weight:650}} td:nth-child(1),td:nth-child(2),th:nth-child(1),th:nth-child(2){{text-align:left}}
.scroll{{overflow:auto;max-height:560px}} button{{border:1px solid var(--line);background:white;border-radius:6px;padding:6px 10px;cursor:pointer}} button.active{{background:#e3f1f9;border-color:#6da4c8}} .controls{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}} input,select{{padding:7px;border:1px solid var(--line);border-radius:6px;background:white}}
.feature{{display:grid;grid-template-columns:minmax(170px,1fr) 2fr 48px;gap:8px;align-items:center;font-size:.88rem;margin:7px 0}} .track{{height:12px;background:#e9eef3;border-radius:6px;overflow:hidden}} .fill{{height:100%;background:#5d99c4;border-radius:6px}}
.callout{{border-left:4px solid var(--accent);padding:12px 14px;background:#fff7e8;margin-top:12px}} .tag{{font-size:.78rem;border-radius:999px;padding:3px 7px;background:#e9f5ef;color:#176047;font-weight:650}} footer{{color:var(--muted);font-size:.85rem;margin-top:28px}}
@media(max-width:900px){{main{{padding:18px}}.cards{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Real Estate Price Model estimation results</h1><p class="sub">Production candidate · {html.escape(overview['timestamp'])} · final estimation package</p>
<div class="cards"><div class="card"><div class="value">{overview['successful']} / {overview['models']}</div><div class="label">models trained successfully</div></div><div class="card"><div class="value">{overview['res_r2']:.3f}</div><div class="label">mean residential validation R²</div></div><div class="card"><div class="value">{overview['nonres_r2']:.3f}</div><div class="label">mean non-residential validation R²</div></div><div class="card"><div class="value">{overview['variables']}</div><div class="label">candidate input variables</div></div><div class="card"><div class="value">17.7 min</div><div class="label">training time; cached inputs</div></div></div>
<section><h2>How to read this dashboard</h2><p>Validation R² is the primary fit measure: higher is better. Validation RMSE and MAE are errors on the log(price-per-square-foot) target scale: lower is better. Feature importance shows each XGBoost model’s relative reliance on a feature; it does <strong>not</strong> show a positive/negative effect, causality, or statistical significance.</p><div class="callout"><strong>Important:</strong> compare model scores mainly within the same hedonic segment. Residential and non-residential markets have different price dispersion and sample sizes. The model explorer shows the final 80/20 holdout metrics; compact decisions use the repeated-CV comparison shown below.</div></section>
<div class="grid"><section class="chart"><h2>Validation R² by model</h2><div class="legend"><span><i class="dot res"></i>Residential</span><span><i class="dot nonres"></i>Non-residential</span></div><div id="r2chart"></div></section><section><h2>Compact-model decisions</h2><p class="muted">Repeated 5×5 CV used for the specification decision.</p><div id="compact"></div></section></div>
<section><h2>Model explorer</h2><div class="controls"><select id="modelSelect"></select></div><div id="modelDetail"></div></section>
<section><h2>All model statistics</h2><div class="controls"><button class="active" data-filter="all">All</button><button data-filter="residential">Residential</button><button data-filter="non-residential">Non-residential</button><button data-filter="compact">Compact overrides</button><input id="search" placeholder="Search area, building type, or ID"></div><div class="scroll"><table><thead><tr><th>Segment</th><th>Type</th><th>Samples</th><th>Features</th><th>Validation R²</th><th>Validation RMSE</th><th>Validation MAE</th><th>Specification</th></tr></thead><tbody id="table"></tbody></table></div></section>
<footer>Source: {html.escape(str(RUN))}/training_summary.yaml and per-model summary.yaml files. Self-contained dashboard; safe to open with VS Code Preview or directly from disk.</footer>
</main><script>const data={payload};
const models=data.models.filter(m=>m.r2_val!==null); let filter='all';
function esc(s){{return String(s).replace(/[&<>]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));}}
function chart(){{let rows=[...models].sort((a,b)=>a.r2_val-b.r2_val);let h=rows.length*23+42, w=900;let svg=`<svg viewBox="0 0 ${{w}} ${{h}}" role="img" aria-label="Validation R squared by model">`;[0,.25,.5,.75,1].forEach(x=>svg+=`<line x1="305" y1="22" x2="835" y2="${{h-12}}" stroke="#e1e8ef" transform="translate(${{x*530}},0)"/><text class="axis" x="${{303+x*530}}" y="14">${{x.toFixed(2)}}</text>`);rows.forEach((m,i)=>{{let y=28+i*23,c=m.type==='residential'?'#2e72a8':'#26866d',title=`${{m.short_id}} · ${{m.label}}`;svg+=`<text class="barlabel" x="299" y="${{y+10}}" text-anchor="end">${{esc(title)}}</text><rect x="305" y="${{y}}" width="${{Math.max(0,m.r2_val)*530}}" height="14" rx="3" fill="${{c}}"/><text class="barvalue" x="${{311+Math.max(0,m.r2_val)*530}}" y="${{y+10}}">${{m.r2_val.toFixed(3)}}</text>`}});return svg+'</svg>';}}
function compact(){{return data.compact.map(x=>`<div style="margin:16px 0;padding-bottom:14px;border-bottom:1px solid var(--line)"><strong>${{x.hedonic_id}}</strong> <span class="tag">${{x.decision}}</span><br><span class="muted">Repeated-CV R²: broad ${{x.broad_r2.toFixed(3)}} → compact ${{x.compact_r2.toFixed(3)}} · features: ${{x.broad_features}} → ${{x.compact_features}}</span><br>${{x.reason}}</div>`).join('');}}
function detail(m){{let max=Math.max(...m.top_features.map(f=>f.importance),.001);return `<div class="grid"><div><h3>${{esc(m.short_id)}}</h3><p class="muted">${{esc(m.label)}} · raw artifact: ${{esc(m.name)}}</p><p><span class="tag">${{esc(m.decision)}}</span></p><table><tbody><tr><th>Large area</th><td>${{esc(m.large_area)}}</td></tr><tr><th>Building type</th><td>${{esc(m.building_type)}}</td></tr><tr><th>Market type</th><td>${{esc(m.type)}}</td></tr><tr><th>Model</th><td>${{esc(m.model_type)}}</td></tr><tr><th>Training records</th><td>${{m.sample_size.toLocaleString()}}</td></tr><tr><th>Input features</th><td>${{m.n_features}}</td></tr><tr><th>Training R²</th><td>${{m.r2_train===null?'not available':m.r2_train.toFixed(3)}}</td></tr><tr><th>Validation R²</th><td>${{m.r2_val.toFixed(3)}}</td></tr><tr><th>Validation RMSE</th><td>${{m.rmse_val.toFixed(3)}}</td></tr><tr><th>Validation MAE</th><td>${{m.mae_val.toFixed(3)}}</td></tr></tbody></table></div><div><h3>Top relative feature importance</h3><p class="muted">Relative split-gain importance; no direction/sign.</p>${{m.top_features.map(f=>`<div class="feature"><span title="${{esc(f.name)}}">${{esc(f.label)}}</span><span class="track"><span class="fill" style="width:${{100*f.importance/max}}%"></span></span><span>${{(100*f.importance).toFixed(1)}}%</span></div>`).join('')}}</div></div>`;}}
function visible(){{let q=document.getElementById('search').value.toLowerCase();return models.filter(m=>(filter==='all'||(filter==='compact'?m.fixed_compact:m.type===filter))&&(`${{m.name}} ${{m.hedonic_id}} ${{m.label}}`).toLowerCase().includes(q));}}
function table(){{document.getElementById('table').innerHTML=visible().sort((a,b)=>a.r2_val-b.r2_val).map(m=>`<tr><td>${{esc(m.short_id)}}<br><span class="muted">${{esc(m.label)}}</span></td><td>${{esc(m.type)}}</td><td>${{m.sample_size.toLocaleString()}}</td><td>${{m.n_features}}</td><td>${{m.r2_val.toFixed(3)}}</td><td>${{m.rmse_val.toFixed(3)}}</td><td>${{m.mae_val.toFixed(3)}}</td><td>${{esc(m.decision)}}</td></tr>`).join('');}}
document.getElementById('r2chart').innerHTML=chart();document.getElementById('compact').innerHTML=compact();let select=document.getElementById('modelSelect');select.innerHTML=models.sort((a,b)=>a.name.localeCompare(b.name)).map(m=>`<option value="${{esc(m.name)}}">${{esc(m.short_id)}} — ${{esc(m.label)}}</option>`).join('');function choose(){{document.getElementById('modelDetail').innerHTML=detail(models.find(m=>m.name===select.value));}}select.value='nonres_repm14731';select.addEventListener('change',choose);choose();table();document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{{filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));table();}}));document.getElementById('search').addEventListener('input',table);
</script></body></html>'''
    out = RUN / "estimation_results_dashboard.html"
    out.write_text(page)
    print(out)


if __name__ == "__main__":
    main()
