"""Build a separate statistical-diagnostics dashboard for a REPM run."""
import json
from pathlib import Path

import joblib
import yaml
from repm.estimation_dashboard import load_hedonic_labels


RUN = Path("/home/da/RDF2055/d_drive/estimation/REPM/repm_20260904_102106")
CATEGORIES = ["Peer market", "Physical / site", "Land use", "Accessibility", "Employment market", "Neighborhood context", "Other"]


def category(name):
    low = name.lower()
    if low.endswith("_excl_self") or "sqft_price" in low:
        return "Peer market"
    if any(token in low for token in ("zoning_", "land_use", "future_use")):
        return "Land use"
    if low.startswith(("drv_", "walk_", "zones_logsum", "zones_transit", "jobs_drive", "fixed_route", "passenger_", "american_job")):
        return "Accessibility"
    if any(token in low for token in ("sector", "jobs", "employment", "job_space", "vacancy", "industrial")):
        return "Employment market"
    if any(token in low for token in ("crime", "race", "income", "hh", "pop", "school", "choice_rate", "ev_hybrid")):
        return "Neighborhood context"
    if any(token in low for token in ("building", "parcel", "land_", "stories", "owner_units", "residential_units", "sqft", "year_built", "age", "lot")):
        return "Physical / site"
    return "Other"


def quality_flags(row):
    flags = []
    fit_floor = 0.75 if row["type"] == "residential" else 0.40
    if row["r2_val"] < fit_floor:
        flags.append("low validation fit")
    if row["gap"] > 0.15:
        flags.append("large train–validation gap")
    if row["sample_size"] < 500:
        flags.append("small sample")
    if row["feature_ratio"] > 0.25:
        flags.append("high features/sample")
    return flags


def models():
    output = []
    hedonic_label = load_hedonic_labels()
    for path in sorted(RUN.iterdir()):
        metadata_path = path / "metadata.pkl"
        if not metadata_path.exists():
            continue
        metadata = joblib.load(metadata_path)
        metrics = metadata["metrics"]
        if not isinstance(metrics["r2_val"], (int, float)):
            continue
        importance = metadata["feature_importance"]
        total = sum(importance.values()) or 1
        groups = {key: 0.0 for key in CATEGORIES}
        for name, value in importance.items():
            groups[category(name)] += value / total
        ordered = sorted(importance.items(), key=lambda item: item[1], reverse=True)
        area, building_type, label = hedonic_label(int(metadata["hedonic_id"]))
        row = {
            "name": metadata["model_name"], "hedonic_id": metadata["hedonic_id"],
            "type": "residential" if metadata["is_residential"] else "non-residential",
            "model_type": metadata["model_type"], "sample_size": metrics["sample_size"],
            "n_features": metadata["n_features"], "r2_train": metrics["r2_train"],
            "r2_val": metrics["r2_val"], "rmse": metrics["rmse_val"], "mae": metrics["mae_val"],
            "gap": metrics["r2_train"] - metrics["r2_val"],
            "feature_ratio": metadata["n_features"] / metrics["sample_size"],
            "top1": ordered[0][1] / total if ordered else 0, "top5": sum(value for _, value in ordered[:5]) / total,
            "categories": groups, "compact": bool(metadata.get("fixed_feature_specification", False)),
            "large_area": area, "building_type": building_type, "label": label,
            "short_id": f"repm{metadata['hedonic_id']}",
        }
        row["flags"] = quality_flags(row)
        output.append(row)
    return output


def main():
    rows = models()
    payload = json.dumps({"models": rows, "categories": CATEGORIES}).replace("</", "<\\/")
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>REPM Statistical Diagnostics</title><style>
:root{{--ink:#172238;--muted:#607085;--line:#dbe4ec;--paper:#fff;--bg:#f5f8fb;--blue:#2e72a8;--green:#26866d;--orange:#c76d1a;--red:#b54444}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}}main{{max-width:1460px;margin:auto;padding:34px}}h1{{margin:0}}h2{{margin:34px 0 8px}}.sub,.muted{{color:var(--muted)}}section,.card{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}}.value{{font-size:1.7rem;font-weight:750;color:#124f79}}.label{{font-size:.85rem;color:var(--muted)}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.chart{{overflow:auto}}svg{{min-width:600px;width:100%}}.axis{{fill:var(--muted);font-size:11px}}.pt{{stroke:#fff;stroke-width:1}}table{{width:100%;border-collapse:collapse;font-size:.9rem}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right}}th{{color:var(--muted)}}td:first-child,th:first-child{{text-align:left}}.scroll{{max-height:520px;overflow:auto}}select{{padding:7px;border:1px solid var(--line);border-radius:6px;background:white;min-width:280px}}.tag{{display:inline-block;border-radius:999px;padding:2px 7px;font-size:.78rem;font-weight:650;margin:2px;background:#f9e4e4;color:#923131}}.ok{{background:#e5f3eb;color:#176047}}.stack{{display:flex;height:22px;border-radius:5px;overflow:hidden;background:#e8edf2}}.legend{{display:flex;gap:12px;flex-wrap:wrap;font-size:.82rem;color:var(--muted)}}.swatch{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:3px}}.callout{{background:#fff7e8;border-left:4px solid var(--orange);padding:12px 14px}}@media(max-width:850px){{main{{padding:18px}}.cards,.grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:600px){{.cards,.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>REPM statistical diagnostics</h1><p class="sub">Separate companion dashboard for the 2026-09-04 production candidate</p><div class="cards" id="cards"></div><section><h2>What these measures add</h2><p>Generalization gap = training R² minus validation R². Larger gaps indicate the model may not generalize as well as its training fit suggests. Feature concentration measures how much of XGBoost’s total relative importance is carried by its top one or top five features. The feature-to-sample ratio is a screening indicator, not a formal statistical test.</p><div class="callout">Flags identify models for review, not automatic rejection. Thresholds are intentionally conservative: validation R² below 0.75 residential / 0.40 non-residential, gap above 0.15, sample below 500, or features/sample above 0.25.</div></section><div class="grid"><section class="chart"><h2>Generalization gap versus sample size</h2><div id="scatter"></div></section><section><h2>Feature-category importance</h2><p class="muted">Select a model to see its full normalized XGBoost importance grouped into interpretable categories.</p><select id="select"></select><div id="categories"></div></section></div><section><h2>Review table</h2><div class="scroll"><table><thead><tr><th>Model</th><th>Sample</th><th>Features</th><th>Validation R²</th><th>Gap</th><th>Top 1</th><th>Top 5</th><th>Features/sample</th><th>Review flags</th></tr></thead><tbody id="table"></tbody></table></div></section><footer class="muted">Category grouping is an interpretive aid. XGBoost relative importance is not a coefficient, causal effect, direction of impact, or statistical significance.</footer></main><script>const data={payload};const colors={{'Peer market':'#536eaa','Physical / site':'#2e72a8','Land use':'#b07522','Accessibility':'#368b83','Employment market':'#8064a2','Neighborhood context':'#ca6b6b','Other':'#8793a1'}};
const valid=data.models;const f=x=>x.toFixed(3);const pct=x=>(100*x).toFixed(1)+'%';
function cards(){{let flagged=valid.filter(x=>x.flags.length).length;let gap=valid.reduce((s,x)=>s+x.gap,0)/valid.length;let top5=valid.reduce((s,x)=>s+x.top5,0)/valid.length;document.getElementById('cards').innerHTML=`<div class="card"><div class="value">${{valid.length}}</div><div class="label">validated models</div></div><div class="card"><div class="value">${{f(gap)}}</div><div class="label">mean generalization gap</div></div><div class="card"><div class="value">${{pct(top5)}}</div><div class="label">mean top-5 importance share</div></div><div class="card"><div class="value">${{flagged}}</div><div class="label">models with a review flag</div></div>`;}}
function scatter(){{let w=650,h=410,x0=62,y0=360,iw=545,ih=300;let vals=valid.map(x=>x.sample_size),logs=vals.map(x=>Math.log10(x));let min=Math.min(...logs),max=Math.max(...logs),gaps=valid.map(x=>x.gap),gy=Math.max(.2,...gaps);let svg=`<svg viewBox="0 0 ${{w}} ${{h}}"><text x="${{x0}}" y="25" class="axis">Each dot is one model; color denotes type.</text>`;[0,.1,.2,.3].forEach(v=>{{let y=y0-v/gy*ih;svg+=`<line x1="${{x0}}" y1="${{y}}" x2="${{x0+iw}}" y2="${{y}}" stroke="#e4ebf1"/><text x="${{x0-8}}" y="${{y+4}}" text-anchor="end" class="axis">${{v.toFixed(1)}}</text>`}});valid.forEach(m=>{{let x=x0+(Math.log10(m.sample_size)-min)/(max-min)*iw,y=y0-m.gap/gy*ih,c=m.type==='residential'?'#2e72a8':'#26866d';svg+=`<circle class="pt" cx="${{x}}" cy="${{y}}" r="5" fill="${{c}}"><title>${{m.name}}: gap ${{f(m.gap)}}, sample ${{m.sample_size}}</title></circle>`}});svg+=`<text x="${{x0+iw/2}}" y="${{h-10}}" text-anchor="middle" class="axis">sample size (log scale)</text><text x="15" y="${{y0-ih/2}}" transform="rotate(-90 15 ${{y0-ih/2}})" text-anchor="middle" class="axis">generalization gap</text></svg>`;document.getElementById('scatter').innerHTML=svg;}}
function detail(m){{let entries=data.categories.map(k=>[k,m.categories[k]]);let legend=entries.map(([k])=>`<span><i class="swatch" style="background:${{colors[k]}}"></i>${{k}}</span>`).join('');let bars=entries.filter(x=>x[1]>0.002).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`<tr><td>${{k}}</td><td><div class="stack"><span style="width:${{100*v}}%;background:${{colors[k]}}"></span></div></td><td>${{pct(v)}}</td></tr>`).join('');document.getElementById('categories').innerHTML=`<h3>${{m.short_id}}</h3><p class="muted">${{m.label}} · raw artifact: ${{m.name}}</p><p>Top-1: <strong>${{pct(m.top1)}}</strong> · Top-5: <strong>${{pct(m.top5)}}</strong> · Generalization gap: <strong>${{f(m.gap)}}</strong></p><div class="legend">${{legend}}</div><table><tbody>${{bars}}</tbody></table>`;}}
function table(){{document.getElementById('table').innerHTML=[...valid].sort((a,b)=>b.gap-a.gap).map(m=>`<tr><td>${{m.short_id}}<br><span class="muted">${{m.label}}</span>${{m.compact?' <span class="tag ok">compact</span>':''}}</td><td>${{m.sample_size.toLocaleString()}}</td><td>${{m.n_features}}</td><td>${{f(m.r2_val)}}</td><td>${{f(m.gap)}}</td><td>${{pct(m.top1)}}</td><td>${{pct(m.top5)}}</td><td>${{m.feature_ratio.toFixed(2)}}</td><td>${{m.flags.length?m.flags.map(x=>`<span class="tag">${{x}}</span>`).join(''):'<span class="tag ok">none</span>'}}</td></tr>`).join('');}}
let select=document.getElementById('select');select.innerHTML=valid.sort((a,b)=>a.name.localeCompare(b.name)).map(m=>`<option value="${{m.name}}">${{m.short_id}} — ${{m.label}}</option>`).join('');select.value='nonres_repm14731';select.onchange=()=>detail(valid.find(m=>m.name===select.value));cards();scatter();table();detail(valid.find(m=>m.name===select.value));</script></body></html>'''
    out = RUN / "estimation_statistical_diagnostics.html"
    out.write_text(page)
    print(out)


if __name__ == "__main__":
    main()
