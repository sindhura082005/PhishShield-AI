// annotate suspicious links on pages (best-effort sampling)
const API = "http://127.0.0.1:8000/predict";

async function check(url) {
  try {
    const r = await fetch(API, { method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify({ url }) });
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

async function annotate() {
  const as = [...document.querySelectorAll("a[href]")].slice(0, 50);
  for (const a of as) {
    const res = await check(a.href);
    if (res && (res.label === "phish" || (res.score||0) >= 0.6)) {
      a.style.outline = "2px dashed #d9534f";
      a.title = `Suspicious (score=${(res.score||0).toFixed(2)})`;
    }
  }
}

if (document.readyState !== "loading") annotate();
else document.addEventListener("DOMContentLoaded", annotate);
