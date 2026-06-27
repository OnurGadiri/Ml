const a = document.getElementById("form");
const b = document.getElementById("out");
const c = document.getElementById("bars");
const d = document.getElementById("info");
const e = document.getElementById("err");
const f = document.getElementById("stats");
const g = document.getElementById("hist");
const h = document.getElementById("presets");
const i = document.getElementById("clear");
const j = document.getElementById("warn");
const k = document.getElementById("btn");

const l = [
  ["sepal_length", "sepal_length_r"],
  ["sepal_width", "sepal_width_r"],
  ["petal_length", "petal_length_r"],
  ["petal_width", "petal_width_r"],
];

l.forEach(([m, n]) => {
  const o = a[m];
  const p = a[n];
  o.addEventListener("input", () => {
    p.value = o.value;
  });
  p.addEventListener("input", () => {
    o.value = p.value;
  });
});

async function m() {
  const n = await fetch("/metrics");
  const o = await n.json();
  const p = await fetch("/dataset");
  const q = await p.json();
  const r = await fetch("/ranges");
  const s = await r.json();

  document.querySelectorAll(".hint").forEach((t) => {
    const u = t.dataset.key;
    if (s[u]) {
      t.textContent = `range: ${s[u].min} - ${s[u].max} (mean ${s[u].mean})`;
      const v = a[u];
      const w = a[`${u}_r`];
      v.min = s[u].min;
      v.max = s[u].max;
      w.min = s[u].min;
      w.max = s[u].max;
    }
  });

  const x = Object.entries(o.model_comparison)
    .map(([y, z]) => {
      const aa = y === o.best_model ? "pill best" : "pill";
      return `<span class="${aa}">${y}: ${(z.cv_mean * 100).toFixed(1)}%</span>`;
    })
    .join("");

  f.innerHTML = `
    <div class="stat"><span>original</span><strong>${q.original_count}</strong></div>
    <div class="stat"><span>expanded</span><strong>${q.expanded_count}</strong></div>
    <div class="stat"><span>test accuracy</span><strong>${(o.accuracy * 100).toFixed(1)}%</strong></div>
    <div class="stat"><span>best model</span><strong>${o.best_model.replace(/_/g, " ")}</strong></div>
  `;

  d.innerHTML = `
    <p><strong>best model:</strong> ${o.best_model}</p>
    <p><strong>tuned params:</strong> ${JSON.stringify(o.tuned_params || {})}</p>
    <p><strong>val accuracy:</strong> ${((o.val_accuracy || o.accuracy) * 100).toFixed(1)}%</p>
    <p><strong>test accuracy:</strong> ${(o.accuracy * 100).toFixed(1)}%</p>
    <p><strong>split:</strong> train ${o.train_count} / val ${o.val_count} / test ${o.test_count}</p>
    <div class="models">${x}</div>
  `;
}

async function n() {
  const o = await fetch("/history");
  const p = await o.json();
  if (!p.length) {
    g.innerHTML = '<p class="empty">no predictions yet</p>';
    return;
  }
  const q = p
    .map(
      (r) => `
    <tr>
      <td>${r.time}</td>
      <td class="sp">${r.result.species}</td>
      <td>${(r.result.confidence * 100).toFixed(1)}%</td>
      <td>${Object.values(r.input).map((s) => s.toFixed(1)).join(", ")}</td>
    </tr>
  `
    )
    .join("");
  g.innerHTML = `<table><thead><tr><th>time</th><th>species</th><th>conf</th><th>input</th></tr></thead><tbody>${q}</tbody></table>`;
}

async function o() {
  const p = await fetch("/samples");
  const q = await p.json();
  h.innerHTML = Object.entries(q)
    .map(([r, s]) => `<button type="button" class="preset" data-species="${r}">${r}</button>`)
    .join("");

  h.querySelectorAll(".preset").forEach((t) => {
    t.addEventListener("click", () => {
      const u = q[t.dataset.species];
      l.forEach(([v]) => {
        a[v].value = u[v];
        a[`${v}_r`].value = u[v];
      });
    });
  });
}

function p(q) {
  c.innerHTML = "";
  const r = Object.entries(q).sort((s, t) => t[1] - s[1]);
  for (const [s, t] of r) {
    const u = document.createElement("div");
    u.className = "row";
    u.innerHTML = `
      <span>${s}</span>
      <div class="track"><div class="fill" style="width:${t * 100}%"></div></div>
      <span>${(t * 100).toFixed(1)}%</span>
    `;
    c.appendChild(u);
  }
}

a.addEventListener("submit", async (q) => {
  q.preventDefault();
  e.textContent = "";
  j.textContent = "";
  b.classList.remove("empty");
  k.disabled = true;
  k.textContent = "predicting...";

  const r = new FormData(a);
  const s = {
    sepal_length: parseFloat(r.get("sepal_length")),
    sepal_width: parseFloat(r.get("sepal_width")),
    petal_length: parseFloat(r.get("petal_length")),
    petal_width: parseFloat(r.get("petal_width")),
  };

  try {
    const t = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(s),
    });
    const u = await t.json();
    if (!t.ok) {
      e.textContent = u.error || "request failed";
      return;
    }
    b.innerHTML = `
      <span class="sp">${u.species}</span>
      <span class="cf">confidence: ${(u.confidence * 100).toFixed(1)}%</span>
    `;
    if (u.warnings && u.warnings.length) {
      j.textContent = u.warnings.join(" | ");
    }
    p(u.probabilities);
    n();
  } catch {
    e.textContent = "could not reach server. run python app.py first";
  } finally {
    k.disabled = false;
    k.textContent = "predict";
  }
});

i.addEventListener("click", async () => {
  await fetch("/history", { method: "DELETE" });
  n();
});

m();
n();
o();
