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
const l = document.getElementById("ring");
const m = document.getElementById("compare");
const n = document.querySelectorAll(".tab");
const o = document.querySelectorAll(".panel");

const p = {
  setosa: "#3ecf8e",
  versicolor: "#5b9fd4",
  virginica: "#e6c07b",
};

const q = [
  ["sepal_length", "sepal_length_r"],
  ["sepal_width", "sepal_width_r"],
  ["petal_length", "petal_length_r"],
  ["petal_width", "petal_width_r"],
];

q.forEach(([r, s]) => {
  const t = a[r];
  const u = a[s];
  t.addEventListener("input", () => {
    u.value = t.value;
  });
  u.addEventListener("input", () => {
    t.value = u.value;
  });
});

n.forEach((v) => {
  v.addEventListener("click", () => {
    n.forEach((w) => w.classList.remove("on"));
    o.forEach((w) => w.classList.remove("on"));
    v.classList.add("on");
    document.getElementById(`${v.dataset.tab}-panel`).classList.add("on");
  });
});

async function r() {
  const s = await fetch("/metrics");
  const t = await s.json();
  const u = await fetch("/dataset");
  const v = await u.json();
  const w = await fetch("/ranges");
  const x = await w.json();

  document.querySelectorAll(".hint").forEach((y) => {
    const z = y.dataset.key;
    if (x[z]) {
      y.textContent = `range: ${x[z].min} - ${x[z].max} (mean ${x[z].mean})`;
      const aa = a[z];
      const ab = a[`${z}_r`];
      aa.min = x[z].min;
      aa.max = x[z].max;
      ab.min = x[z].min;
      ab.max = x[z].max;
    }
  });

  const ac = Object.entries(t.model_comparison)
    .map(([ad, ae]) => {
      const af = ad === t.best_model ? "pill best" : "pill";
      return `<span class="${af}">${ad}: ${(ae.cv_mean * 100).toFixed(1)}%</span>`;
    })
    .join("");

  f.innerHTML = `
    <div class="stat"><span>original</span><strong>${v.original_count}</strong></div>
    <div class="stat"><span>expanded</span><strong>${v.expanded_count}</strong></div>
    <div class="stat"><span>f1 macro</span><strong>${((t.f1_macro || 0) * 100).toFixed(1)}%</strong></div>
    <div class="stat"><span>test accuracy</span><strong>${(t.accuracy * 100).toFixed(1)}%</strong></div>
  `;

  d.innerHTML = `
    <p><strong>best model:</strong> ${t.best_model}</p>
    <p><strong>trained at:</strong> ${t.trained_at || "n/a"}</p>
    <p><strong>tuned params:</strong> ${JSON.stringify(t.tuned_params || {})}</p>
    <p><strong>val accuracy:</strong> ${((t.val_accuracy || t.accuracy) * 100).toFixed(1)}%</p>
    <p><strong>test accuracy:</strong> ${(t.accuracy * 100).toFixed(1)}%</p>
    <p><strong>split:</strong> train ${t.train_count} / val ${t.val_count} / test ${t.test_count}</p>
    <div class="models">${ac}</div>
  `;

  try {
    const ag = await fetch("/compare");
    const ah = await ag.json();
    m.innerHTML = `
      <table>
        <tr><th>dataset</th><th>samples</th><th>accuracy</th></tr>
        <tr><td>original only</td><td>${ah.original_only.samples}</td><td>${(ah.original_only.test_accuracy * 100).toFixed(1)}%</td></tr>
        <tr><td>expanded</td><td>${ah.expanded.samples}</td><td>${(ah.expanded.test_accuracy * 100).toFixed(1)}%</td></tr>
      </table>
    `;
  } catch {
    m.innerHTML = '<p class="empty">run compare.py first</p>';
  }
}

async function s() {
  const t = await fetch("/history");
  const u = await t.json();
  if (!u.length) {
    g.innerHTML = '<p class="empty">no predictions yet</p>';
    return;
  }
  const v = u
    .map(
      (w) => `
    <tr>
      <td>${w.time}</td>
      <td class="sp ${w.result.species}">${w.result.species}</td>
      <td>${(w.result.confidence * 100).toFixed(1)}%</td>
      <td>${Object.values(w.input).map((x) => x.toFixed(1)).join(", ")}</td>
    </tr>
  `
    )
    .join("");
  g.innerHTML = `<table><thead><tr><th>time</th><th>species</th><th>conf</th><th>input</th></tr></thead><tbody>${v}</tbody></table>`;
}

async function t() {
  const u = await fetch("/samples");
  const v = await u.json();
  h.innerHTML = Object.entries(v)
    .map(([w]) => `<button type="button" class="preset ${w}" data-species="${w}">${w}</button>`)
    .join("");

  h.querySelectorAll(".preset").forEach((x) => {
    x.addEventListener("click", () => {
      const y = v[x.dataset.species];
      q.forEach(([z]) => {
        a[z].value = y[z];
        a[`${z}_r`].value = y[z];
      });
    });
  });
}

function u(v, w) {
  const x = p[w] || "#3ecf8e";
  l.innerHTML = `
    <svg viewBox="0 0 36 36" class="ring-svg">
      <path class="ring-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
      <path class="ring-fill" stroke="${x}" stroke-dasharray="${v * 100}, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
      <text x="18" y="20.35" class="ring-txt">${(v * 100).toFixed(0)}%</text>
    </svg>
  `;
}

function v(w) {
  c.innerHTML = "";
  const x = Object.entries(w).sort((y, z) => z[1] - y[1]);
  for (const [y, z] of x) {
    const aa = document.createElement("div");
    aa.className = "row";
    const ab = p[y] || "#3ecf8e";
    aa.innerHTML = `
      <span class="${y}">${y}</span>
      <div class="track"><div class="fill" style="width:${z * 100}%;background:${ab}"></div></div>
      <span>${(z * 100).toFixed(1)}%</span>
    `;
    c.appendChild(aa);
  }
}

a.addEventListener("submit", async (w) => {
  w.preventDefault();
  e.textContent = "";
  j.textContent = "";
  b.classList.remove("empty");
  k.disabled = true;
  k.textContent = "predicting...";

  const x = new FormData(a);
  const y = {
    sepal_length: parseFloat(x.get("sepal_length")),
    sepal_width: parseFloat(x.get("sepal_width")),
    petal_length: parseFloat(x.get("petal_length")),
    petal_width: parseFloat(x.get("petal_width")),
  };

  try {
    const z = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(y),
    });
    const aa = await z.json();
    if (!z.ok) {
      e.textContent = aa.error || "request failed";
      return;
    }
    const ab = p[aa.species] || "#3ecf8e";
    b.innerHTML = `
      <span class="sp ${aa.species}" style="color:${ab}">${aa.species}</span>
      <span class="cf">confidence: ${(aa.confidence * 100).toFixed(1)}%</span>
    `;
    if (aa.warnings && aa.warnings.length) {
      j.textContent = aa.warnings.join(" | ");
    }
    u(aa.confidence, aa.species);
    v(aa.probabilities);
    s();
  } catch {
    e.textContent = "could not reach server. run python app.py first";
  } finally {
    k.disabled = false;
    k.textContent = "predict";
  }
});

i.addEventListener("click", async () => {
  await fetch("/history", { method: "DELETE" });
  s();
});

r();
s();
t();
