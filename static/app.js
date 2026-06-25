const a = document.getElementById("form");
const b = document.getElementById("out");
const c = document.getElementById("bars");
const d = document.getElementById("info");
const e = document.getElementById("err");
const f = document.getElementById("stats");
const g = document.getElementById("hist");
const h = document.getElementById("presets");

async function i() {
  const j = await fetch("/metrics");
  const k = await j.json();
  const l = await fetch("/dataset");
  const m = await l.json();
  const n = Object.entries(k.model_comparison)
    .map(([o, p]) => {
      const q = o === k.best_model ? "pill best" : "pill";
      return `<span class="${q}">${o}: ${(p.cv_mean * 100).toFixed(1)}%</span>`;
    })
    .join("");

  f.innerHTML = `
    <div class="stat"><span>original</span><strong>${m.original_count}</strong></div>
    <div class="stat"><span>expanded</span><strong>${m.expanded_count}</strong></div>
    <div class="stat"><span>test accuracy</span><strong>${(k.accuracy * 100).toFixed(1)}%</strong></div>
    <div class="stat"><span>best model</span><strong>${k.best_model.replace(/_/g, " ")}</strong></div>
  `;

  d.innerHTML = `
    <p><strong>best model:</strong> ${k.best_model}</p>
    <p><strong>tuned params:</strong> ${JSON.stringify(k.tuned_params || {})}</p>
    <p><strong>val accuracy:</strong> ${((k.val_accuracy || k.accuracy) * 100).toFixed(1)}%</p>
    <p><strong>test accuracy:</strong> ${(k.accuracy * 100).toFixed(1)}%</p>
    <p><strong>split:</strong> train ${k.train_count} / val ${k.val_count} / test ${k.test_count}</p>
    <div class="models">${n}</div>
  `;
}

async function j() {
  const k = await fetch("/history");
  const l = await k.json();
  if (!l.length) {
    g.innerHTML = '<p class="empty">no predictions yet</p>';
    return;
  }
  const m = l
    .map(
      (n) => `
    <tr>
      <td>${n.time}</td>
      <td class="sp">${n.result.species}</td>
      <td>${(n.result.confidence * 100).toFixed(1)}%</td>
      <td>${Object.values(n.input).map((o) => o.toFixed(1)).join(", ")}</td>
    </tr>
  `
    )
    .join("");
  g.innerHTML = `<table><thead><tr><th>time</th><th>species</th><th>conf</th><th>input</th></tr></thead><tbody>${m}</tbody></table>`;
}

async function k() {
  const l = await fetch("/samples");
  const m = await l.json();
  h.innerHTML = Object.entries(m)
    .map(
      ([n, o]) =>
        `<button type="button" class="preset" data-species="${n}">${n}</button>`
    )
    .join("");

  h.querySelectorAll(".preset").forEach((p) => {
    p.addEventListener("click", () => {
      const q = m[p.dataset.species];
      a.sepal_length.value = q.sepal_length;
      a.sepal_width.value = q.sepal_width;
      a.petal_length.value = q.petal_length;
      a.petal_width.value = q.petal_width;
    });
  });
}

function l(m) {
  c.innerHTML = "";
  const n = Object.entries(m).sort((o, p) => p[1] - o[1]);
  for (const [o, p] of n) {
    const q = document.createElement("div");
    q.className = "row";
    q.innerHTML = `
      <span>${o}</span>
      <div class="track"><div class="fill" style="width:${p * 100}%"></div></div>
      <span>${(p * 100).toFixed(1)}%</span>
    `;
    c.appendChild(q);
  }
}

a.addEventListener("submit", async (m) => {
  m.preventDefault();
  e.textContent = "";
  b.classList.remove("empty");

  const n = new FormData(a);
  const o = {
    sepal_length: parseFloat(n.get("sepal_length")),
    sepal_width: parseFloat(n.get("sepal_width")),
    petal_length: parseFloat(n.get("petal_length")),
    petal_width: parseFloat(n.get("petal_width")),
  };

  try {
    const p = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(o),
    });
    const q = await p.json();
    if (!p.ok) {
      e.textContent = q.error || "request failed";
      return;
    }
    b.innerHTML = `
      <span class="sp">${q.species}</span>
      <span class="cf">confidence: ${(q.confidence * 100).toFixed(1)}%</span>
    `;
    l(q.probabilities);
    j();
  } catch {
    e.textContent = "could not reach server. run python app.py first";
  }
});

i();
j();
k();
