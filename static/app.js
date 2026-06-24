const a = document.getElementById("form");
const b = document.getElementById("out");
const c = document.getElementById("bars");
const d = document.getElementById("info");
const e = document.getElementById("err");
const f = document.getElementById("chart");

async function g() {
  const h = await fetch("/metrics");
  const i = await h.json();
  const j = Object.entries(i.model_comparison)
    .map(([k, l]) => {
      const m = k === i.best_model ? "pill best" : "pill";
      return `<span class="${m}">${k}: ${(l.cv_mean * 100).toFixed(1)}%</span>`;
    })
    .join("");
  d.innerHTML = `
    <p><strong>best model:</strong> ${i.best_model}</p>
    <p><strong>test accuracy:</strong> ${(i.accuracy * 100).toFixed(1)}%</p>
    <p><strong>samples:</strong> ${i.sample_count}</p>
    <p><strong>classes:</strong> ${i.classes.join(", ")}</p>
    <div class="models">${j}</div>
  `;
}

function h(i) {
  c.innerHTML = "";
  const j = Object.entries(i).sort((k, l) => l[1] - k[1]);
  for (const [k, l] of j) {
    const m = document.createElement("div");
    m.className = "row";
    m.innerHTML = `
      <span>${k}</span>
      <div class="track"><div class="fill" style="width:${l * 100}%"></div></div>
      <span>${(l * 100).toFixed(1)}%</span>
    `;
    c.appendChild(m);
  }
}

a.addEventListener("submit", async (i) => {
  i.preventDefault();
  e.textContent = "";
  b.classList.remove("empty");

  const j = new FormData(a);
  const k = {
    sepal_length: parseFloat(j.get("sepal_length")),
    sepal_width: parseFloat(j.get("sepal_width")),
    petal_length: parseFloat(j.get("petal_length")),
    petal_width: parseFloat(j.get("petal_width")),
  };

  try {
    const l = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(k),
    });
    const m = await l.json();
    if (!l.ok) {
      e.textContent = m.error || "request failed";
      return;
    }
    b.innerHTML = `
      <span class="sp">${m.species}</span>
      <span class="cf">confidence: ${(m.confidence * 100).toFixed(1)}%</span>
    `;
    h(m.probabilities);
  } catch {
    e.textContent = "could not reach server. run python app.py first";
  }
});

f.onerror = () => {
  f.style.display = "none";
};

g();
