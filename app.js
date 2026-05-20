// ===== CANVAS BACKGROUND =====
const canvas = document.getElementById("neuro-canvas");
const ctx = canvas.getContext("2d");

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

window.addEventListener("resize", () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
});

let dots = Array.from({ length: 80 }, () => ({
  x: Math.random() * canvas.width,
  y: Math.random() * canvas.height,
  vx: (Math.random() - 0.5) * 1,
  vy: (Math.random() - 0.5) * 1
}));

function animate() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  dots.forEach(d => {
    d.x += d.vx;
    d.y += d.vy;

    if (d.x < 0 || d.x > canvas.width) d.vx *= -1;
    if (d.y < 0 || d.y > canvas.height) d.vy *= -1;

    ctx.beginPath();
    ctx.arc(d.x, d.y, 2, 0, Math.PI * 2);
    ctx.fillStyle = "#00cfff";
    ctx.fill();
  });

  requestAnimationFrame(animate);
}
animate();


// ===== SIGNAL SYSTEM =====
let history = [];

function triggerSignal(intent, icon) {
  const result = document.getElementById("output-result");
  const sub = document.getElementById("output-sub");
  const hist = document.getElementById("output-history");

  result.textContent = `${icon} ${intent}`;
  sub.textContent = "Neural signal decoded successfully";

  history.unshift({
    intent,
    time: new Date().toLocaleTimeString()
  });

  if (history.length > 5) history.pop();

  hist.innerHTML = history.map(h =>
    `<div>${h.intent} - ${h.time}</div>`
  ).join("");
}


// ===== DISEASE FILTER =====
const diseases = [
  "ALS",
  "Stroke",
  "Locked-In Syndrome",
  "Parkinson's",
  "Spinal Cord Injury"
];

function render(list) {
  document.getElementById("disease-grid").innerHTML =
    list.map(d => `<div>${d}</div>`).join("");
}

render(diseases);

function filterDiseases(q) {
  render(diseases.filter(d =>
    d.toLowerCase().includes(q.toLowerCase())
  ));
}
