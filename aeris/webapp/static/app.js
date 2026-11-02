// AERIS web UI — vanilla JS, no framework, no build step.

// ---------------- Navigation ----------------

const navItems = document.querySelectorAll(".nav-item");
navItems.forEach(btn => {
  btn.addEventListener("click", () => {
    navItems.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    const target = btn.dataset.module;
    document.querySelectorAll(".panel").forEach(p => p.classList.add("hidden"));
    document.getElementById(`panel-${target}`).classList.remove("hidden");
    if (target === "logbook") loadLogbook();
  });
});

// ---------------- Generic polling + chart helper ----------------

function drawTrace(canvas, points) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (points.length < 2) {
    ctx.fillStyle = "#7E90A0";
    ctx.font = "12px IBM Plex Mono";
    ctx.fillText("Waiting for generations...", 12, h / 2);
    return;
  }
  const scores = points.map(p => p.score).filter(s => isFinite(s));
  const minS = Math.min(...scores), maxS = Math.max(...scores);
  const pad = 18;
  const xStep = (w - 2 * pad) / Math.max(points.length - 1, 1);

  // grid
  ctx.strokeStyle = "rgba(94,201,216,0.12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + (h - 2 * pad) * (i / 4);
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
  }

  // line
  ctx.strokeStyle = "#5EC9D8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = pad + i * xStep;
    const norm = maxS === minS ? 0.5 : (p.score - minS) / (maxS - minS);
    const y = h - pad - norm * (h - 2 * pad);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // last point marker
  const last = points[points.length - 1];
  const lastNorm = maxS === minS ? 0.5 : (last.score - minS) / (maxS - minS);
  const lastX = pad + (points.length - 1) * xStep;
  const lastY = h - pad - lastNorm * (h - 2 * pad);
  ctx.fillStyle = "#F2A65A";
  ctx.beginPath(); ctx.arc(lastX, lastY, 4, 0, Math.PI * 2); ctx.fill();

  ctx.fillStyle = "#7E90A0";
  ctx.font = "11px IBM Plex Mono";
  ctx.fillText(`gen ${points.length}  best score ${last.score.toFixed(3)}`, pad, 14);
}

function drawPolar(canvas, alpha, yData, yLabel, yMin, yMax, operatingAlpha) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  
  if (!alpha || alpha.length < 2) {
    ctx.fillStyle = "#7E90A0";
    ctx.font = "12px IBM Plex Mono";
    ctx.fillText("No polar data", 12, h / 2);
    return;
  }

  const pad = 50;
  const plotW = w - 2 * pad;
  const plotH = h - 2 * pad;

  // Determine y range if not provided
  const validY = yData.filter(v => isFinite(v));
  const dataMin = yMin !== undefined ? yMin : Math.min(...validY);
  const dataMax = yMax !== undefined ? yMax : Math.max(...validY);
  const yRange = dataMax - dataMin || 1;

  // Grid
  ctx.strokeStyle = "rgba(94,201,216,0.12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + (plotH) * (i / 4);
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
    const yVal = dataMax - (dataMax - dataMin) * (i / 4);
    ctx.fillStyle = "#7E90A0";
    ctx.font = "10px IBM Plex Mono";
    ctx.fillText(yVal.toFixed(2), 4, y + 4);
  }
  for (let i = 0; i <= 5; i++) {
    const x = pad + plotH * (i / 5);
    ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, h - pad); ctx.stroke();
  }

  // Y axis label
  ctx.save();
  ctx.translate(15, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "#7E90A0";
  ctx.font = "11px IBM Plex Mono";
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();

  // X axis label
  ctx.fillStyle = "#7E90A0";
  ctx.font = "11px IBM Plex Mono";
  ctx.fillText("Alpha (deg)", w / 2 - 30, h - 10);

  // Alpha values on x-axis
  const xTicks = [-5, 0, 5, 10, 15, 20];
  for (const a of xTicks) {
    if (a >= Math.min(...alpha) && a <= Math.max(...alpha)) {
      const x = pad + ((a - Math.min(...alpha)) / (Math.max(...alpha) - Math.min(...alpha))) * plotW;
      ctx.beginPath(); ctx.moveTo(x, h - pad); ctx.lineTo(x, h - pad + 5); ctx.stroke();
      ctx.fillStyle = "#7E90A0";
      ctx.font = "10px IBM Plex Mono";
      ctx.fillText(a.toString(), x - 10, h - pad + 18);
    }
  }

  // Y-axis values
  for (let i = 0; i <= 4; i++) {
    const y = pad + plotH * (i / 4);
    const yVal = dataMax - (dataMax - dataMin) * (i / 4);
    ctx.fillStyle = "#7E90A0";
    ctx.font = "9px IBM Plex Mono";
    ctx.fillText(yVal.toFixed(3), 2, y + 3);
  }

  // Plot line
  ctx.strokeStyle = "#5EC9D8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  alpha.forEach((a, i) => {
    const x = pad + ((a - Math.min(...alpha)) / (Math.max(...alpha) - Math.min(...alpha))) * plotW;
    const y = h - pad - ((yData[i] - dataMin) / yRange) * plotH;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Operating alpha marker
  if (operatingAlpha !== undefined) {
    const xOp = pad + ((operatingAlpha - Math.min(...alpha)) / (Math.max(...alpha) - Math.min(...alpha))) * plotW;
    ctx.strokeStyle = "#F2A65A";
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(xOp, pad);
    ctx.lineTo(xOp, h - pad);
    ctx.stroke();
    ctx.setLineDash([]);
    
    ctx.fillStyle = "#F2A65A";
    ctx.font = "10px IBM Plex Mono";
    ctx.fillText(`α=${operatingAlpha}°`, xOp + 5, pad + 15);
  }

  // Title
  ctx.fillStyle = "#7E90A0";
  ctx.font = "11px IBM Plex Mono";
  ctx.fillText(yLabel, pad, 14);
}

function renderMetrics(container, metrics) {
  const skip = new Set(["_valid", "_var_names", "_validity_conditions", "polar", "baseline_polar", "constraints"]);
  const rows = Object.entries(metrics)
    .filter(([k]) => !skip.has(k))
    .map(([k, v]) => {
      const val = typeof v === "number" ? v.toFixed(4) : v;
      return `<div class="metric"><span class="k">${k}</span><span class="v">${val}</span></div>`;
    }).join("");
  container.innerHTML = rows || '<div class="metric-empty">No metrics.</div>';
}

async function pollJob(jobId, canvas, onDone) {
  const poll = async () => {
    const res = await fetch(`/api/status/${jobId}`);
    const job = await res.json();
    drawTrace(canvas, job.live_log || []);
    if (job.status === "running") {
      setTimeout(poll, 800);
    } else if (job.status === "done") {
      onDone(job);
    } else if (job.status === "error") {
      onDone(job, job.error);
    }
  };
  poll();
}

async function fetchPhysicsHealth(jobId, module) {
  try {
    const res = await fetch(`/api/physics_health/${jobId}`);
    if (res.ok) {
      const data = await res.json();
      renderPhysicsHealth(module, data);
    }
  } catch (e) {
    console.warn("Failed to fetch physics health:", e);
  }
}

function renderPhysicsHealth(module, data) {
  const badge = document.getElementById(`${module}-physics-badge`);
  const content = document.getElementById(`${module}-physics-content`);
  const panel = document.getElementById(`${module}-physics-health`);
  
  if (!panel || !badge || !content) return;
  
  panel.style.display = "block";
  
  // Set badge
  const overall = data.overall;
  badge.textContent = overall;
  badge.className = `physics-badge ${overall.toLowerCase()}`;
  
  // Render checks
  let html = "";
  for (const check of data.checks) {
    html += `
      <div class="physics-check">
        <span class="physics-check-icon ${check.status.toLowerCase()}">${check.icon}</span>
        <div>
          <div class="physics-check-text"><strong>${check.name}</strong>: ${check.message}</div>
          ${check.threshold !== undefined && check.actual !== undefined 
            ? `<div class="physics-check-detail">Threshold: ${check.threshold}, Actual: ${check.actual.toFixed(3)}</div>` 
            : ''}
        </div>
      </div>
    `;
  }
  content.innerHTML = html || '<div class="physics-check-text">No checks available</div>';
}

// ---------------- Airfoil module ----------------

document.getElementById("af-run").addEventListener("click", async () => {
  const btn = document.getElementById("af-run");
  const hint = document.getElementById("af-hint");
  btn.disabled = true;
  hint.textContent = "Running physics + optimizer...";
  document.getElementById("af-export").style.display = "none";

  const body = {
    alpha_deg: parseFloat(document.getElementById("af-alpha").value),
    velocity_mps: parseFloat(document.getElementById("af-velocity").value),
    chord_m: parseFloat(document.getElementById("af-chord").value),
    span_m: parseFloat(document.getElementById("af-span").value),
    use_biomimicry: document.getElementById("af-bio").checked,
  };
  const res = await fetch("/api/run/airfoil", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const { job_id } = await res.json();
  const canvas = document.getElementById("af-canvas");

  pollJob(job_id, canvas, (job, error) => {
    btn.disabled = false;
    if (error) { hint.textContent = `Error: ${error}`; return; }
    hint.textContent = "Done.";
    renderMetrics(document.getElementById("af-metrics"), job.metrics);
    document.getElementById("af-runid").textContent = `run ID: ${job.run_id}`;
    document.getElementById("af-export").style.display = "flex";
    document.getElementById("af-dxf").onclick = () =>
      window.location = `/api/export/${job_id}/dxf`;
    document.getElementById("af-stl").onclick = () =>
      window.location = `/api/export/${job_id}/stl`;
    fetchPhysicsHealth(job_id, "af");
    
    // Render polar plots if available
    if (job.metrics.polar) {
      const polar = job.metrics.polar;
      const opAlpha = document.getElementById("af-alpha").value;
      drawPolar(document.getElementById("af-polar-cl"), polar.alpha_deg, polar.CL, "CL", null, null, opAlpha);
      drawPolar(document.getElementById("af-polar-cd"), polar.alpha_deg, polar.CD, "CD", null, null, opAlpha);
      drawPolar(document.getElementById("af-polar-ld"), polar.alpha_deg, polar.LD, "L/D", null, null, opAlpha);
      drawPolar(document.getElementById("af-polar-cm"), polar.alpha_deg, polar.Cm_c4, "Cm", null, null, opAlpha);
    }
  });
});

// ---------------- Winglet module ----------------

document.getElementById("wl-run").addEventListener("click", async () => {
  const btn = document.getElementById("wl-run");
  const hint = document.getElementById("wl-hint");
  btn.disabled = true;
  hint.textContent = "Running physics + optimizer...";

  const body = {
    base_span_m: parseFloat(document.getElementById("wl-span").value),
    base_area_m2: parseFloat(document.getElementById("wl-area").value),
    operating_CL: parseFloat(document.getElementById("wl-cl").value),
    base_CD0: parseFloat(document.getElementById("wl-cd0").value),
  };
  const res = await fetch("/api/run/winglet", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const { job_id } = await res.json();
  const canvas = document.getElementById("wl-canvas");

  pollJob(job_id, canvas, (job, error) => {
    btn.disabled = false;
    if (error) { hint.textContent = `Error: ${error}`; return; }
    hint.textContent = "Done.";
    renderMetrics(document.getElementById("wl-metrics"), job.metrics);
    document.getElementById("wl-runid").textContent = `run ID: ${job.run_id}`;
    document.getElementById("wl-export").style.display = "flex";
    fetchPhysicsHealth(job_id, "wl");
  });
});

// ---------------- Aircraft module ----------------

document.getElementById("ac-run").addEventListener("click", async () => {
  const btn = document.getElementById("ac-run");
  const hint = document.getElementById("ac-hint");
  btn.disabled = true;
  hint.textContent = "Running physics + optimizer (10 variables, this takes a while)...";

  const body = {
    cruise_velocity_mps: parseFloat(document.getElementById("ac-velocity").value),
    target_CL: parseFloat(document.getElementById("ac-clt").value),
    altitude_m: parseFloat(document.getElementById("ac-alt").value),
  };
  const res = await fetch("/api/run/aircraft", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const { job_id } = await res.json();
  const canvas = document.getElementById("ac-canvas");

  pollJob(job_id, canvas, (job, error) => {
    btn.disabled = false;
    if (error) { hint.textContent = `Error: ${error}`; return; }
    hint.textContent = "Done.";
    renderMetrics(document.getElementById("ac-metrics"), job.metrics);
    document.getElementById("ac-runid").textContent = `run ID: ${job.run_id}`;
    document.getElementById("ac-export").style.display = "flex";
    document.getElementById("ac-stl").onclick = () =>
      window.location = `/api/export/${job_id}/stl`;
    fetchPhysicsHealth(job_id, "ac");
  });
});

// ---------------- Logbook ----------------

async function loadLogbook() {
  const res = await fetch("/api/logbook");
  const { entries } = await res.json();
  const container = document.getElementById("lb-table");
  if (!entries.length) {
    container.innerHTML = '<div class="metric-empty">No runs yet — try the other tabs first.</div>';
    return;
  }
  const rows = entries.slice().reverse().map(e => {
    const date = new Date(e.timestamp * 1000).toLocaleString();
    const statusTag = e.measured_metrics
      ? '<span class="tag ok">tested</span>'
      : '<span class="tag pending">not tested</span>';
    return `<tr><td>${e.run_id}</td><td>${e.module}</td><td>${date}</td><td>${statusTag}</td></tr>`;
  }).join("");
  container.innerHTML = `
    <table>
      <thead><tr><th>Run ID</th><th>Module</th><th>Date</th><th>Status</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

document.getElementById("lb-submit").addEventListener("click", async () => {
  const hint = document.getElementById("lb-hint");
  const run_id = document.getElementById("lb-runid").value.trim();
  if (!run_id) { hint.textContent = "Enter a run ID first."; return; }

  const measured = {};
  const cl = document.getElementById("lb-cl").value;
  const cd = document.getElementById("lb-cd").value;
  const ld = document.getElementById("lb-ld").value;
  if (cl) measured.CL = parseFloat(cl);
  if (cd) measured.CD = parseFloat(cd);
  if (ld) measured.L_D = parseFloat(ld);

  if (Object.keys(measured).length === 0) {
    hint.textContent = "Enter at least one measured value.";
    return;
  }

  const res = await fetch("/api/test_result", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id, measured }),
  });
  const data = await res.json();
  if (data.error) {
    hint.textContent = `Error: ${data.error}`;
    return;
  }
  const errText = Object.entries(data.error || {})
    .map(([k, v]) => v === null ? `${k}: n/a` : `${k}: ${(v * 100).toFixed(1)}% error`)
    .join(", ");
  hint.textContent = `Saved. ${errText}`;
  loadLogbook();
});
