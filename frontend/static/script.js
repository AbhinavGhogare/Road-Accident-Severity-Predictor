const sections = ["dashboard","predictor","analytics","about"];

function showSection(id) {
    sections.forEach(s => {
        document.getElementById(s).classList.toggle("active-section", s === id);
    });
    document.querySelectorAll(".nav-item").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.section === id);
    });
    const titles = {
        dashboard: "Dashboard Overview",
        predictor: "Accident Prediction",
        analytics: "Dataset Analytics",
        about: "About the Model"
    };
    document.getElementById("pageTitle").textContent = titles[id];
    window.scrollTo({top:0, behavior:"smooth"});
}

document.querySelectorAll(".nav-item").forEach(btn => {
    btn.addEventListener("click", () => showSection(btn.dataset.section));
});

function renderBars(containerId, values) {
    const el = document.getElementById(containerId);
    el.innerHTML = "";
    const entries = Object.entries(values || {});
    const max = Math.max(...entries.map(x => x[1]), 1);
    entries.forEach(([label, value]) => {
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
            <span class="bar-label">${label}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${(value/max)*100}%"></div></div>
            <span class="bar-value">${value}</span>`;
        el.appendChild(row);
    });
}

async function loadDashboard() {
    try {
        const [statusRes, statsRes] = await Promise.all([
            fetch("/api/status"),
            fetch("/api/stats")
        ]);
        const status = await statusRes.json();
        const stats = await statsRes.json();

        if (!statusRes.ok) throw new Error(status.error || "Model failed to start");

        document.getElementById("statusDot").style.background = "#18a66a";
        document.getElementById("statusText").textContent = "Model ready";
        document.getElementById("totalRecords").textContent = status.rows.toLocaleString();
        document.getElementById("accuracy").textContent = status.accuracy + "%";
        document.getElementById("avgSpeed").textContent = stats.avg_speed + " km/h";
        document.getElementById("avgVisibility").textContent = stats.avg_visibility + " km";

        document.getElementById("avgVehicles").textContent = stats.avg_vehicles;
        document.getElementById("avgSpeed2").textContent = stats.avg_speed + " km/h";
        document.getElementById("avgVisibility2").textContent = stats.avg_visibility + " km";

        renderBars("severityBars", stats.severity);
        renderBars("weatherBars", stats.weather);
    } catch (err) {
        document.getElementById("statusDot").style.background = "#d95d5d";
        document.getElementById("statusText").textContent = "Model unavailable";
        document.getElementById("formError").textContent = err.message;
    }
}

function setSampleCase() {
    const f = document.getElementById("predictionForm");
    f.Weather.value = "Rain";
    f.Road_Type.value = "Highway";
    f.Light_Condition.value = "Night";
    f.Road_Surface.value = "Wet";
    f.Speed_kmph.value = 85;
    f.Vehicles_Involved.value = 3;
    f.Visibility_km.value = 3;
    f.Traffic_Density.value = "High";
    f.Alcohol_Involvement.value = "Yes";
}

document.getElementById("sampleBtn").addEventListener("click", setSampleCase);

document.getElementById("predictionForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const payload = Object.fromEntries(new FormData(form).entries());
    payload.Speed_kmph = Number(payload.Speed_kmph);
    payload.Vehicles_Involved = Number(payload.Vehicles_Involved);
    payload.Visibility_km = Number(payload.Visibility_km);

    const error = document.getElementById("formError");
    error.textContent = "";

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: {"Content-Type":"application/json"},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Prediction failed");

        document.querySelector(".result-empty").classList.add("hidden");
        document.getElementById("resultContent").classList.remove("hidden");

        const result = document.getElementById("severityResult");
        result.textContent = data.prediction;
        result.className = "severity-result " + data.prediction.toLowerCase();

        const list = document.getElementById("probabilityList");
        list.innerHTML = "";

        Object.entries(data.probabilities).forEach(([label, probability]) => {
            const row = document.createElement("div");
            row.className = "prob-row";
            row.innerHTML = `
                <div class="prob-top"><span>${label}</span><span>${probability}%</span></div>
                <div class="prob-track"><div class="prob-fill" style="width:${probability}%"></div></div>`;
            list.appendChild(row);
        });
    } catch (err) {
        error.textContent = err.message;
    }
});

loadDashboard();
