document.addEventListener("DOMContentLoaded", () => {
  const dropBox = document.querySelector(".border-dashed");
  const fileInput = document.getElementById("audioFile");
  const btn = document.getElementById("classifyBtn");
  const fileNameText = document.getElementById("selectedFileName");
  const audioPreview = document.getElementById("audioPreview");
  const audioPlayer = document.getElementById("audioPlayer");
  const resultContainer = document.getElementById("resultContainer");

  let selectedFile = null;

  if (!dropBox || !fileInput || !btn) {
    console.error("Required elements not found");
    return;
  }

  // Initial button state
  btn.disabled = true;
  btn.textContent = "UPLOAD FILE FIRST";
  btn.style.opacity = "0.6";
  btn.classList.remove("animate-pulse");

  function setSelectedFile(file) {
    selectedFile = file;

    if (fileNameText) {
      fileNameText.textContent = `Selected: ${file.name}`;
    }

    if (audioPlayer && audioPreview) {
      const audioURL = URL.createObjectURL(file);
      audioPlayer.src = audioURL;
      audioPreview.classList.remove("hidden");
    }

    btn.disabled = false;
    btn.textContent = "READY TO ANALYZE";
    btn.style.opacity = "1";
    btn.classList.add("animate-pulse");
  }

  dropBox.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
    }
  });

  dropBox.addEventListener("dragover", (e) => {
    e.preventDefault();
  });

  dropBox.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      setSelectedFile(file);
    }
  });

  btn.addEventListener("click", () => {
    if (!selectedFile) {
      alert("Please select an audio file first.");
      return;
    }

    uploadAndPredict(selectedFile);
  });

  async function uploadAndPredict(file) {
    btn.textContent = "Analyzing...";
    btn.disabled = true;
    btn.style.opacity = "1";
    btn.classList.remove("animate-pulse");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || "Prediction failed");
      }

      showResult(result.genre, result.scores, result.filename);
    } catch (err) {
      console.error(err);
      alert("Error: " + err.message);
    } finally {
      btn.textContent = "CLASSIFY AGAIN";
      btn.disabled = false;
      btn.style.opacity = "1";
      btn.classList.add("animate-pulse");
    }
  }

  function showResult(genre, scores, filename) {
    document.getElementById("gs-result")?.remove();

    const topScores = Object.entries(scores)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    const maxScore = Math.max(...topScores.map((s) => s[1]), 1);
    const minScore = Math.min(...topScores.map((s) => s[1]), 0);

    const barsHTML = topScores.length
      ? topScores
          .map(([g, score]) => {
            const pct = (
              ((score - minScore) / (maxScore - minScore + 0.001)) *
              100
            ).toFixed(0);

            const isTop = g.toUpperCase() === genre.toUpperCase();

            return `
              <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;">
                  <span style="text-transform:capitalize;font-weight:${isTop ? 600 : 400}">
                    ${g}
                  </span>
                  <span style="opacity:0.6">${score.toFixed(2)}</span>
                </div>
                <div style="height:6px;background:rgba(255,255,255,0.08);border-radius:3px;">
                  <div style="height:100%;width:${pct}%;border-radius:3px;background:${
                    isTop
                      ? "linear-gradient(90deg,#8ff5ff,#d575ff)"
                      : "rgba(143,245,255,0.3)"
                  }"></div>
                </div>
              </div>
            `;
          })
          .join("")
      : "<div>No confidence scores returned.</div>";

    const div = document.createElement("div");
    div.id = "gs-result";
    div.style =
      "margin-top:32px;background:rgba(15,25,48,0.8);border:1px solid rgba(143,245,255,0.15);border-radius:12px;padding:32px;";

    div.innerHTML = `
      <div style="font-size:13px;opacity:0.5;margin-bottom:24px;">${filename}</div>
      <div style="font-size:14px;text-transform:uppercase;letter-spacing:2px;color:#8ff5ff;margin-bottom:8px;">Detected Genre</div>
      <div style="font-size:52px;font-weight:700;margin-bottom:32px;">${genre}</div>
      <div style="font-size:13px;text-transform:uppercase;letter-spacing:1px;opacity:0.4;margin-bottom:16px;">Confidence Scores</div>
      ${barsHTML}
    `;

    if (resultContainer) {
      resultContainer.innerHTML = "";
      resultContainer.appendChild(div);
    }
  }
});
