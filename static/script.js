const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const loadingState = document.getElementById('loading');
const resultsSection = document.getElementById('results');
const resetBtn = document.getElementById('reset-btn');

// Elements to update
const tbProbBar = document.getElementById('tb-prob-bar');
const tbProbVal = document.getElementById('tb-prob-val');
const tbBadge = document.getElementById('tb-badge');

const sevBar = document.getElementById('sev-bar');
const sevVal = document.getElementById('sev-val');
const sevBadge = document.getElementById('sev-badge');

const origImg = document.getElementById('orig-img');
const heatmapImg = document.getElementById('heatmap-img');

// Drag and drop handlers
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        handleFile(files[0]);
    }
});

dropZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

resetBtn.addEventListener('click', () => {
    resultsSection.classList.add('hidden');
    dropZone.classList.remove('hidden');
    fileInput.value = '';
});

async function handleFile(file) {
    // Show loading
    dropZone.classList.add('hidden');
    loadingState.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        alert('Failed to analyze image. Ensure the backend server is running and the model weights are loaded.');
        loadingState.classList.add('hidden');
        dropZone.classList.remove('hidden');
    }
}

function displayResults(data) {
    loadingState.classList.add('hidden');
    resultsSection.classList.remove('hidden');

    // TB Probability
    const prob = data.tb_probability;
    const probPercent = (prob * 100).toFixed(1);
    tbProbVal.textContent = `${probPercent}%`;
    tbProbBar.style.width = `${probPercent}%`;
    
    if (prob > 0.5) {
        tbProbBar.style.backgroundColor = 'var(--accent-red)';
        tbBadge.textContent = 'Positive for TB';
        tbBadge.className = 'badge positive';
    } else {
        tbProbBar.style.backgroundColor = 'var(--accent-green)';
        tbBadge.textContent = 'Negative for TB';
        tbBadge.className = 'badge negative';
    }

    // Timika Severity
    // Force severity to 0 if TB is negative (like cell9_infer.py does)
    let sev = prob > 0.5 ? data.timika_severity : 0;
    sev = Math.min(Math.max(sev, 0), 140);
    
    const sevPercent = (sev / 140) * 100;
    sevVal.textContent = `${sev.toFixed(1)} / 140`;
    sevBar.style.width = `${sevPercent}%`;

    if (sev === 0) {
        sevBar.style.backgroundColor = 'var(--accent-green)';
        sevBadge.textContent = 'Healthy';
        sevBadge.className = 'badge negative';
    } else if (sev < 40) {
        sevBar.style.backgroundColor = 'var(--accent-yellow)';
        sevBadge.textContent = 'Mild';
        sevBadge.className = 'badge mild';
    } else {
        sevBar.style.backgroundColor = 'var(--accent-red)';
        sevBadge.textContent = 'Severe';
        sevBadge.className = 'badge positive';
    }

    // Images
    origImg.src = data.original_image;
    heatmapImg.src = data.heatmap_image;
}
