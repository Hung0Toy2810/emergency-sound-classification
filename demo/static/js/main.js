/**
 * SoundClassify — Frontend Logic (Enhanced)
 * Upload audio, fetch predictions from both models, render results.
 * IntersectionObserver for scroll animations, active nav tracking.
 */

// ── Class metadata ──────────────────────────────────────────────────
const CLASS_INFO = {
    'Crash_Collision':   { emoji: '💥', label: 'Crash / Collision' },
    'Siren':             { emoji: '🚨', label: 'Siren' },
    'Scream_Shout':      { emoji: '😱', label: 'Scream / Shout' },
    'Gunshot_Explosion': { emoji: '💣', label: 'Gunshot / Explosion' },
};

// ── DOM Elements ────────────────────────────────────────────────────
const dropZone       = document.getElementById('dropZone');
const audioInput     = document.getElementById('audioInput');
const uploadContent  = document.getElementById('uploadContent');
const fileInfo       = document.getElementById('fileInfo');
const fileName       = document.getElementById('fileName');
const fileSize       = document.getElementById('fileSize');
const fileRemove     = document.getElementById('fileRemove');
const audioPlayerWrap= document.getElementById('audioPlayerWrap');
const audioElement   = document.getElementById('audioElement');
const predictBtn     = document.getElementById('predictBtn');
const loading        = document.getElementById('loading');
const results        = document.getElementById('results');
const errorMsg       = document.getElementById('errorMsg');
const errorText      = document.getElementById('errorText');

let selectedFile = null;
let loadingInterval = null;

// ── Drag & Drop ─────────────────────────────────────────────────────
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
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

audioInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

fileRemove.addEventListener('click', (e) => {
    e.stopPropagation();
    clearSelection();
});

predictBtn.addEventListener('click', runPrediction);

// ── File Selection ──────────────────────────────────────────────────
function handleFileSelect(file) {
    if (!file.type.startsWith('audio/') && !isAudioExtension(file.name)) {
        showError('Vui lòng chọn file audio (.wav, .mp3, .flac, .ogg, .m4a)');
        return;
    }

    selectedFile = file;

    // Show file info
    uploadContent.classList.add('hidden');
    fileInfo.classList.remove('hidden');
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);

    // Show audio player
    const url = URL.createObjectURL(file);
    audioElement.src = url;
    audioPlayerWrap.classList.remove('hidden');

    // Show predict button
    predictBtn.classList.remove('hidden');

    // Hide previous results
    results.classList.add('hidden');
    errorMsg.classList.add('hidden');
}

function clearSelection() {
    selectedFile = null;
    audioInput.value = '';
    uploadContent.classList.remove('hidden');
    fileInfo.classList.add('hidden');
    audioPlayerWrap.classList.add('hidden');
    predictBtn.classList.add('hidden');
    results.classList.add('hidden');
    errorMsg.classList.add('hidden');
    audioElement.src = '';
}

function isAudioExtension(name) {
    const ext = name.split('.').pop().toLowerCase();
    return ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'wma'].includes(ext);
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

// ── Prediction ──────────────────────────────────────────────────────
async function runPrediction() {
    if (!selectedFile) return;

    // Show loading, hide results
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    errorMsg.classList.add('hidden');
    predictBtn.disabled = true;

    // Animate loading steps
    animateLoadingSteps();

    const formData = new FormData();
    formData.append('audio', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Đã xảy ra lỗi');
        }

        // Hide loading, show results
        loading.classList.add('hidden');
        results.classList.remove('hidden');

        renderResults(data);

    } catch (err) {
        loading.classList.add('hidden');
        showError(err.message);
    } finally {
        predictBtn.disabled = false;
        if (loadingInterval) {
            clearInterval(loadingInterval);
            loadingInterval = null;
        }
    }
}

function animateLoadingSteps() {
    if (loadingInterval) {
        clearInterval(loadingInterval);
    }

    const steps = ['stepPreprocess', 'stepCnn', 'stepAst', 'stepDone'];
    steps.forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove('active', 'done');
    });
    document.getElementById('stepPreprocess').classList.add('active');

    let i = 0;
    loadingInterval = setInterval(() => {
        if (i > 0) {
            document.getElementById(steps[i - 1]).classList.remove('active');
            document.getElementById(steps[i - 1]).classList.add('done');
        }
        if (i < steps.length) {
            document.getElementById(steps[i]).classList.add('active');
        } else {
            clearInterval(loadingInterval);
            loadingInterval = null;
        }
        i++;
    }, 800);
}

// ── Render Results ──────────────────────────────────────────────────
function renderResults(data) {
    // Spectrogram
    if (data.spectrogram) {
        document.getElementById('spectrogramImg').src = 'data:image/png;base64,' + data.spectrogram;
        document.getElementById('spectrogramWrap').classList.remove('hidden');
    }

    // CNN Result
    renderModelResult('cnn', data.cnn);

    // AST Result
    renderModelResult('ast', data.ast);

    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderModelResult(prefix, result) {
    if (!result) return;

    if (result.available === false) {
        document.getElementById(`${prefix}PredictedClass`).textContent = 'Model unavailable';
        document.getElementById(`${prefix}PredictedConf`).textContent = '—';
        document.getElementById(`${prefix}InferenceTime`).textContent = '— ms';

        const barsContainer = document.getElementById(`${prefix}ConfBars`);
        barsContainer.innerHTML = `
            <div class="model-error">
                ${result.error || 'Không thể load model này trong phiên hiện tại.'}
            </div>
        `;
        return;
    }

    // Predicted class
    const classInfo = CLASS_INFO[result.predicted_class] || { emoji: '❓', label: result.predicted_class };
    document.getElementById(`${prefix}PredictedClass`).textContent =
        classInfo.emoji + ' ' + classInfo.label;
    document.getElementById(`${prefix}PredictedConf`).textContent =
        result.confidence.toFixed(1) + '%';
    document.getElementById(`${prefix}InferenceTime`).textContent =
        result.inference_ms.toFixed(0) + ' ms';

    // Confidence bars
    const barsContainer = document.getElementById(`${prefix}ConfBars`);
    barsContainer.innerHTML = '';

    // Sort by probability (descending)
    const entries = Object.entries(result.probabilities)
        .sort((a, b) => b[1] - a[1]);

    const topClass = entries[0][0];

    entries.forEach(([cls, pct]) => {
        const info = CLASS_INFO[cls] || { emoji: '❓', label: cls };
        const isTop = (cls === topClass);

        const item = document.createElement('div');
        item.className = 'conf-bar-item';
        item.innerHTML = `
            <div class="conf-bar-header">
                <span class="conf-bar-name">${info.emoji} ${info.label}</span>
                <span class="conf-bar-pct">${pct.toFixed(1)}%</span>
            </div>
            <div class="conf-bar-track">
                <div class="conf-bar-fill ${isTop ? 'is-top' : ''}"
                     style="width: 0;" data-width="${Math.max(pct, 1)}%"></div>
            </div>
        `;
        barsContainer.appendChild(item);
    });

    // Animate bars after a short delay
    requestAnimationFrame(() => {
        setTimeout(() => {
            barsContainer.querySelectorAll('.conf-bar-fill').forEach(bar => {
                bar.style.width = bar.dataset.width;
            });
        }, 100);
    });
}

async function fetchHealth() {
    try {
        const response = await fetch('/health');
        if (!response.ok) return;
        const data = await response.json();
        const issues = Object.entries(data.models || {})
            .filter(([, status]) => !status.ready)
            .map(([name, status]) => `${name.toUpperCase()}: ${status.error || 'unavailable'}`);

        if (issues.length > 0) {
            showError('Một số model chưa sẵn sàng: ' + issues.join(' | '));
        }
    } catch (_) {
        // Silent fallback for demo.
    }
}

// ── Error Handling ──────────────────────────────────────────────────
function showError(msg) {
    errorText.textContent = msg;
    errorMsg.classList.remove('hidden');
}

// ═══════════════════════════════════════════════════════════════════
// INTERSECTION OBSERVER — Scroll Reveal & Bar Animations
// ═══════════════════════════════════════════════════════════════════

// Reveal elements on scroll
const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            revealObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

// Animate metric bars when visible
const barObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.querySelectorAll('.bar-fill:not(.animated)').forEach(bar => {
                bar.classList.add('animated');
            });
            barObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.2 });

const classBarsEl = document.getElementById('classBars');
if (classBarsEl) barObserver.observe(classBarsEl);

// Animate big numbers when visible
const numberObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const el = entry.target;
            if (el.dataset.animated) return;
            el.dataset.animated = 'true';

            const target = parseFloat(el.dataset.value);
            const duration = 1500;
            const start = performance.now();

            function tick(now) {
                const progress = Math.min((now - start) / duration, 1);
                const ease = 1 - Math.pow(1 - progress, 3); // easeOutCubic
                const current = (target * ease).toFixed(target % 1 !== 0 ? 2 : 0);
                el.textContent = current;
                if (progress < 1) requestAnimationFrame(tick);
            }
            requestAnimationFrame(tick);
            numberObserver.unobserve(el);
        }
    });
}, { threshold: 0.5 });

document.querySelectorAll('.big-number[data-value]').forEach(el => numberObserver.observe(el));

// ═══════════════════════════════════════════════════════════════════
// ACTIVE NAV LINK HIGHLIGHTING
// ═══════════════════════════════════════════════════════════════════

const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('section[id]');

const navObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const id = entry.target.id;
            navLinks.forEach(link => {
                link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
            });
        }
    });
}, { threshold: 0.3, rootMargin: '-80px 0px -40% 0px' });

sections.forEach(section => navObserver.observe(section));

// ── Init ─────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
    fetchHealth();
});
