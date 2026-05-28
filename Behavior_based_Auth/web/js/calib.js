class CalibrationManager {
    constructor() {
        this.currentSection = 'welcome';
        this.currentPassage = 0;
        this.currentExercise = 0;
        this.typingStartTime = null;
        this.mouseExerciseStartTime = null;
        this.keystrokeData = [];
        this.mouseData = [];
        this.typingStats = { charactersTyped: 0, wpm: 0, accuracy: 100, samples: 0 };
        this.mouseStats = { distance: 0, clicks: 0, velocity: 0, exercises: 0 };

        this.typingPassages = [
            "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is commonly used for typing practice. It helps develop muscle memory and finger coordination while maintaining proper typing posture.",
            "Behavioral biometrics represents a fascinating frontier in cybersecurity technology. Unlike traditional authentication methods that rely on what you know or what you have, behavioral biometrics focuses on what you do and how you do it.",
            "Machine learning algorithms can detect subtle patterns in human behavior that are virtually impossible to replicate. These patterns include typing rhythm, mouse movement trajectories, and even the way someone holds their mobile device.",
            "Continuous authentication provides a significant advantage over one-time password systems. By constantly monitoring user behavior, it can detect unauthorized access attempts in real-time, even after initial login verification has been completed successfully.",
            "The future of cybersecurity lies in adaptive systems that learn and evolve with user behavior. These intelligent systems can distinguish between genuine changes in behavior patterns and potential security threats."
        ];

        this.mouseExercises = [
            { name: "Click Timing Exercise", description: "Click the targets as they appear", duration: 30, type: "targets" },
            { name: "Tracking Exercise", description: "Follow the moving target with your cursor", duration: 25, type: "tracking" },
            { name: "Navigation Exercise", description: "Navigate through the maze path", duration: 35, type: "navigation" },
            { name: "Precision Exercise", description: "Click precisely on small targets", duration: 20, type: "precision" }
        ];

        this.initializeElements();
        this.setupEventListeners();
    }

    initializeElements() {
        this.progressSteps = document.querySelectorAll('.step');
        this.overallProgress = document.getElementById('overallProgress');
        this.progressText = document.getElementById('progressText');
        this.sections = document.querySelectorAll('.calibration-section');
        this.welcomeSection = document.getElementById('welcomeSection');
        this.typingSection = document.getElementById('typingSection');
        this.mouseSection = document.getElementById('mouseSection');
        this.completionSection = document.getElementById('completionSection');
        this.passageText = document.getElementById('passageText');
        this.typingArea = document.getElementById('typingArea');
        this.passageNumber = document.getElementById('passageNumber');
        this.passageProgress = document.getElementById('passageProgress');
        this.typingFeedback = document.getElementById('typingFeedback');
        this.mousePlayground = document.getElementById('mousePlayground');
        this.exerciseName = document.getElementById('exerciseName');
        this.exerciseTimer = document.getElementById('exerciseTimer');
        this.playgroundInstructions = document.getElementById('playgroundInstructions');
        this.startCalibration = document.getElementById('startCalibration');
        this.skipPassage = document.getElementById('skipPassage');
        this.nextToMouse = document.getElementById('nextToMouse');
        this.startMouseExercise = document.getElementById('startMouseExercise');
        this.nextMouseExerciseBtn = document.getElementById('nextMouseExercise');
        this.completeCalibrationBtn = document.getElementById('completeCalibration');
        this.continueToChallenge = document.getElementById('continueToChallenge');
        this.typingProgressEl = document.getElementById('typingProgress');
        this.typingWPMEl = document.getElementById('typingWPM');
        this.typingAccuracyEl = document.getElementById('typingAccuracy');
        this.samplesCollectedEl = document.getElementById('samplesCollected');
        this.mouseDistanceEl = document.getElementById('mouseDistance');
        this.mouseClicksEl = document.getElementById('mouseClicks');
        this.mouseVelocityEl = document.getElementById('mouseVelocity');
        this.mouseExercisesEl = document.getElementById('mouseExercises');
        this.statusOverlay = document.getElementById('statusOverlay');
        this.statusTitle = document.getElementById('statusTitle');
        this.statusMessage = document.getElementById('statusMessage');
    }

    setupEventListeners() {
        this.startCalibration.addEventListener('click', () => this.startTypingCalibration());
        this.skipPassage.addEventListener('click', () => this.nextPassage());
        this.nextToMouse.addEventListener('click', () => this.startMouseCalibration());
        this.startMouseExercise.addEventListener('click', () => this.startCurrentMouseExercise());
        this.nextMouseExerciseBtn.addEventListener('click', () => this.nextMouseExercise());
        this.completeCalibrationBtn.addEventListener('click', () => this.completeCalibration());
        this.continueToChallenge.addEventListener('click', () => { window.location.href = 'challenge.html'; });

        this.typingArea.addEventListener('input', (e) => this.handleTypingInput(e));
        this.typingArea.addEventListener('keydown', (e) => this.captureKeystroke(e));
        this.typingArea.addEventListener('keyup', (e) => this.captureKeystroke(e));

        document.addEventListener('mousemove', (e) => this.captureMouseMovement(e));
        document.addEventListener('mousedown', (e) => this.captureMouseClick(e));
        document.addEventListener('mouseup', (e) => this.captureMouseClick(e));
    }

    startTypingCalibration() {
        this.currentSection = 'typing';
        this.showSection('typingSection');
        this.updateProgress(25);
        this.updateProgressSteps(2);
        this.loadCurrentPassage();
        this.typingArea.disabled = false;
        this.typingArea.focus();
        this.typingStartTime = Date.now();
        eel.start_calibration_listeners()();
    }

    loadCurrentPassage() {
        if (this.currentPassage < this.typingPassages.length) {
            this.passageText.textContent = this.typingPassages[this.currentPassage];
            this.passageNumber.textContent = `Passage ${this.currentPassage + 1} of ${this.typingPassages.length}`;
            this.typingArea.value = '';
            this.typingArea.placeholder = 'Start typing the passage above...';
            this.updatePassageProgress(0);
        }
    }

    handleTypingInput(e) {
        const typed = e.target.value;
        const original = this.typingPassages[this.currentPassage];
        const progress = (typed.length / original.length) * 100;
        this.updatePassageProgress(Math.min(progress, 100));

        let correctChars = 0;
        for (let i = 0; i < Math.min(typed.length, original.length); i++) {
            if (typed[i] === original[i]) correctChars++;
        }
        const accuracy = typed.length > 0 ? (correctChars / typed.length) * 100 : 100;

        const timeElapsed = (Date.now() - this.typingStartTime) / 60000;
        const wordsTyped = typed.length / 5;
        const wpm = timeElapsed > 0 ? Math.round(wordsTyped / timeElapsed) : 0;

        this.typingStats.charactersTyped = typed.length;
        this.typingStats.wpm = wpm;
        this.typingStats.accuracy = Math.round(accuracy);
        this.updateTypingStats();

        if (typed.length >= original.length * 0.95) {
            this.onPassageComplete();
        }
        this.updateTypingFeedback(typed, original);
    }

    captureKeystroke(e) {
        if (this.currentSection !== 'typing') return;
        const ks = { key: e.key, code: e.code, type: e.type, timestamp: Date.now(), ctrlKey: e.ctrlKey, shiftKey: e.shiftKey };
        if (e.type === 'keydown') ks.downTime = Date.now();
        else if (e.type === 'keyup') {
            ks.upTime = Date.now();
            ks.holdTime = ks.upTime - (ks.downTime || ks.upTime);
        }
        this.keystrokeData.push(ks);
        if (this.keystrokeData.length >= 50) {
            eel.send_calibration_data('keystroke', this.keystrokeData.slice())();
            this.typingStats.samples++;
            this.updateTypingStats();
            this.keystrokeData = [];
        }
    }

    captureMouseMovement(e) {
        if (this.currentSection !== 'mouse') return;
        const ev = { type: 'move', x: e.clientX, y: e.clientY, timestamp: Date.now() };
        if (this.lastMousePosition) {
            const dx = e.clientX - this.lastMousePosition.x;
            const dy = e.clientY - this.lastMousePosition.y;
            const dt = Date.now() - this.lastMousePosition.timestamp;
            ev.velocity = Math.sqrt(dx * dx + dy * dy) / (dt || 1);
            ev.distance = Math.sqrt(dx * dx + dy * dy);
            this.mouseStats.distance += ev.distance;
            this.updateMouseStats();
        }
        this.lastMousePosition = { x: e.clientX, y: e.clientY, timestamp: Date.now() };
        this.mouseData.push(ev);
        if (this.mouseData.length >= 100) {
            eel.send_calibration_data('mouse', this.mouseData.slice())();
            this.mouseData = [];
        }
    }

    captureMouseClick(e) {
        const ev = { type: 'click', button: e.button, x: e.clientX, y: e.clientY, timestamp: Date.now(), eventType: e.type };
        if (e.type === 'mousedown') ev.downTime = Date.now();
        else if (e.type === 'mouseup') {
            ev.upTime = Date.now();
            ev.duration = ev.upTime - (ev.downTime || ev.upTime);
            this.mouseStats.clicks++;
            this.updateMouseStats();
        }
        this.mouseData.push(ev);
    }

    onPassageComplete() {
        this.skipPassage.disabled = false;
        this.nextToMouse.disabled = this.currentPassage < this.typingPassages.length - 1;
        this.typingFeedback.innerHTML = `<span class="feedback-text" style="color: var(--secondary-color);"><i class="fas fa-check-circle"></i> Passage completed! ${this.currentPassage < this.typingPassages.length - 1 ? 'Continue to next passage.' : 'All passages completed!'}</span>`;
    }

    nextPassage() {
        if (this.currentPassage < this.typingPassages.length - 1) {
            this.currentPassage++;
            this.loadCurrentPassage();
            this.skipPassage.disabled = true;
        }
    }

    startMouseCalibration() {
        this.currentSection = 'mouse';
        this.showSection('mouseSection');
        this.updateProgress(50);
        this.updateProgressSteps(3);
        this.currentExercise = 0;
        this.loadCurrentMouseExercise();
    }

    loadCurrentMouseExercise() {
        const ex = this.mouseExercises[this.currentExercise];
        this.exerciseName.textContent = ex.name;
        this.exerciseTimer.textContent = `${ex.duration}s`;
        this.playgroundInstructions.textContent = ex.description;
        this.startMouseExercise.disabled = false;
        this.nextMouseExerciseBtn.disabled = true;
        this.completeCalibrationBtn.disabled = true;
    }

    startCurrentMouseExercise() {
        const ex = this.mouseExercises[this.currentExercise];
        this.mouseExerciseStartTime = Date.now();
        this.startMouseExercise.disabled = true;
        this.mousePlayground.innerHTML = '';
        if (ex.type === 'targets') this.startTargetsExercise(ex.duration);
        else if (ex.type === 'tracking') this.startTrackingExercise(ex.duration);
        else if (ex.type === 'navigation') this.startNavigationExercise(ex.duration);
        else if (ex.type === 'precision') this.startPrecisionExercise(ex.duration);
        this.startExerciseCountdown(ex.duration);
    }

    startTargetsExercise(duration) {
        const createTarget = () => {
            const t = document.createElement('div');
            t.className = 'mouse-target';
            t.style.cssText = `position:absolute;width:40px;height:40px;background:var(--primary-color);border-radius:50%;cursor:pointer;left:${Math.random() * (this.mousePlayground.offsetWidth - 40)}px;top:${Math.random() * (this.mousePlayground.offsetHeight - 40)}px;`;
            t.addEventListener('click', () => { t.remove(); setTimeout(createTarget, 500); });
            this.mousePlayground.appendChild(t);
            setTimeout(() => { if (t.parentNode) { t.remove(); createTarget(); } }, 3000);
        };
        createTarget();
    }

    startTrackingExercise(duration) {
        const t = document.createElement('div');
        t.style.cssText = `position:absolute;width:30px;height:30px;background:var(--accent-color);border-radius:50%;pointer-events:none;`;
        this.mousePlayground.appendChild(t);
        let angle = 0;
        const cx = this.mousePlayground.offsetWidth / 2, cy = this.mousePlayground.offsetHeight / 2, r = Math.min(cx, cy) - 50;
        const move = () => {
            angle += 0.05;
            t.style.left = `${cx + Math.cos(angle) * r - 15}px`;
            t.style.top = `${cy + Math.sin(angle) * r - 15}px`;
            if (this.mouseExerciseStartTime && Date.now() - this.mouseExerciseStartTime < duration * 1000) requestAnimationFrame(move);
        };
        move();
    }

    startNavigationExercise(duration) {
        const svg = document.createElement('svg');
        svg.style.cssText = `position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;`;
        svg.innerHTML = `<path d="M 50 50 Q 200 100 350 50 T 350 200 Q 200 250 50 200 Z" stroke="var(--primary-color)" stroke-width="3" fill="none" stroke-dasharray="5,5"><animate attributeName="stroke-dashoffset" values="0;10" dur="1s" repeatCount="indefinite"/></path>`;
        this.mousePlayground.appendChild(svg);
    }

    startPrecisionExercise(duration) {
        for (let i = 0; i < 8; i++) {
            const t = document.createElement('div');
            t.style.cssText = `position:absolute;width:20px;height:20px;background:var(--danger-color);border-radius:50%;cursor:pointer;left:${50 + i * 40}px;top:${100 + (i % 2) * 60}px;`;
            t.addEventListener('click', () => { t.style.background = 'var(--secondary-color)'; t.style.transform = 'scale(1.5)'; });
            this.mousePlayground.appendChild(t);
        }
    }

    startExerciseCountdown(duration) {
        let remaining = duration;
        const update = () => {
            this.exerciseTimer.textContent = `${remaining}s`;
            remaining--;
            if (remaining >= 0) setTimeout(update, 1000);
            else this.onExerciseComplete();
        };
        update();
    }

    onExerciseComplete() {
        this.mouseStats.exercises++;
        this.updateMouseStats();
        this.mousePlayground.innerHTML = `<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;color:var(--secondary-color);"><i class="fas fa-check-circle" style="font-size:3rem;margin-bottom:1rem;"></i><h4>Exercise Complete!</h4></div>`;
        if (this.currentExercise < this.mouseExercises.length - 1) this.nextMouseExerciseBtn.disabled = false;
        else this.completeCalibrationBtn.disabled = false;
    }

    nextMouseExercise() {
        if (this.currentExercise < this.mouseExercises.length - 1) {
            this.currentExercise++;
            this.loadCurrentMouseExercise();
        }
    }

    async completeCalibration() {
        this.showStatusOverlay('Processing Calibration', 'Training behavioral models with your data...');
        if (this.keystrokeData.length > 0) await eel.send_calibration_data('keystroke', this.keystrokeData)();
        if (this.mouseData.length > 0) await eel.send_calibration_data('mouse', this.mouseData)();
        try {
            const result = await eel.complete_calibration()();
            this.hideStatusOverlay();
            this.showCompletionSection(result);
        } catch (error) {
            console.error('Calibration error:', error);
            alert('Calibration failed: ' + error);
        }
    }

    showCompletionSection(data) {
        this.currentSection = 'completion';
        this.showSection('completionSection');
        this.updateProgress(100);
        this.updateProgressSteps(4);
        document.getElementById('finalKeystrokeSamples').textContent = this.typingStats.samples;
        document.getElementById('finalMouseSamples').textContent = this.mouseStats.exercises;
        document.getElementById('modelAccuracy').textContent = Math.round((data.accuracy || 0.8) * 100) + '%';
        const tt = Math.round((Date.now() - this.typingStartTime) / 60000);
        document.getElementById('trainingTime').textContent = `${tt} min`;
    }

    showSection(sectionId) {
        this.sections.forEach(s => s.classList.remove('active'));
        document.getElementById(sectionId).classList.add('active');
    }

    updateProgress(p) { this.overallProgress.style.width = `${p}%`; this.progressText.textContent = `${p}% Complete`; }
    updateProgressSteps(s) { this.progressSteps.forEach((st, i) => st.classList.toggle('active', i < s)); }
    updatePassageProgress(p) { this.passageProgress.style.width = `${Math.min(p, 100)}%`; }
    updateTypingStats() {
        this.typingProgressEl.textContent = this.typingStats.charactersTyped;
        this.typingWPMEl.textContent = this.typingStats.wpm;
        this.typingAccuracyEl.textContent = this.typingStats.accuracy;
        this.samplesCollectedEl.textContent = this.typingStats.samples;
    }
    updateMouseStats() {
        this.mouseDistanceEl.textContent = Math.round(this.mouseStats.distance);
        this.mouseClicksEl.textContent = this.mouseStats.clicks;
        this.mouseVelocityEl.textContent = Math.round(this.mouseStats.velocity || 0);
        this.mouseExercisesEl.textContent = this.mouseStats.exercises;
    }
    updateTypingFeedback(typed, original) {
        const errors = this.countTypingErrors(typed, original);
        let f = '';
        if (errors === 0) f = '<span style="color:var(--secondary-color);"><i class="fas fa-check"></i> Perfect typing!</span>';
        else if (errors <= 3) f = '<span style="color:var(--accent-color);"><i class="fas fa-exclamation-triangle"></i> Good accuracy</span>';
        else f = '<span style="color:var(--warning-color);"><i class="fas fa-times"></i> Focus on accuracy</span>';
        this.typingFeedback.innerHTML = `<span class="feedback-text">${f}</span>`;
    }
    countTypingErrors(typed, original) {
        let errors = 0;
        for (let i = 0; i < Math.min(typed.length, original.length); i++) if (typed[i] !== original[i]) errors++;
        return errors;
    }
    showStatusOverlay(title, msg) { this.statusTitle.textContent = title; this.statusMessage.textContent = msg; this.statusOverlay.style.display = 'flex'; }
    hideStatusOverlay() { this.statusOverlay.style.display = 'none'; }
}

document.addEventListener('DOMContentLoaded', () => {
    const cm = new CalibrationManager();
    console.log('Locksy Calibration Initialized');
    window.calibrationManager = cm;
});
