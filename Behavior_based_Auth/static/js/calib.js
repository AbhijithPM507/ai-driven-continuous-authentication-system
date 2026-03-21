/**
 * Calibration Page JavaScript
 * Handles behavioral data collection and calibration process
 */

class CalibrationManager {
    constructor() {
        this.socket = null;
        this.sessionId = localStorage.getItem('session_id');
        this.userId = localStorage.getItem('user_id');
        this.username = localStorage.getItem('username');
        
        // Calibration state
        this.currentSection = 'welcome';
        this.currentPassage = 0;
        this.currentExercise = 0;
        this.typingStartTime = null;
        this.mouseExerciseStartTime = null;
        
        // Data collection
        this.keystrokeData = [];
        this.mouseData = [];
        this.typingStats = {
            charactersTyped: 0,
            wpm: 0,
            accuracy: 100,
            samples: 0
        };
        this.mouseStats = {
            distance: 0,
            clicks: 0,
            velocity: 0,
            exercises: 0
        };
        
        // Typing passages
        this.typingPassages = [
            "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is commonly used for typing practice. It helps develop muscle memory and finger coordination while maintaining proper typing posture.",
            
            "Behavioral biometrics represents a fascinating frontier in cybersecurity technology. Unlike traditional authentication methods that rely on what you know or what you have, behavioral biometrics focuses on what you do and how you do it.",
            
            "Machine learning algorithms can detect subtle patterns in human behavior that are virtually impossible to replicate. These patterns include typing rhythm, mouse movement trajectories, and even the way someone holds their mobile device.",
            
            "Continuous authentication provides a significant advantage over one-time password systems. By constantly monitoring user behavior, it can detect unauthorized access attempts in real-time, even after initial login verification has been completed successfully.",
            
            "The future of cybersecurity lies in adaptive systems that learn and evolve with user behavior. These intelligent systems can distinguish between genuine changes in behavior patterns and potential security threats, providing seamless protection without compromising user experience."
        ];
        
        // Mouse exercises
        this.mouseExercises = [
            {
                name: "Click Timing Exercise",
                description: "Click the targets as they appear",
                duration: 30,
                type: "targets"
            },
            {
                name: "Tracking Exercise",
                description: "Follow the moving target with your cursor",
                duration: 25,
                type: "tracking"
            },
            {
                name: "Navigation Exercise",
                description: "Navigate through the maze path",
                duration: 35,
                type: "navigation"
            },
            {
                name: "Precision Exercise",
                description: "Click precisely on small targets",
                duration: 20,
                type: "precision"
            }
        ];
        
        this.initializeElements();
        this.setupEventListeners();
        this.connectWebSocket();
        this.updateUserInfo();
    }

    initializeElements() {
        // Progress elements
        this.progressSteps = document.querySelectorAll('.step');
        this.overallProgress = document.getElementById('overallProgress');
        this.progressText = document.getElementById('progressText');
        
        // Section elements
        this.sections = document.querySelectorAll('.calibration-section');
        this.welcomeSection = document.getElementById('welcomeSection');
        this.typingSection = document.getElementById('typingSection');
        this.mouseSection = document.getElementById('mouseSection');
        this.completionSection = document.getElementById('completionSection');
        
        // Typing elements
        this.passageText = document.getElementById('passageText');
        this.typingArea = document.getElementById('typingArea');
        this.passageNumber = document.getElementById('passageNumber');
        this.passageProgress = document.getElementById('passageProgress');
        this.typingFeedback = document.getElementById('typingFeedback');
        
        // Mouse elements
        this.mousePlayground = document.getElementById('mousePlayground');
        this.exerciseName = document.getElementById('exerciseName');
        this.exerciseTimer = document.getElementById('exerciseTimer');
        this.playgroundInstructions = document.getElementById('playgroundInstructions');
        
        // Buttons
        this.startCalibration = document.getElementById('startCalibration');
        this.nextPassageBtn = document.getElementById('nextPassage');
        this.skipPassage = document.getElementById('skipPassage');
        this.nextToMouse = document.getElementById('nextToMouse');
        this.startMouseExercise = document.getElementById('startMouseExercise');
        this.nextMouseExerciseBtn = document.getElementById('nextMouseExercise');
        this.completeCalibrationBtn = document.getElementById('completeCalibration');
        this.continueToChallenge = document.getElementById('continueToChallenge');
        
        // Stats elements
        this.typingProgressEl = document.getElementById('typingProgress');
        this.typingWPMEl = document.getElementById('typingWPM');
        this.typingAccuracyEl = document.getElementById('typingAccuracy');
        this.samplesCollectedEl = document.getElementById('samplesCollected');
        
        this.mouseDistanceEl = document.getElementById('mouseDistance');
        this.mouseClicksEl = document.getElementById('mouseClicks');
        this.mouseVelocityEl = document.getElementById('mouseVelocity');
        this.mouseExercisesEl = document.getElementById('mouseExercises');
        
        // Overlays
        this.statusOverlay = document.getElementById('statusOverlay');
        this.statusTitle = document.getElementById('statusTitle');
        this.statusMessage = document.getElementById('statusMessage');
    }

    setupEventListeners() {
        // Button events
        this.startCalibration.addEventListener('click', () => this.startTypingCalibration());
        this.nextPassageBtn.addEventListener('click', () => this.nextPassage());
        this.skipPassage.addEventListener('click', () => this.skipPassageAction());
        this.nextToMouse.addEventListener('click', () => this.startMouseCalibration());
        this.startMouseExercise.addEventListener('click', () => this.startCurrentMouseExercise());
        this.nextMouseExerciseBtn.addEventListener('click', () => this.nextMouseExercise());
        this.completeCalibrationBtn.addEventListener('click', () => this.completeCalibration());
        this.continueToChallenge.addEventListener('click', () => this.redirectToChallenge());
        
        // Typing events
        this.typingArea.addEventListener('input', (e) => this.handleTypingInput(e));
        this.typingArea.addEventListener('keydown', (e) => this.captureKeystroke(e));
        this.typingArea.addEventListener('keyup', (e) => this.captureKeystroke(e));
        
        // Mouse events
        document.addEventListener('mousemove', (e) => this.captureMouseMovement(e));
        document.addEventListener('mousedown', (e) => this.captureMouseClick(e));
        document.addEventListener('mouseup', (e) => this.captureMouseClick(e));
        
        // Logout
        document.getElementById('logoutBtn').addEventListener('click', () => this.logout());
        
        // Prevent accidental page navigation
        window.addEventListener('beforeunload', (e) => {
            if (this.currentSection !== 'completion') {
                e.preventDefault();
                e.returnValue = 'Calibration is in progress. Are you sure you want to leave?';
            }
        });
    }

    connectWebSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to calibration server');
            this.socket.emit('join_session', { session_id: this.sessionId });
        });
        
        this.socket.on('session_joined', (data) => {
            console.log('Session joined:', data.message);
        });
        
        this.socket.on('session_error', (data) => {
            console.error('Session error:', data.error);
            this.showError('Session error. Please log in again.');
            setTimeout(() => window.location.href = '/login', 2000);
        });
        
        this.socket.on('error', (data) => {
            console.error('Socket error:', data.error);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            // Attempt to reconnect
            setTimeout(() => this.connectWebSocket(), 2000);
        });
    }

    updateUserInfo() {
        document.getElementById('username').textContent = this.username || 'User';
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
        this.startDataCollection();
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
        
        // Calculate progress
        const progress = (typed.length / original.length) * 100;
        this.updatePassageProgress(Math.min(progress, 100));
        
        // Calculate accuracy
        let correctChars = 0;
        for (let i = 0; i < Math.min(typed.length, original.length); i++) {
            if (typed[i] === original[i]) {
                correctChars++;
            }
        }
        
        const accuracy = typed.length > 0 ? (correctChars / typed.length) * 100 : 100;
        
        // Calculate WPM
        const timeElapsed = (Date.now() - this.typingStartTime) / 60000; // minutes
        const wordsTyped = typed.length / 5; // average word length
        const wpm = timeElapsed > 0 ? Math.round(wordsTyped / timeElapsed) : 0;
        
        // Update stats
        this.typingStats.charactersTyped = typed.length;
        this.typingStats.wpm = wpm;
        this.typingStats.accuracy = Math.round(accuracy);
        this.updateTypingStats();
        
        // Show "Next Passage" button after typing 20+ characters
        if (typed.length >= 20 && this.currentPassage < this.typingPassages.length - 1) {
            this.nextPassageBtn.disabled = false;
            this.nextPassageBtn.style.display = 'block';
        }
        
        // Enable "Continue to Mouse Exercises" button after typing 20+ characters
        if (typed.length >= 20) {
            this.nextToMouse.disabled = false;
        }
        
        // Check if passage is complete
        if (typed.length >= original.length * 0.95) { // 95% completion
            this.onPassageComplete();
        }
        
        // Update feedback
        this.updateTypingFeedback(typed, original);
    }

    captureKeystroke(e) {
        if (this.currentSection !== 'typing') return;
        
        const keystroke = {
            key: e.key,
            code: e.code,
            type: e.type, // keydown or keyup
            timestamp: Date.now(),
            ctrlKey: e.ctrlKey,
            shiftKey: e.shiftKey,
            altKey: e.altKey
        };
        
        // Calculate timing metrics
        if (e.type === 'keydown') {
            keystroke.downTime = Date.now();
        } else if (e.type === 'keyup') {
            keystroke.upTime = Date.now();
            keystroke.holdTime = keystroke.upTime - (keystroke.downTime || keystroke.upTime);
        }
        
        this.keystrokeData.push(keystroke);
        
        // Send data in batches
        if (this.keystrokeData.length >= 50) {
            this.sendBehavioralData('keystroke', this.keystrokeData.slice());
            this.keystrokeData = [];
        }
    }

    captureMouseMovement(e) {
        if (this.currentSection !== 'mouse') return;
        
        const mouseEvent = {
            type: 'move',
            x: e.clientX,
            y: e.clientY,
            timestamp: Date.now(),
            target: e.target.tagName
        };
        
        // Calculate velocity if we have previous position
        if (this.lastMousePosition) {
            const dx = e.clientX - this.lastMousePosition.x;
            const dy = e.clientY - this.lastMousePosition.y;
            const dt = Date.now() - this.lastMousePosition.timestamp;
            
            mouseEvent.velocity = Math.sqrt(dx * dx + dy * dy) / (dt || 1);
            mouseEvent.distance = Math.sqrt(dx * dx + dy * dy);
            
            this.mouseStats.distance += mouseEvent.distance;
            this.updateMouseStats();
        }
        
        this.lastMousePosition = {
            x: e.clientX,
            y: e.clientY,
            timestamp: Date.now()
        };
        
        this.mouseData.push(mouseEvent);
        
        // Send data in batches
        if (this.mouseData.length >= 100) {
            this.sendBehavioralData('mouse', this.mouseData.slice());
            this.mouseData = [];
        }
    }

    captureMouseClick(e) {
        const clickEvent = {
            type: 'click',
            button: e.button, // 0: left, 1: middle, 2: right
            x: e.clientX,
            y: e.clientY,
            timestamp: Date.now(),
            eventType: e.type, // mousedown or mouseup
            target: e.target.tagName
        };
        
        if (e.type === 'mousedown') {
            clickEvent.downTime = Date.now();
        } else if (e.type === 'mouseup') {
            clickEvent.upTime = Date.now();
            clickEvent.duration = clickEvent.upTime - (clickEvent.downTime || clickEvent.upTime);
            this.mouseStats.clicks++;
            this.updateMouseStats();
        }
        
        this.mouseData.push(clickEvent);
    }

    sendBehavioralData(type, data) {
        if (this.socket && data.length > 0) {
            this.socket.emit('behavioral_data', {
                type: type,
                events: data,
                timestamp: Date.now()
            });
            
            this.typingStats.samples++;
            this.updateTypingStats();
        }
    }

    onPassageComplete() {
        this.skipPassage.disabled = false;
        // Enable mouse exercises button after ANY passage is completed
        this.nextToMouse.disabled = false;
        
        this.typingFeedback.innerHTML = `
            <span class="feedback-text" style="color: var(--secondary-color);">
                <i class="fas fa-check-circle"></i> Passage completed! 
                ${this.currentPassage < this.typingPassages.length - 1 ? 'Continue to next passage or move to mouse exercises.' : 'All passages completed!'}
            </span>
        `;
    }

    nextPassage() {
        if (this.currentPassage < this.typingPassages.length - 1) {
            this.currentPassage++;
            this.loadCurrentPassage();
            this.nextPassageBtn.disabled = true;
            this.nextPassageBtn.style.display = 'none';
            this.skipPassage.disabled = true;
        }
    }
    
    skipPassageAction() {
        this.nextPassage();
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
        const exercise = this.mouseExercises[this.currentExercise];
        this.exerciseName.textContent = exercise.name;
        this.exerciseTimer.textContent = `${exercise.duration}s`;
        this.playgroundInstructions.textContent = exercise.description;
        
        this.startMouseExercise.disabled = false;
        this.nextMouseExerciseBtn.disabled = true;
        this.completeCalibrationBtn.disabled = true;
    }

    startCurrentMouseExercise() {
        const exercise = this.mouseExercises[this.currentExercise];
        this.mouseExerciseStartTime = Date.now();
        
        this.startMouseExercise.disabled = true;
        this.mousePlayground.innerHTML = '';
        
        switch (exercise.type) {
            case 'targets':
                this.startTargetsExercise(exercise.duration);
                break;
            case 'tracking':
                this.startTrackingExercise(exercise.duration);
                break;
            case 'navigation':
                this.startNavigationExercise(exercise.duration);
                break;
            case 'precision':
                this.startPrecisionExercise(exercise.duration);
                break;
        }
        
        // Start countdown
        this.startExerciseCountdown(exercise.duration);
    }

    startTargetsExercise(duration) {
        const createTarget = () => {
            const target = document.createElement('div');
            target.className = 'mouse-target';
            target.style.cssText = `
                position: absolute;
                width: 40px;
                height: 40px;
                background: var(--primary-color);
                border-radius: 50%;
                cursor: pointer;
                transition: all 0.2s ease-out;
                left: ${Math.random() * (this.mousePlayground.offsetWidth - 40)}px;
                top: ${Math.random() * (this.mousePlayground.offsetHeight - 40)}px;
            `;
            
            target.addEventListener('click', () => {
                target.style.background = 'var(--secondary-color)';
                target.style.transform = 'scale(1.2)';
                setTimeout(() => target.remove(), 200);
                
                // Create new target
                setTimeout(createTarget, 500);
            });
            
            target.addEventListener('mouseenter', () => {
                target.style.transform = 'scale(1.1)';
            });
            
            target.addEventListener('mouseleave', () => {
                target.style.transform = 'scale(1)';
            });
            
            this.mousePlayground.appendChild(target);
            
            // Auto-remove after 3 seconds
            setTimeout(() => {
                if (target.parentNode) {
                    target.remove();
                    createTarget();
                }
            }, 3000);
        };
        
        createTarget();
    }

    startTrackingExercise(duration) {
        const target = document.createElement('div');
        target.className = 'tracking-target';
        target.style.cssText = `
            position: absolute;
            width: 30px;
            height: 30px;
            background: var(--accent-color);
            border-radius: 50%;
            pointer-events: none;
        `;
        
        this.mousePlayground.appendChild(target);
        
        let angle = 0;
        const centerX = this.mousePlayground.offsetWidth / 2;
        const centerY = this.mousePlayground.offsetHeight / 2;
        const radius = Math.min(centerX, centerY) - 50;
        
        const moveTarget = () => {
            angle += 0.05;
            const x = centerX + Math.cos(angle) * radius - 15;
            const y = centerY + Math.sin(angle) * radius - 15;
            
            target.style.left = `${x}px`;
            target.style.top = `${y}px`;
            
            if (this.mouseExerciseStartTime && Date.now() - this.mouseExerciseStartTime < duration * 1000) {
                requestAnimationFrame(moveTarget);
            }
        };
        
        moveTarget();
    }

    startNavigationExercise(duration) {
        // Create a simple maze path
        const path = document.createElement('svg');
        path.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        `;
        
        path.innerHTML = `
            <path d="M 50 50 Q 200 100 350 50 T 350 200 Q 200 250 50 200 Z" 
                  stroke="var(--primary-color)" 
                  stroke-width="3" 
                  fill="none" 
                  stroke-dasharray="5,5">
                <animate attributeName="stroke-dashoffset" values="0;10" dur="1s" repeatCount="indefinite"/>
            </path>
        `;
        
        this.mousePlayground.appendChild(path);
        
        const instructions = document.createElement('div');
        instructions.style.cssText = `
            position: absolute;
            top: 10px;
            left: 10px;
            color: var(--text-secondary);
            font-size: 14px;
        `;
        instructions.textContent = 'Follow the animated path with your cursor';
        this.mousePlayground.appendChild(instructions);
    }

    startPrecisionExercise(duration) {
        for (let i = 0; i < 8; i++) {
            const target = document.createElement('div');
            target.className = 'precision-target';
            target.style.cssText = `
                position: absolute;
                width: 20px;
                height: 20px;
                background: var(--danger-color);
                border-radius: 50%;
                cursor: pointer;
                transition: all 0.2s ease-out;
                left: ${50 + i * 40}px;
                top: ${100 + (i % 2) * 60}px;
            `;
            
            target.addEventListener('click', () => {
                target.style.background = 'var(--secondary-color)';
                target.style.transform = 'scale(1.5)';
            });
            
            this.mousePlayground.appendChild(target);
        }
    }

    startExerciseCountdown(duration) {
        let remaining = duration;
        
        const updateTimer = () => {
            this.exerciseTimer.textContent = `${remaining}s`;
            remaining--;
            
            if (remaining >= 0) {
                setTimeout(updateTimer, 1000);
            } else {
                this.onExerciseComplete();
            }
        };
        
        updateTimer();
    }

    onExerciseComplete() {
        this.mouseStats.exercises++;
        this.updateMouseStats();
        
        this.mousePlayground.innerHTML = `
            <div style="
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                text-align: center;
                color: var(--secondary-color);
            ">
                <i class="fas fa-check-circle" style="font-size: 3rem; margin-bottom: 1rem;"></i>
                <h4>Exercise Complete!</h4>
            </div>
        `;
        
        if (this.currentExercise < this.mouseExercises.length - 1) {
            this.nextMouseExerciseBtn.disabled = false;
        } else {
            this.completeCalibrationBtn.disabled = false;
        }
    }

    nextMouseExercise() {
        if (this.currentExercise < this.mouseExercises.length - 1) {
            this.currentExercise++;
            this.loadCurrentMouseExercise();
        }
    }

    async completeCalibration() {
        this.showStatusOverlay('Processing Calibration', 'Training behavioral models with your data...');
        
        // Send any remaining data
        if (this.keystrokeData.length > 0) {
            this.sendBehavioralData('keystroke', this.keystrokeData);
        }
        if (this.mouseData.length > 0) {
            this.sendBehavioralData('mouse', this.mouseData);
        }
        
        try {
            const response = await fetch('/api/calibration/complete', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.hideStatusOverlay();
                this.showCompletionSection(data);
            } else {
                this.showError(data.error || 'Calibration failed');
            }
        } catch (error) {
            console.error('Calibration completion error:', error);
            this.showError('Network error during calibration completion');
        }
    }

    showCompletionSection(data) {
        this.currentSection = 'completion';
        this.showSection('completionSection');
        this.updateProgress(100);
        this.updateProgressSteps(4);
        
        // Update completion stats
        if (data.training_results) {
            document.getElementById('finalKeystrokeSamples').textContent = this.typingStats.samples;
            document.getElementById('finalMouseSamples').textContent = this.mouseStats.exercises;
            document.getElementById('modelAccuracy').textContent = 
                Math.round((data.training_results.accuracy || 0.8) * 100) + '%';
        }
        
        const trainingTime = Math.round((Date.now() - this.typingStartTime) / 60000);
        document.getElementById('trainingTime').textContent = `${trainingTime} min`;
    }

    redirectToChallenge() {
        window.location.href = '/challenge';
    }

    startDataCollection() {
        // Start collecting mouse movement data
        this.lastMousePosition = null;
        this.dataCollectionActive = true;
    }

    showSection(sectionId) {
        this.sections.forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById(sectionId).classList.add('active');
    }

    updateProgress(percent) {
        this.overallProgress.style.width = `${percent}%`;
        this.progressText.textContent = `${percent}% Complete`;
    }

    updateProgressSteps(activeStep) {
        this.progressSteps.forEach((step, index) => {
            if (index < activeStep) {
                step.classList.add('active');
            } else {
                step.classList.remove('active');
            }
        });
    }

    updatePassageProgress(percent) {
        this.passageProgress.style.width = `${percent}%`;
    }
    

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
        let feedback = '';
        
        if (errors === 0) {
            feedback = '<span style="color: var(--secondary-color);"><i class="fas fa-check"></i> Perfect typing!</span>';
        } else if (errors <= 3) {
            feedback = '<span style="color: var(--accent-color);"><i class="fas fa-exclamation-triangle"></i> Good accuracy</span>';
        } else {
            feedback = '<span style="color: var(--warning-color);"><i class="fas fa-times"></i> Focus on accuracy</span>';
        }
        
        this.typingFeedback.innerHTML = `<span class="feedback-text">${feedback}</span>`;
    }

    countTypingErrors(typed, original) {
        let errors = 0;
        for (let i = 0; i < Math.min(typed.length, original.length); i++) {
            if (typed[i] !== original[i]) {
                errors++;
            }
        }
        return errors;
    }

    showStatusOverlay(title, message) {
        this.statusTitle.textContent = title;
        this.statusMessage.textContent = message;
        this.statusOverlay.style.display = 'flex';
    }

    hideStatusOverlay() {
        this.statusOverlay.style.display = 'none';
    }

    showError(message) {
        // Implementation similar to login page
        console.error('Calibration error:', message);
        alert(message); // Simple fallback - could be improved with modal
    }

    async logout() {
        try {
            await fetch('/api/logout', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId
                })
            });
        } catch (error) {
            console.error('Logout error:', error);
        } finally {
            localStorage.clear();
            window.location.href = '/login';
        }
    }
}

// Initialize calibration manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check authentication
    const sessionId = localStorage.getItem('session_id');
    if (!sessionId) {
        window.location.href = '/login';
        return;
    }
    
    const calibrationManager = new CalibrationManager();
    
    console.log('🧠 Behavioral Calibration System Initialized');
});

// Export for potential use in other modules
window.CalibrationManager = CalibrationManager;