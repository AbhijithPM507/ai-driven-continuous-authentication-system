// js/challenge.js

/**
 * Show lockdown countdown overlay when intruder is detected
 */
function showLockdownCountdown() {
    const overlay = document.createElement('div');
    overlay.id = 'lockdown-overlay';
    overlay.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(220, 38, 38, 0.95); z-index: 9999;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        color: white; font-family: monospace;
    `;
    overlay.innerHTML = `
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">
            ⚠ INTRUDER DETECTED
        </h1>
        <p style="font-size: 1.2rem; margin-bottom: 2rem;">
            Behavioural pattern does not match authorized user
        </p>
        <h2 id="countdown-num" style="font-size: 5rem; font-weight: bold;">10</h2>
        <p style="font-size: 1rem; margin-top: 1rem; opacity: 0.8;">
            Locking workstation... Press ESC to cancel if this is a false alarm
        </p>
    `;
    document.body.appendChild(overlay);
    
    let count = 10;
    const interval = setInterval(() => {
        count--;
        const el = document.getElementById('countdown-num');
        if (el) el.textContent = count;
        if (count <= 0) {
            clearInterval(interval);
            overlay.innerHTML = '<h1 style="font-size:3rem">🔒 Workstation Locked</h1>';
        }
    }, 1000);
    
    // ESC key cancels lockdown
    document.addEventListener('keydown', function cancelLock(e) {
        if (e.key === 'Escape') {
            clearInterval(interval);
            overlay.remove();
            document.removeEventListener('keydown', cancelLock);
            console.log('Lockdown cancelled by user');
        }
    });
}

// --- Global keyboard collection (reliable top-level listener) ---
const keystrokeBuffer = [];
let lastKeyUpTime = null;
let typingTimeout;

document.addEventListener('keydown', function(e) {
    if (e.key === 'Shift' || e.key === 'Control' || e.key === 'Alt' || e.key === 'Meta') return;
    keystrokeBuffer.push({
        key: e.key,
        press_time: Date.now(),
        flight_time: lastKeyUpTime ? Date.now() - lastKeyUpTime : 0
    });
});

document.addEventListener('keyup', function(e) {
    if (e.key === 'Shift' || e.key === 'Control' || e.key === 'Alt' || e.key === 'Meta') return;
    lastKeyUpTime = Date.now();
    const last = keystrokeBuffer[keystrokeBuffer.length - 1];
    if (last && !last.dwell_time) {
        last.dwell_time = Date.now() - last.press_time;
    }
    clearTimeout(typingTimeout);
    typingTimeout = setTimeout(() => {
        if (keystrokeBuffer.length >= 3) {
            sendBehavioralDataForAuth();
        }
    }, 1000);
});

document.addEventListener('DOMContentLoaded', () => {
    const challengeInput = document.getElementById('challengeInput');
    const authStatusBox = document.getElementById('authStatusBox');
    const authStatusMessage = document.getElementById('authStatusMessage');
    const logoutBtn = document.getElementById('logoutBtn');

    const SLIDING_WINDOW_SIZE_MS = 30 * 1000; // 30 seconds
    const AUTHENTICATION_INTERVAL_MS = 5 * 1000; // Send data every 5 seconds for continuous auth

    let mouseEvents = [];
    let lastMouseMoveTime = 0;
    let authenticationInterval;
    let webSocket; // Simulated WebSocket

    // --- UI Update Function ---
    function updateAuthStatus(status, message) {
        const ids = ['auth-status', 'status-indicator', 'security-badge', 'risk-level'];
        ids.forEach(id => {
            const el = document.getElementById(id);
            if (!el) return;
            el.classList.remove('authorized', 'anomaly', 'warning', 'safe', 'danger');
            if (status === 'authorized') el.classList.add('authorized');
            if (status === 'anomaly') el.classList.add('anomaly');
        });

        const textEl = document.getElementById('auth-status-text');
        if (textEl) textEl.textContent = status;

        if (authStatusBox) {
            authStatusBox.classList.remove('auth', 'risk', 'high-risk', 'loading');
            if (message && authStatusMessage) authStatusMessage.textContent = message;
            const spinner = authStatusBox.querySelector('svg');
            if (spinner) spinner.remove();

            switch (status) {
                case 'authorized':
                    authStatusBox.classList.add('auth');
                    break;
                case 'anomaly':
                case 'risk':
                    authStatusBox.classList.add('risk');
                    break;
                case 'high-risk':
                    authStatusBox.classList.add('high-risk');
                    break;
                case 'loading':
                default:
                    authStatusBox.classList.add('loading');
                    addSpinner(authStatusBox);
                    break;
            }
        }
    }

    // Helper to add a loading spinner
    function addSpinner(element) {
        const spinnerSvg = `
            <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-gray-700" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
        `;
        // Insert spinner at the beginning of the status box
        element.insertAdjacentHTML('afterbegin', spinnerSvg);
    }

    // --- Data Collection ---

    // Mouse event listeners (on the whole document for broader capture)
    document.addEventListener('mousemove', (e) => {
        const currentTime = performance.now();
        if (currentTime - lastMouseMoveTime > 50) {
            mouseEvents.push({
                type: 'mousemove',
                x: e.clientX,
                y: e.clientY,
                timestamp: currentTime
            });
            lastMouseMoveTime = currentTime;
        }
    });

    document.addEventListener('mousedown', (e) => {
        mouseEvents.push({
            type: 'mousedown',
            button: e.button,
            x: e.clientX,
            y: e.clientY,
            timestamp: performance.now()
        });
    });

    document.addEventListener('mouseup', (e) => {
        mouseEvents.push({
            type: 'mouseup',
            button: e.button,
            x: e.clientX,
            y: e.clientY,
            timestamp: performance.now()
        });
    });

    // --- Sliding Window & WebSocket Simulation ---

    function sendBehavioralDataForAuth() {
        try {
            const currentTime = performance.now();
            const windowStartTime = currentTime - SLIDING_WINDOW_SIZE_MS;

            // Transfer full keystroke buffer into payload
            const keystrokeData = [...keystrokeBuffer];
            keystrokeBuffer.length = 0;

            // Filter mouse events within the current sliding window
            const windowMouseEvents = mouseEvents.filter(event => event.timestamp >= windowStartTime);
            mouseEvents = windowMouseEvents;

            if (keystrokeData.length >= 5 || windowMouseEvents.length > 10) {
                const dataToSend = {
                    keystroke_data: keystrokeData,
                    mouse_data: windowMouseEvents,
                    timestamp: Date.now()
                };

                console.log('Sending behavioral data for authentication:', dataToSend);
                updateAuthStatus('loading', 'Authenticating your behavior...');
            } else {
                updateAuthStatus('authenticated', 'No activity, session secure.');
            }
        } catch (err) {
            console.warn('Auth send error:', err.message);
        }
    }

    // --- Initialization ---
    function initChallenge() {
        // Simulate WebSocket connection
        // webSocket = new WebSocket('ws://your-backend-websocket-url/challenge');
        // webSocket.onopen = () => console.log('WebSocket connected for challenge');
        // webSocket.onmessage = (event) => {
        //     const data = JSON.parse(event.data);
        //     console.log('Received auth status:', data);
        //     updateAuthStatus(data.status, data.message);
        // };
        // webSocket.onerror = (error) => console.error('WebSocket error:', error);
        // webSocket.onclose = () => console.log('WebSocket closed');

        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => {
                clearInterval(authenticationInterval);
                // if (webSocket) webSocket.close();
                console.log('Logged out.');
                window.location.href = 'login.html'; // Redirect to login page on logout
            });
        }

        // Initial status display
        updateAuthStatus('loading', 'Authenticating your behavior...');
        if (challengeInput) {
            challengeInput.focus();
        }
    }

    initChallenge();
});
/**
 * Challenge/Dashboard Page JavaScript
 * Handles real-time behavioral monitoring and dashboard functionality
 */

class DashboardManager {
    constructor() {
        this.socket = null;
        this.sessionId = localStorage.getItem('session_id');
        this.userId = localStorage.getItem('user_id');
        this.username = localStorage.getItem('username');
        
        // Real-time monitoring state
        this.isMonitoring = false;
        this.behavioralBuffer = {
            keystroke: [],
            mouse: []
        };
        this.lastMousePosition = null;
        this.lastKeystroke = null;
        
        // Authentication state
        this.currentAuthScore = 0.0;
        this.confidenceLevel = 0.0;
        this.anomalyRisk = 'Low';
        this.securityScore = 85;
        
        // Charts
        this.behaviorChart = null;
        this.driftChart = null;
        this.patternsChart = null;
        this.timeChart = null;
        
        // Dashboard state
        this.currentSection = 'dashboard';
        this.notificationCount = 0;
        this.securityAlerts = [];
        
        // Timers
        this.authUpdateInterval = null;
        this.chartUpdateInterval = null;
        this.statsUpdateInterval = null;
        
        this.initializeElements();
        this.setupEventListeners();
        this.connectWebSocket();
        this.initializeCharts();
        this.startRealTimeMonitoring();
        this.updateUserInfo();
    }

    initializeElements() {
        // Navigation
        this.navLinks = document.querySelectorAll('.nav-link');
        this.contentSections = document.querySelectorAll('.content-section');
        this.pageTitle = document.getElementById('pageTitle');
        this.sidebarToggle = document.getElementById('sidebarToggle');
        this.sidebar = document.querySelector('.sidebar');
        
        // Header elements
        this.authStatus = document.getElementById('authStatus');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        this.sidebarUsername = document.getElementById('sidebarUsername');
        
        // Notifications
        this.notificationBtn = document.getElementById('notificationBtn');
        this.notificationBadge = document.getElementById('notificationBadge');
        this.notificationDropdown = document.getElementById('notificationDropdown');
        this.notificationList = document.getElementById('notificationList');
        this.markAllRead = document.getElementById('markAllRead');
        
        // Dashboard stats
        this.securityScoreEl = document.getElementById('securityScore');
        this.securityScoreCircle = document.getElementById('securityScoreCircle');
        this.authScoreEl = document.getElementById('authScore');
        this.confidenceLevelEl = document.getElementById('confidenceLevel');
        this.anomalyRiskEl = document.getElementById('anomalyRisk');
        this.keystrokeSamplesEl = document.getElementById('keystrokeSamples');
        this.mouseSamplesEl = document.getElementById('mouseSamples');
        this.monitorStatus = document.getElementById('monitorStatus');
        
        // Activity log
        this.recentActivityList = document.getElementById('recentActivityList');
        this.activityTableBody = document.getElementById('activityTableBody');
        this.activityFilter = document.getElementById('activityFilter');
        this.dateFilter = document.getElementById('dateFilter');
        this.refreshActivity = document.getElementById('refreshActivity');
        
        // Quick actions
        this.runSecurityCheck = document.getElementById('runSecurityCheck');
        this.updateModels = document.getElementById('updateModels');
        this.exportLogs = document.getElementById('exportLogs');
        this.testBehavior = document.getElementById('testBehavior');
        
        // Settings
        this.enableRealTimeAuth = document.getElementById('enableRealTimeAuth');
        this.enableAnomalyAlerts = document.getElementById('enableAnomalyAlerts');
        this.enableDriftDetection = document.getElementById('enableDriftDetection');
        this.authThreshold = document.getElementById('authThreshold');
        this.anomalySensitivity = document.getElementById('anomalySensitivity');
        
        // Modals
        this.securityAlertModal = document.getElementById('securityAlertModal');
        this.alertTitle = document.getElementById('alertTitle');
        this.alertMessage = document.getElementById('alertMessage');
        this.alertDetails = document.getElementById('alertDetails');
        this.acknowledgeAlert = document.getElementById('acknowledgeAlert');
        this.investigateAlert = document.getElementById('investigateAlert');
        
        // Test area
        this.testArea = document.getElementById('testArea');
        this.startTest = document.getElementById('startTest');
        this.stopTest = document.getElementById('stopTest');
    }

    setupEventListeners() {
        // Navigation
        if (this.navLinks && this.navLinks.length > 0) {
            this.navLinks.forEach(link => {
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    const section = link.dataset.section;
                    this.showSection(section);
                });
            });
        }
        
        // Sidebar toggle
        if (this.sidebarToggle && this.sidebar) {
            this.sidebarToggle.addEventListener('click', () => {
                this.sidebar.classList.toggle('open');
            });
        }
        
        // Notifications
        if (this.notificationBtn) {
            this.notificationBtn.addEventListener('click', () => {
                this.toggleNotificationDropdown();
            });
        }
        
        if (this.markAllRead) {
            this.markAllRead.addEventListener('click', () => {
                this.markAllNotificationsRead();
            });
        }
        
        // Quick actions
        if (this.runSecurityCheck) {
            this.runSecurityCheck.addEventListener('click', () => this.runSecurityCheck());
        }
        if (this.updateModels) {
            this.updateModels.addEventListener('click', () => this.updateModels());
        }
        if (this.exportLogs) {
            this.exportLogs.addEventListener('click', () => this.exportLogs());
        }
        if (this.testBehavior) {
            this.testBehavior.addEventListener('click', () => this.toggleTestArea());
        }
        
        // Activity log
        if (this.refreshActivity) {
            this.refreshActivity.addEventListener('click', () => this.loadActivityLog());
        }
        if (this.activityFilter) {
            this.activityFilter.addEventListener('change', () => this.loadActivityLog());
        }
        if (this.dateFilter) {
            this.dateFilter.addEventListener('change', () => this.loadActivityLog());
        }
        
        // Settings
        if (this.authThreshold) {
            this.authThreshold.addEventListener('input', (e) => {
                e.target.nextElementSibling.textContent = e.target.value;
                this.updateSettings();
            });
        }
        
        if (this.anomalySensitivity) {
            this.anomalySensitivity.addEventListener('input', (e) => {
                e.target.nextElementSibling.textContent = e.target.value;
                this.updateSettings();
            });
        }
        
        // Real-time monitoring events
        document.addEventListener('keydown', (e) => this.captureKeystroke(e));
        document.addEventListener('keyup', (e) => this.captureKeystroke(e));
        document.addEventListener('mousemove', (e) => this.captureMouseMovement(e));
        document.addEventListener('mousedown', (e) => this.captureMouseClick(e));
        document.addEventListener('mouseup', (e) => this.captureMouseClick(e));
        document.addEventListener('click', (e) => this.captureMouseClick(e));
        
        // Test area
        if (this.startTest) {
            this.startTest.addEventListener('click', () => this.startBehaviorTest());
        }
        if (this.stopTest) {
            this.stopTest.addEventListener('click', () => this.stopBehaviorTest());
        }
        
        // Alert modal
        if (this.acknowledgeAlert) {
            this.acknowledgeAlert.addEventListener('click', () => this.acknowledgeSecurityAlert());
        }
        if (this.investigateAlert) {
            this.investigateAlert.addEventListener('click', () => this.investigateSecurityAlert());
        }
        
        // Logout
        const logoutBtn = document.getElementById('logoutBtn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => this.logout());
        }
        
        // Close dropdowns when clicking outside
        document.addEventListener('click', (e) => {
            if (!this.notificationBtn.contains(e.target)) {
                this.notificationDropdown.style.display = 'none';
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.shiftKey) {
                switch (e.key) {
                    case 'D':
                        e.preventDefault();
                        this.showSection('dashboard');
                        break;
                    case 'S':
                        e.preventDefault();
                        this.showSection('security');
                        break;
                    case 'A':
                        e.preventDefault();
                        this.showSection('analytics');
                        break;
                    case 'T':
                        e.preventDefault();
                        this.toggleTestArea();
                        break;
                }
            }
        });
    }

    connectWebSocket() {
        this.socket = io({
            reconnection: true,
            reconnectionAttempts: Infinity,
            reconnectionDelay: 1000,
            timeout: 60000
        });
        
        this.socket.on('connect', () => {
            console.log('Connected to dashboard server');
            this.socket.emit('join_session', { session_id: this.sessionId });
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('session_joined', (data) => {
            console.log('Dashboard session joined:', data.message);
            this.statusText.textContent = 'Monitoring Active';
        });
        
        this.socket.on('auth_result', (data) => {
            this.handleAuthResult(data);
        });
        
        this.socket.on('security_alert', (data) => {
            this.handleSecurityAlert(data);
        });
        
        this.socket.on('drift_analysis', (data) => {
            this.handleDriftAnalysis(data);
        });
        
        this.socket.on('session_error', (data) => {
            console.error('Session error:', data.error);
            this.showAlert('Session Error', data.error, 'error');
            setTimeout(() => window.location.href = '/login', 3000);
        });
        
        this.socket.on('error', (data) => {
            console.error('Socket error:', data.error);
            this.updateConnectionStatus(false);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus(false);
            this.statusText.textContent = 'Reconnecting...';
            
            // Attempt to reconnect
            setTimeout(() => this.connectWebSocket(), 2000);
        });
    }

    initializeCharts() {
        // Behavior monitoring chart
        const behaviorCtx = document.getElementById('behaviorChart');
        if (behaviorCtx) {
            this.behaviorChart = new Chart(behaviorCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Authentication Score',
                        data: [],
                        borderColor: 'rgb(99, 102, 241)',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                        },
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: 'rgba(255, 255, 255, 0.7)' } }
                    }
                }
            });
        }
        
        // Drift analysis chart
        const driftCtx = document.getElementById('driftChart');
        if (driftCtx) {
            this.driftChart = new Chart(driftCtx, {
                type: 'radar',
                data: {
                    labels: ['Typing Speed', 'Key Timing', 'Mouse Velocity', 'Click Patterns', 'Movement Efficiency'],
                    datasets: [{
                        label: 'Current Behavior',
                        data: [0.8, 0.9, 0.7, 0.85, 0.75],
                        borderColor: 'rgb(99, 102, 241)',
                        backgroundColor: 'rgba(99, 102, 241, 0.2)',
                        pointBackgroundColor: 'rgb(99, 102, 241)'
                    }, {
                        label: 'Baseline',
                        data: [0.8, 0.8, 0.8, 0.8, 0.8],
                        borderColor: 'rgb(16, 185, 129)',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        pointBackgroundColor: 'rgb(16, 185, 129)'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 1,
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                            pointLabels: { color: 'rgba(255, 255, 255, 0.7)' },
                            ticks: { color: 'rgba(255, 255, 255, 0.5)' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: 'rgba(255, 255, 255, 0.7)' } }
                    }
                }
            });
        }
        
        // Behavioral patterns chart
        const patternsCtx = document.getElementById('patternsChart');
        if (patternsCtx) {
            this.patternsChart = new Chart(patternsCtx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Keystroke Patterns',
                        data: [],
                        backgroundColor: 'rgba(99, 102, 241, 0.6)',
                        borderColor: 'rgb(99, 102, 241)'
                    }, {
                        label: 'Mouse Patterns',
                        data: [],
                        backgroundColor: 'rgba(16, 185, 129, 0.6)',
                        borderColor: 'rgb(16, 185, 129)'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                        },
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: 'rgba(255, 255, 255, 0.7)' } }
                    }
                }
            });
        }
        
        // Time-based analysis chart
        const timeCtx = document.getElementById('timeChart');
        if (timeCtx) {
            this.timeChart = new Chart(timeCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Authenticity Score',
                        data: [],
                        borderColor: 'rgb(99, 102, 241)',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        yAxisID: 'y'
                    }, {
                        label: 'Anomaly Score',
                        data: [],
                        borderColor: 'rgb(239, 68, 68)',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            beginAtZero: true,
                            max: 1,
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            beginAtZero: true,
                            max: 1,
                            grid: { drawOnChartArea: false },
                            ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                        },
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: 'rgba(255, 255, 255, 0.7)' } }
                    }
                }
            });
        }
    }

    startRealTimeMonitoring() {
        this.isMonitoring = true;
        this.updateMonitoringStatus(true);
        
        // Update authentication stats periodically
        this.authUpdateInterval = setInterval(() => {
            this.updateAuthenticationDisplay();
        }, 5000);
        
        // Update charts periodically
        this.chartUpdateInterval = setInterval(() => {
            this.updateCharts();
        }, 10000);
        
        // Update general stats
        this.statsUpdateInterval = setInterval(() => {
            this.updateGeneralStats();
        }, 30000);
    }

    captureKeystroke(e) {
        if (!this.isMonitoring) return;
        
        const keystroke = {
            key: e.key,
            code: e.code,
            type: e.type,
            timestamp: Date.now(),
            ctrlKey: e.ctrlKey,
            shiftKey: e.shiftKey,
            altKey: e.altKey
        };
        
        // Calculate timing metrics
        if (this.lastKeystroke && e.type === 'keydown') {
            keystroke.flightTime = keystroke.timestamp - this.lastKeystroke.timestamp;
        }
        
        if (e.type === 'keydown') {
            keystroke.downTime = keystroke.timestamp;
        } else if (e.type === 'keyup') {
            keystroke.upTime = keystroke.timestamp;
            if (this.lastKeystroke && this.lastKeystroke.downTime) {
                keystroke.holdTime = keystroke.upTime - this.lastKeystroke.downTime;
            }
        }
        
        this.behavioralBuffer.keystroke.push(keystroke);
        this.lastKeystroke = keystroke;
        
        // Send data in batches
        if (this.behavioralBuffer.keystroke.length >= 20) {
            this.sendBehavioralData('keystroke', this.behavioralBuffer.keystroke.slice());
            this.behavioralBuffer.keystroke = [];
        }
    }

    captureMouseMovement(e) {
        if (!this.isMonitoring) return;
        
        const mouseEvent = {
            type: 'move',
            x: e.clientX,
            y: e.clientY,
            timestamp: Date.now(),
            target: e.target.tagName
        };
        
        // Calculate movement metrics
        if (this.lastMousePosition) {
            const dx = e.clientX - this.lastMousePosition.x;
            const dy = e.clientY - this.lastMousePosition.y;
            const dt = Date.now() - this.lastMousePosition.timestamp;
            
            mouseEvent.velocity = Math.sqrt(dx * dx + dy * dy) / (dt || 1);
            mouseEvent.distance = Math.sqrt(dx * dx + dy * dy);
            mouseEvent.direction = Math.atan2(dy, dx);
        }
        
        this.lastMousePosition = {
            x: e.clientX,
            y: e.clientY,
            timestamp: Date.now()
        };
        
        this.behavioralBuffer.mouse.push(mouseEvent);
        
        // Send data in batches
        if (this.behavioralBuffer.mouse.length >= 50) {
            this.sendBehavioralData('mouse', this.behavioralBuffer.mouse.slice());
            this.behavioralBuffer.mouse = [];
        }
    }

    captureMouseClick(e) {
        if (!this.isMonitoring) return;
        
        const clickEvent = {
            type: 'click',
            button: e.button,
            x: e.clientX,
            y: e.clientY,
            timestamp: Date.now(),
            eventType: e.type,
            target: e.target.tagName
        };
        
        this.behavioralBuffer.mouse.push(clickEvent);
    }

    sendBehavioralData(type, data) {
        if (this.socket && data.length > 0) {
            this.socket.emit('behavioral_data', {
                type: type,
                events: data,
                timestamp: Date.now()
            });
        }
    }

    handleAuthResult(data) {
        this.currentAuthScore = data.authenticity_score || 0;
        this.confidenceLevel = (data.confidence || 0) * 100;
        this.anomalyRisk = this.getAnomalyRiskLevel(data.anomaly_score || 0);
        
        this.updateAuthenticationDisplay();
        this.addAuthDataToChart(data);
        
        // Update samples count
        if (this.keystrokeSamplesEl) {
            const current = parseInt(this.keystrokeSamplesEl.textContent) || 0;
            this.keystrokeSamplesEl.textContent = current + 1;
        }
    }

    handleSecurityAlert(data) {
        this.addSecurityAlert(data);
        
        // Show lockdown countdown for critical alerts
        if (data.level === 'critical') {
            showLockdownCountdown();
            this.updateNotificationBadge();
        }
        
        if (data.level === 'warning' || data.level === 'critical') {
            this.showSecurityAlertModal(data);
        }
        
        // Add to activity log
        this.addActivityItem({
            type: 'anomaly',
            message: data.message,
            risk: this.getAlertLevelText(data.level === 'critical' ? 3 : 2),
            timestamp: new Date().toISOString()
        });
    }

    handleDriftAnalysis(data) {
        if (data.drift_detected) {
            this.addSecurityAlert({
                level: 1,
                message: 'Behavioral drift detected',
                confidence: 0.8,
                recommendations: data.feature_analysis ? ['Monitor behavior changes'] : []
            });
        }
        
        // Update drift chart
        if (this.driftChart && data.feature_analysis) {
            this.updateDriftChart(data.feature_analysis);
        }
    }

    updateAuthenticationDisplay() {
        // Update auth score
        if (this.authScoreEl) {
            this.authScoreEl.textContent = this.currentAuthScore.toFixed(2);
        }
        
        // Update confidence level
        if (this.confidenceLevelEl) {
            this.confidenceLevelEl.textContent = Math.round(this.confidenceLevel) + '%';
        }
        
        // Update anomaly risk
        if (this.anomalyRiskEl) {
            this.anomalyRiskEl.textContent = this.anomalyRisk;
            this.anomalyRiskEl.className = 'metric-value ' + this.anomalyRisk.toLowerCase();
        }
        
        // Update security score
        const newSecurityScore = Math.round(this.currentAuthScore * 100);
        if (this.securityScoreEl && newSecurityScore !== this.securityScore) {
            this.securityScore = newSecurityScore;
            this.securityScoreEl.textContent = this.securityScore;
            this.updateSecurityScoreCircle();
        }
        
        // Update status indicator
        this.updateStatusIndicator();
    }

    updateSecurityScoreCircle() {
        if (this.securityScoreCircle) {
            this.securityScoreCircle.style.setProperty('--score', this.securityScore);
        }
    }

    updateStatusIndicator() {
        const indicator = this.statusIndicator.querySelector('.status-dot');
        
        if (this.currentAuthScore >= 0.8) {
            indicator.style.background = 'var(--secondary-color)';
            this.statusText.textContent = 'Secure';
        } else if (this.currentAuthScore >= 0.6) {
            indicator.style.background = 'var(--accent-color)';
            this.statusText.textContent = 'Monitoring';
        } else {
            indicator.style.background = 'var(--danger-color)';
            this.statusText.textContent = 'Alert';
        }
    }

    addAuthDataToChart(data) {
        if (this.behaviorChart) {
            const chart = this.behaviorChart;
            const now = new Date().toLocaleTimeString();
            
            chart.data.labels.push(now);
            chart.data.datasets[0].data.push(data.authenticity_score);
            
            // Keep only last 20 data points
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }
            
            chart.update('none');
        }
        
        if (this.timeChart) {
            const chart = this.timeChart;
            const now = new Date().toLocaleTimeString();
            
            chart.data.labels.push(now);
            chart.data.datasets[0].data.push(data.authenticity_score);
            chart.data.datasets[1].data.push(data.anomaly_score);
            
            // Keep only last 30 data points
            if (chart.data.labels.length > 30) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
                chart.data.datasets[1].data.shift();
            }
            
            chart.update('none');
        }
    }

    updateDriftChart(featureAnalysis) {
        if (!this.driftChart || !featureAnalysis) return;
        
        const keystrokes = featureAnalysis.keystroke || {};
        const mouse = featureAnalysis.mouse || {};
        
        const driftScores = [
            keystrokes.typing_speed_wpm?.drift_score || 0,
            keystrokes.hold_time_mean?.drift_score || 0,
            mouse.velocity_mean?.drift_score || 0,
            mouse.click_duration_mean?.drift_score || 0,
            mouse.movement_efficiency?.drift_score || 0
        ];
        
        this.driftChart.data.datasets[0].data = driftScores;
        this.driftChart.update();
    }

    addSecurityAlert(alert) {
        this.securityAlerts.unshift({
            ...alert,
            id: Date.now(),
            timestamp: new Date(),
            read: false
        });
        
        // Keep only last 50 alerts
        if (this.securityAlerts.length > 50) {
            this.securityAlerts = this.securityAlerts.slice(0, 50);
        }
        
        this.updateNotificationDropdown();
    }

    updateNotificationBadge() {
        const unreadCount = this.securityAlerts.filter(alert => !alert.read).length;
        this.notificationCount = unreadCount;
        
        if (unreadCount > 0) {
            this.notificationBadge.textContent = unreadCount;
            this.notificationBadge.style.display = 'block';
        } else {
            this.notificationBadge.style.display = 'none';
        }
    }

    updateNotificationDropdown() {
        if (this.securityAlerts.length === 0) {
            this.notificationList.innerHTML = `
                <div class="no-notifications">
                    <i class="fas fa-check-circle"></i>
                    <p>No alerts - all systems secure</p>
                </div>
            `;
        } else {
            this.notificationList.innerHTML = this.securityAlerts.slice(0, 10).map(alert => `
                <div class="notification-item ${alert.read ? 'read' : 'unread'}" data-id="${alert.id}">
                    <div class="notification-icon ${this.getAlertLevelClass(alert.level)}">
                        <i class="fas ${this.getAlertIcon(alert.level)}"></i>
                    </div>
                    <div class="notification-content">
                        <div class="notification-title">${alert.message}</div>
                        <div class="notification-time">${this.formatRelativeTime(alert.timestamp)}</div>
                    </div>
                </div>
            `).join('');
        }
        
        this.updateNotificationBadge();
    }

    showSecurityAlertModal(alert) {
        this.alertTitle.textContent = `Security Alert - Level ${alert.level}`;
        this.alertMessage.textContent = alert.message;
        
        const details = `
            <div class="alert-detail-item">
                <strong>Confidence:</strong> ${Math.round((alert.confidence || 0) * 100)}%
            </div>
            <div class="alert-detail-item">
                <strong>Time:</strong> ${new Date().toLocaleString()}
            </div>
            ${alert.recommendations ? `
                <div class="alert-detail-item">
                    <strong>Recommendations:</strong>
                    <ul>
                        ${alert.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
        `;
        
        this.alertDetails.innerHTML = details;
        this.securityAlertModal.style.display = 'flex';
    }

    showSection(sectionId) {
        // Update navigation
        this.navLinks.forEach(link => {
            link.parentElement.classList.remove('active');
        });
        
        const activeLink = document.querySelector(`[data-section="${sectionId}"]`);
        if (activeLink) {
            activeLink.parentElement.classList.add('active');
        }
        
        // Show content section
        this.contentSections.forEach(section => {
            section.classList.remove('active');
        });
        
        const activeSection = document.getElementById(`${sectionId}Section`);
        if (activeSection) {
            activeSection.classList.add('active');
        }
        
        // Update page title
        const titles = {
            dashboard: 'Dashboard',
            security: 'Security Status',
            analytics: 'Behavioral Analytics',
            activity: 'Activity Log',
            settings: 'Settings'
        };
        
        this.pageTitle.textContent = titles[sectionId] || 'Dashboard';
        this.currentSection = sectionId;
        
        // Load section-specific data
        switch (sectionId) {
            case 'activity':
                this.loadActivityLog();
                break;
            case 'analytics':
                this.loadAnalyticsData();
                break;
            case 'security':
                this.loadSecurityData();
                break;
        }
    }

    toggleNotificationDropdown() {
        const isVisible = this.notificationDropdown.style.display === 'block';
        this.notificationDropdown.style.display = isVisible ? 'none' : 'block';
    }

    markAllNotificationsRead() {
        this.securityAlerts.forEach(alert => alert.read = true);
        this.updateNotificationDropdown();
        this.updateNotificationBadge();
    }

    acknowledgeSecurityAlert() {
        this.securityAlertModal.style.display = 'none';
        // Mark current alert as acknowledged
    }

    investigateSecurityAlert() {
        this.securityAlertModal.style.display = 'none';
        this.showSection('security');
    }

    toggleTestArea() {
        const isVisible = this.testArea.style.display === 'block';
        this.testArea.style.display = isVisible ? 'none' : 'block';
    }

    startBehaviorTest() {
        this.isMonitoring = true;
        this.startTest.disabled = true;
        this.stopTest.disabled = false;
        
        // Clear previous data
        this.behavioralBuffer.keystroke = [];
        this.behavioralBuffer.mouse = [];
        
        console.log('Behavior test started');
    }

    stopBehaviorTest() {
        this.isMonitoring = false;
        this.startTest.disabled = false;
        this.stopTest.disabled = true;
        
        console.log('Behavior test stopped');
    }

    updateConnectionStatus(connected) {
        const statusDot = this.statusIndicator.querySelector('.status-dot');
        if (connected) {
            statusDot.style.background = 'var(--secondary-color)';
            statusDot.style.animation = 'pulse 2s infinite';
        } else {
            statusDot.style.background = 'var(--danger-color)';
            statusDot.style.animation = 'none';
        }
    }

    updateMonitoringStatus(active) {
        const monitorStatus = this.monitorStatus;
        if (monitorStatus) {
            const statusDot = monitorStatus.querySelector('.status-dot');
            const statusText = monitorStatus.querySelector('span');
            
            if (active) {
                statusDot.classList.add('active');
                statusText.textContent = 'Active';
            } else {
                statusDot.classList.remove('active');
                statusText.textContent = 'Inactive';
            }
        }
    }

    updateUserInfo() {
        if (this.sidebarUsername) {
            this.sidebarUsername.textContent = this.username || 'User';
        }
    }

    loadActivityLog() {
        // Simulate loading activity log
        const activities = [
            {
                time: '2 minutes ago',
                type: 'login',
                description: 'User authentication successful',
                risk: 'Low'
            },
            {
                time: '15 minutes ago',
                type: 'anomaly',
                description: 'Unusual typing pattern detected',
                risk: 'Medium'
            },
            {
                time: '1 hour ago',
                type: 'drift',
                description: 'Behavioral drift monitoring started',
                risk: 'Low'
            }
        ];
        
        if (this.activityTableBody) {
            this.activityTableBody.innerHTML = activities.map(activity => `
                <tr>
                    <td>${activity.time}</td>
                    <td>${activity.type}</td>
                    <td>${activity.description}</td>
                    <td><span class="risk-level ${activity.risk.toLowerCase()}">${activity.risk}</span></td>
                    <td><button class="view-details-btn">View</button></td>
                </tr>
            `).join('');
        }
    }

    updateCharts() {
        // Add random data for demonstration
        if (this.patternsChart) {
            const newKeystroke = {
                x: Math.random() * 100,
                y: Math.random() * 100
            };
            const newMouse = {
                x: Math.random() * 100,
                y: Math.random() * 100
            };
            
            this.patternsChart.data.datasets[0].data.push(newKeystroke);
            this.patternsChart.data.datasets[1].data.push(newMouse);
            
            // Keep only last 50 points
            if (this.patternsChart.data.datasets[0].data.length > 50) {
                this.patternsChart.data.datasets[0].data.shift();
                this.patternsChart.data.datasets[1].data.shift();
            }
            
            this.patternsChart.update('none');
        }
    }

    updateGeneralStats() {
        // Update mouse samples count
        if (this.mouseSamplesEl) {
            const current = parseInt(this.mouseSamplesEl.textContent) || 0;
            this.mouseSamplesEl.textContent = current + Math.floor(Math.random() * 5);
        }
    }

    // Utility methods
    getAnomalyRiskLevel(score) {
        if (score < 0.3) return 'Low';
        if (score < 0.7) return 'Medium';
        return 'High';
    }

    getAlertLevelText(level) {
        const levels = ['Info', 'Low', 'Medium', 'High', 'Critical'];
        return levels[level] || 'Unknown';
    }

    getAlertLevelClass(level) {
        const classes = ['info', 'low', 'medium', 'high', 'critical'];
        return classes[level] || 'info';
    }

    getAlertIcon(level) {
        const icons = ['fa-info', 'fa-exclamation', 'fa-exclamation-triangle', 'fa-exclamation-circle', 'fa-times-circle'];
        return icons[level] || 'fa-info';
    }

    formatRelativeTime(timestamp) {
        const now = new Date();
        const diff = now - timestamp;
        const minutes = Math.floor(diff / 60000);
        
        if (minutes < 1) return 'Just now';
        if (minutes < 60) return `${minutes}m ago`;
        
        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours}h ago`;
        
        const days = Math.floor(hours / 24);
        return `${days}d ago`;
    }

    addActivityItem(activity) {
        if (this.recentActivityList) {
            const item = document.createElement('div');
            item.className = 'activity-item';
            item.innerHTML = `
                <div class="activity-icon ${activity.type}">
                    <i class="fas ${this.getActivityIcon(activity.type)}"></i>
                </div>
                <div class="activity-details">
                    <span class="activity-text">${activity.message}</span>
                    <span class="activity-time">${this.formatRelativeTime(new Date(activity.timestamp))}</span>
                </div>
            `;
            
            this.recentActivityList.insertBefore(item, this.recentActivityList.firstChild);
            
            // Keep only last 5 items
            while (this.recentActivityList.children.length > 5) {
                this.recentActivityList.removeChild(this.recentActivityList.lastChild);
            }
        }
    }

    getActivityIcon(type) {
        const icons = {
            login: 'fa-sign-in-alt',
            logout: 'fa-sign-out-alt',
            anomaly: 'fa-exclamation-triangle',
            drift: 'fa-chart-line',
            success: 'fa-check'
        };
        return icons[type] || 'fa-info';
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
            // Clean up intervals
            if (this.authUpdateInterval) clearInterval(this.authUpdateInterval);
            if (this.chartUpdateInterval) clearInterval(this.chartUpdateInterval);
            if (this.statsUpdateInterval) clearInterval(this.statsUpdateInterval);
            
            // Clear local storage and redirect
            localStorage.clear();
            window.location.href = '/login';
        }
    }

    // Placeholder methods for quick actions
    runSecurityCheck() {
        console.log('Running security check...');
        this.addActivityItem({
            type: 'success',
            message: 'Security check completed',
            timestamp: new Date().toISOString()
        });
    }

    updateModels() {
        console.log('Updating behavioral models...');
        this.addActivityItem({
            type: 'success',
            message: 'Behavioral models updated',
            timestamp: new Date().toISOString()
        });
    }

    exportLogs() {
        console.log('Exporting activity logs...');
        // Simulate file download
        const data = 'Activity logs exported at ' + new Date().toISOString();
        const blob = new Blob([data], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'activity-logs.txt';
        a.click();
        URL.revokeObjectURL(url);
    }

    updateSettings() {
        console.log('Settings updated');
    }

    loadAnalyticsData() {
        console.log('Loading analytics data...');
    }

    loadSecurityData() {
        console.log('Loading security data...');
    }
}

// Initialize dashboard manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check authentication
    const sessionId = localStorage.getItem('session_id');
    if (!sessionId) {
        window.location.href = '/login';
        return;
    }
    
    const dashboardManager = new DashboardManager();
    
    // Handle responsive sidebar
    const handleResize = () => {
        if (window.innerWidth <= 768) {
            dashboardManager.sidebar.classList.remove('open');
        }
    };
    
    window.addEventListener('resize', handleResize);
    handleResize();
    
    console.log('🛡️ Behavioral Authentication Dashboard Initialized');
});

// Export for potential use in other modules
window.DashboardManager = DashboardManager;

document.addEventListener('DOMContentLoaded', () => {
    const socket = io();  // Connect to Flask-SocketIO
    const sessionId = localStorage.getItem('session_id');
    const SLIDING_WINDOW_MS = 30000;
    const AUTH_INTERVAL_MS = 5000;

    let keystrokeEvents = [];
    let mouseEvents = [];

    // Emit session join to backend
    socket.emit('join_session', { session_id: sessionId });

    // Listen for input and mouse activity
    document.addEventListener('keydown', e => {
        keystrokeEvents.push({
            key: e.key,
            type: 'keydown',
            timestamp: performance.now()
        });
    });

    document.addEventListener('keyup', e => {
        keystrokeEvents.push({
            key: e.key,
            type: 'keyup',
            timestamp: performance.now()
        });
    });

    document.addEventListener('mousemove', e => {
        mouseEvents.push({
            type: 'move',
            x: e.clientX,
            y: e.clientY,
            timestamp: performance.now()
        });
    });

    document.addEventListener('mousedown', e => {
        mouseEvents.push({
            type: 'mousedown',
            x: e.clientX,
            y: e.clientY,
            timestamp: performance.now()
        });
    });

    document.addEventListener('mouseup', e => {
        mouseEvents.push({
            type: 'mouseup',
            x: e.clientX,
            y: e.clientY,
            timestamp: performance.now()
        });
    });

    setInterval(() => {
        const now = performance.now();
        const start = now - SLIDING_WINDOW_MS;

        const recentKeystrokes = keystrokeEvents.filter(e => e.timestamp >= start);
        const recentMouse = mouseEvents.filter(e => e.timestamp >= start);

        if (recentKeystrokes.length > 5) {
            socket.emit('behavioral_data', {
                type: 'keystroke',
                events: recentKeystrokes,
                timestamp: Date.now()
            });
        }

        if (recentMouse.length > 10) {
            socket.emit('behavioral_data', {
                type: 'mouse',
                events: recentMouse,
                timestamp: Date.now()
            });
        }

        keystrokeEvents = recentKeystrokes;
        mouseEvents = recentMouse;
    }, AUTH_INTERVAL_MS);

    socket.on('auth_result', data => {
        const authScore = data.authenticity_score || 0;
        const confidence = data.confidence || 0;
        const anomaly = data.anomaly_score || 0;

        document.getElementById('authScore').textContent = authScore.toFixed(2);
        document.getElementById('confidenceLevel').textContent = (confidence * 100).toFixed(1) + '%';
        document.getElementById('anomalyRisk').textContent =
            anomaly < 0.3 ? 'Low' : anomaly < 0.7 ? 'Medium' : 'High';

        const statusText = document.getElementById('statusText');
        const statusDot = document.getElementById('statusIndicator').querySelector('.status-dot');

        if (authScore >= 0.8) {
            statusText.textContent = 'Authenticated';
            statusDot.className = 'status-dot green';
        } else if (authScore >= 0.6 || anomaly < 0.5) {
            statusText.textContent = 'Risk Detected';
            statusDot.className = 'status-dot yellow';
        } else {
            statusText.textContent = 'High Risk';
            statusDot.className = 'status-dot red';
        }
    });
});

window.addEventListener('load', function() {
    var typingTimeout;
    document.addEventListener('keyup', function() {
        clearTimeout(typingTimeout);
        typingTimeout = setTimeout(function() {
            if (typeof sendBehavioralDataForAuth === 'function') {
                if (keystrokeBuffer && keystrokeBuffer.length >= 3) {
                    sendBehavioralDataForAuth();
                }
            }
        }, 1500);
    });
    
    setTimeout(function() {
        if (typeof sendBehavioralDataForAuth === 'function') {
            setInterval(sendBehavioralDataForAuth, 5000);
        } else {
            console.error('sendBehavioralDataForAuth not found');
        }
    }, 2000);
});
