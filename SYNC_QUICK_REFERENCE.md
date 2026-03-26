# Quick Reference: Synchronization Fixes

## 1️⃣ Backend Emit (app.py) - Lines 895-911

```python
# NEW: Emit auth_update for synchronized UI updates
keystroke_count = len(behavioral_buffers[user_id]['keystrokes'])
mouse_count = len(behavioral_buffers[user_id]['mouse'])
anomaly_risk_text = 'High' if not is_authorized else 'Low'

emit('auth_update', {
    'auth_score': float(score),
    'confidence': float(model_confidence),
    'anomaly_risk': anomaly_risk_text,
    'keystroke_count': keystroke_count,
    'mouse_count': mouse_count,
    'strikes': int(current_strikes),
    'is_locked': current_strikes >= 3
}, room=request.sid)
```

**Location:** After line 893 (after auth_response emit)  
**Requires:** Variables already calculated:
- `score` (from auth_result)
- `model_confidence` (from auth_result)
- `is_authorized` (calculated from anomaly_detected)
- `current_strikes` (from consecutive_anomalies dict)

---

## 2️⃣ Frontend Listener (challenge.js) - Lines 463-541

```javascript
socket.on('auth_update', (data) => {
    console.log('UI received update:', data);
    
    // Update Authentication Score
    const authScoreVal = document.getElementById('auth-score-val');
    if (authScoreVal) {
        authScoreVal.innerText = (data.auth_score * 100).toFixed(1) + '%';
        console.log('[UI] Updated auth-score-val:', authScoreVal.innerText);
    }
    
    // Update Confidence Level
    const confLevelVal = document.getElementById('conf-level-val');
    if (confLevelVal) {
        confLevelVal.innerText = (data.confidence * 100).toFixed(1) + '%';
        console.log('[UI] Updated conf-level-val:', confLevelVal.innerText);
    }
    
    // Update Anomaly Risk
    const anomalyRiskVal = document.getElementById('anomaly-risk-val');
    if (anomalyRiskVal) {
        anomalyRiskVal.innerText = data.anomaly_risk;
        console.log('[UI] Updated anomaly-risk-val:', anomalyRiskVal.innerText);
    }
    
    // Update Keystroke Count
    const keyCount = document.getElementById('key-count');
    if (keyCount) {
        keyCount.innerText = data.keystroke_count;
        console.log('[UI] Updated key-count:', keyCount.innerText);
    }
    
    // Update Mouse Count
    const mouseCount = document.getElementById('mouse-count');
    if (mouseCount) {
        mouseCount.innerText = data.mouse_count;
        console.log('[UI] Updated mouse-count:', mouseCount.innerText);
    }
    
    // Check if locked (3+ strikes)
    if (data.is_locked === true) {
        console.log('[LOCKDOWN] User is locked, calling showLockScreen()');
        if (typeof showLockScreen === 'function') {
            showLockScreen();
        } else if (typeof showLockdownCountdown === 'function') {
            showLockdownCountdown();
        } else {
            console.warn('[LOCKDOWN] No lockdown function available');
        }
    }
    
    // Update Chart.js if available
    if (window.authChart) {
        try {
            window.authChart.data.datasets[0].data.push(data.auth_score);
            window.authChart.update();
            console.log('[CHART] Updated authChart with score:', data.auth_score);
        } catch (err) {
            console.warn('[CHART] Error updating authChart:', err.message);
        }
    } else {
        console.log('[CHART] authChart not yet available');
    }
});
```

**Location:** After socket.on('auth_response') listener  
**Socket Event Name:** `auth_update`  
**Data Fields Expected:**
- `data.auth_score` (decimal 0.0-1.0)
- `data.confidence` (decimal 0.0-1.0)
- `data.anomaly_risk` (string: "High" or "Low")
- `data.keystroke_count` (integer)
- `data.mouse_count` (integer)
- `data.is_locked` (boolean)

---

## 3️⃣ HTML Elements (challenge.html)

### Security Metrics Section (~Line 195)
```html
<div class="security-metrics">
    <div class="metric">
        <span class="metric-label">Authentication Score</span>
        <span class="metric-value" id="auth-score-val">0.0%</span>
    </div>
    <div class="metric">
        <span class="metric-label">Confidence Level</span>
        <span class="metric-value" id="conf-level-val">0%</span>
    </div>
    <div class="metric">
        <span class="metric-label">Anomaly Risk</span>
        <span class="metric-value" id="anomaly-risk-val">Low</span>
    </div>
</div>
```

### Monitor Stats Section (~Line 225)
```html
<div class="monitor-stats">
    <div class="stat">
        <i class="fas fa-keyboard"></i>
        <span class="stat-label">Keystroke Samples</span>
        <span class="stat-value" id="key-count">0</span>
    </div>
    <div class="stat">
        <i class="fas fa-mouse-pointer"></i>
        <span class="stat-label">Mouse Samples</span>
        <span class="stat-value" id="mouse-count">0</span>
    </div>
</div>
```

**Required IDs (exact spelling):**
- `auth-score-val` - displays auth score as percentage
- `conf-level-val` - displays confidence as percentage
- `anomaly-risk-val` - displays "High" or "Low"
- `key-count` - displays keystroke count
- `mouse-count` - displays mouse count

---

## Verification Commands

### Check Backend Emit
```bash
grep -n "auth_update" Behavior_based_Auth/app.py
# Should show: 2 results (emit line + payload line)
```

### Check Frontend Listener
```bash
grep -n "socket.on('auth_update'" Behavior_based_Auth/static/js/challenge.js
# Should show: 1 result
```

### Check HTML IDs
```bash
grep -n "auth-score-val\|conf-level-val\|anomaly-risk-val\|key-count\|mouse-count" Behavior_based_Auth/templates/challenge.html
# Should show: 5 results (one for each ID)
```

---

## Payload Examples

### Normal Authorized User
```json
{
  "auth_score": 0.15,
  "confidence": 0.85,
  "anomaly_risk": "Low",
  "keystroke_count": 105,
  "mouse_count": 84,
  "strikes": 0,
  "is_locked": false
}
```

**Result:** UI shows 15% anomaly score, 85% confidence, "Low" risk

### User Behaving Abnormally (Strike 1)
```json
{
  "auth_score": 0.65,
  "confidence": 0.92,
  "anomaly_risk": "High",
  "keystroke_count": 105,
  "mouse_count": 84,
  "strikes": 1,
  "is_locked": false
}
```

**Result:** UI shows 65% anomaly score, 92% confidence, "High" risk, Strike 1/3

### User Locked Out (3 Strikes)
```json
{
  "auth_score": 0.78,
  "confidence": 0.95,
  "anomaly_risk": "High",
  "keystroke_count": 105,
  "mouse_count": 84,
  "strikes": 3,
  "is_locked": true
}
```

**Result:** UI shows 78% anomaly score, lockdown countdownfunctionality triggers

---

## Testing in Browser Console

### Check if listener is registered
```javascript
// Should see messages like:
// "UI received update: {auth_score: 0.456, confidence: 0.78, ...}"
console.log('Look for "UI received update:" messages');
```

### Manually test DOM updates
```javascript
// Simulate what the listener does:
document.getElementById('auth-score-val').innerText = '45.6%';
document.getElementById('conf-level-val').innerText = '78.0%'; 
document.getElementById('anomaly-risk-val').innerText = 'Low';
document.getElementById('key-count').innerText = '105';
document.getElementById('mouse-count').innerText = '84';
```

### Check if Chart.js is available
```javascript
console.log(window.authChart ? "authChart exists" : "authChart not found");
```

---

## Debugging Checklist

- [ ] Backend emits 'auth_update' message (check Flask terminal)
- [ ] Browser receives message (check DevTools Network tab → WS)
- [ ] Console shows "UI received update:" message
- [ ] All 5 DOM elements update with values > 0
- [ ] Anomaly risk shows "High" or "Low" (not number)
- [ ] Character counts match keystroke_count and mouse_count
- [ ] Authentication score shows as percentage (not decimal)
- [ ] At 3 strikes, lockdown function is called
- [ ] Chart updates if window.authChart exists

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `Cannot read property 'innerText'` | Element ID doesn't exist | Verify HTML has exact ID |
| `socket.on is not a function` | Socket.IO not initialized | Check script include in HTML |
| UI shows 0% | Data not arriving | Check browser Network tab |
| Lockdown not showing | Function name mismatch | Use showLockScreen() or showLockdownCountdown() |
| Chart error | authChart undefined | Ensure Chart.js initialized before listener |

---

## Files Changed Summary

```
✅ app.py          → Added auth_update emit (17 lines)
✅ challenge.js    → Added socket listener (79 lines)
✅ challenge.html  → Updated 5 element IDs (changed id names only)

Total new code: ~96 lines
Total changes: 3 files
Status: COMPLETE
```

---

**Ready for Testing** ✅
