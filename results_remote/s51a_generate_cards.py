"""
S5.1a Case Card Generator: Generate self-contained HTML for clinician review.

Each card presents patient context + two blinded treatment plans (A, B).
Clinicians rate each plan via radio buttons, then click "Download CSV"
to export all ratings. Everything is client-side — no server needed.

Input: s51a_selected_cases.json
Output: s51a_case_cards.html (single self-contained file)
"""

import json
from pathlib import Path

INPUT_PATH = Path("s51a_selected_cases.json")
OUTPUT_PATH = Path("s51a_case_cards.html")


def fmt_treatment_table(plan, tau):
    """Format a treatment sequence as an HTML table."""
    seq = plan['treatment_sequence']
    rows = ""
    for step in range(tau):
        vaso = seq[step][0]
        vent = seq[step][1]
        vaso_str = f"{vaso:.2f}" if vaso > 0.01 else "None"
        vent_str = "Yes" if vent > 0.5 else "No"
        rows += f"<tr><td>Hour {step+1}</td><td>{vaso_str}</td><td>{vent_str}</td></tr>\n"
    return f"""
    <table class="treatment-table">
        <thead><tr><th>Step</th><th>Vasopressor</th><th>Ventilation</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """


def fmt_dbp_bar(value):
    """Format a DBP value with neutral visual bar (no color coding to avoid bias)."""
    width = min(max(value / 120 * 100, 5), 100)
    return f"""
    <div class="dbp-bar-container">
        <span class="dbp-label">{value:.1f} mmHg</span>
        <div class="dbp-bar" style="width:{width:.0f}%;background:#78909c;"></div>
    </div>
    """


def generate_case_card(case, tau):
    case_id = case['case_id']
    obs_dbp = case['observed_dbp']
    n_feas = case['n_feasible']
    plan_a = case['plan_a']
    plan_b = case['plan_b']

    return f"""
    <div class="case-card" id="{case_id}">
        <div class="case-header">
            <h2>Case {case_id}</h2>
            <span class="completion-badge" id="badge_{case_id}">Incomplete</span>
        </div>

        <div class="patient-context">
            <h3>Patient Context</h3>
            <table class="context-table">
                <tr><td><strong>Setting:</strong></td><td>ICU patient, MIMIC-III database</td></tr>
                <tr><td><strong>Current diastolic BP:</strong></td><td>{obs_dbp:.0f} mmHg</td></tr>
                <tr><td><strong>Target DBP range:</strong></td><td>60–90 mmHg (hemodynamic stability)</td></tr>
                <tr><td><strong>Safety bounds:</strong></td><td>40–120 mmHg (avoid shock / hypertensive crisis)</td></tr>
                <tr><td><strong>Planning horizon:</strong></td><td>{tau} hours ahead</td></tr>
                <tr><td><strong>Feasible plans available:</strong></td><td>{n_feas}/100 candidates meet safety criteria</td></tr>
            </table>
            <p class="context-note">Two AI-recommended treatment plans are shown below.
            Vasopressor values are normalized (0 = none, 1 = maximum observed dose).
            Please rate each plan independently.</p>
        </div>

        <div class="plans-container">
            <div class="plan-column">
                <h3>Plan A</h3>
                {fmt_treatment_table(plan_a, tau)}
                <div class="predicted-outcome">
                    <strong>Predicted terminal DBP:</strong>
                    {fmt_dbp_bar(plan_a['dbp_terminal'])}
                </div>

                <div class="rating-box">
                    <p><strong>Rate Plan A:</strong></p>
                    <div class="likert-scale">
                        <label><input type="radio" name="{case_id}_a" value="1" onchange="updateProgress()"> 1 – Dangerous</label>
                        <label><input type="radio" name="{case_id}_a" value="2" onchange="updateProgress()"> 2 – Questionable</label>
                        <label><input type="radio" name="{case_id}_a" value="3" onchange="updateProgress()"> 3 – Acceptable</label>
                        <label><input type="radio" name="{case_id}_a" value="4" onchange="updateProgress()"> 4 – Good</label>
                        <label><input type="radio" name="{case_id}_a" value="5" onchange="updateProgress()"> 5 – Excellent</label>
                    </div>
                </div>
            </div>

            <div class="plan-column">
                <h3>Plan B</h3>
                {fmt_treatment_table(plan_b, tau)}
                <div class="predicted-outcome">
                    <strong>Predicted terminal DBP:</strong>
                    {fmt_dbp_bar(plan_b['dbp_terminal'])}
                </div>

                <div class="rating-box">
                    <p><strong>Rate Plan B:</strong></p>
                    <div class="likert-scale">
                        <label><input type="radio" name="{case_id}_b" value="1" onchange="updateProgress()"> 1 – Dangerous</label>
                        <label><input type="radio" name="{case_id}_b" value="2" onchange="updateProgress()"> 2 – Questionable</label>
                        <label><input type="radio" name="{case_id}_b" value="3" onchange="updateProgress()"> 3 – Acceptable</label>
                        <label><input type="radio" name="{case_id}_b" value="4" onchange="updateProgress()"> 4 – Good</label>
                        <label><input type="radio" name="{case_id}_b" value="5" onchange="updateProgress()"> 5 – Excellent</label>
                    </div>
                </div>
            </div>
        </div>

        <div class="preference-box">
            <p><strong>Overall preference:</strong></p>
            <label><input type="radio" name="{case_id}_pref" value="A" onchange="updateProgress()"> Prefer Plan A</label>
            <label><input type="radio" name="{case_id}_pref" value="B" onchange="updateProgress()"> Prefer Plan B</label>
            <label><input type="radio" name="{case_id}_pref" value="N" onchange="updateProgress()"> No preference</label>
        </div>

        <div class="comments-box">
            <p><strong>Comments (optional):</strong></p>
            <textarea id="{case_id}_comment" rows="2"></textarea>
        </div>
    </div>
    """


def generate_css():
    return """
<style>
    @media print { .case-card { page-break-after: always; } .no-print { display: none; } }
    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; color: #333; }
    h1 { text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; }
    .case-card {
        border: 2px solid #666; border-radius: 8px; padding: 20px; margin: 20px 0;
        background: #fafafa;
    }
    .case-header { display: flex; justify-content: space-between; align-items: center;
        border-bottom: 1px solid #aaa; margin-bottom: 10px; padding-bottom: 5px; }
    .case-header h2 { margin: 0; color: #1a5276; }
    .completion-badge {
        padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: bold;
    }
    .completion-badge.incomplete { background: #ffcdd2; color: #c62828; }
    .completion-badge.complete { background: #c8e6c9; color: #2e7d32; }
    .patient-context { background: #eaf2f8; padding: 12px; border-radius: 6px; margin-bottom: 15px; }
    .patient-context h3 { margin-top: 0; }
    .context-table td { padding: 3px 10px; }
    .context-note { font-style: italic; color: #555; font-size: 0.9em; }
    .plans-container { display: flex; gap: 20px; }
    .plan-column { flex: 1; background: white; border: 1px solid #ccc; border-radius: 6px; padding: 12px; }
    .plan-column h3 { text-align: center; margin-top: 0; color: #2c3e50; }
    .treatment-table { width: 100%; border-collapse: collapse; margin: 8px 0; }
    .treatment-table th, .treatment-table td { border: 1px solid #ddd; padding: 6px; text-align: center; }
    .treatment-table th { background: #ecf0f1; }
    .predicted-outcome { margin: 10px 0; padding: 8px; background: #f9f9f9; border-radius: 4px; }
    .dbp-bar-container { margin: 5px 0; }
    .dbp-bar { height: 18px; border-radius: 3px; margin-top: 3px; }
    .dbp-label { font-size: 0.95em; }
    .rating-box { margin: 10px 0; padding: 10px; background: #fff8e1; border: 1px solid #f0e68c; border-radius: 4px; }
    .likert-scale label { display: block; margin: 4px 0; cursor: pointer; }
    .preference-box { margin: 15px 0; padding: 10px; background: #e8f5e9; border-radius: 4px; }
    .preference-box label { margin-right: 20px; cursor: pointer; }
    .comments-box { margin: 10px 0; }
    .comments-box textarea { width: 100%; box-sizing: border-box; }
    .instructions { background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; padding: 15px; margin: 20px 0; }
    .instructions h3 { margin-top: 0; }
    .summary-stats { background: #e3f2fd; padding: 12px; border-radius: 6px; margin: 20px 0; }

    /* Progress bar + download section */
    .toolbar {
        position: sticky; top: 0; z-index: 100; background: white;
        border-bottom: 2px solid #333; padding: 10px 20px; margin: -20px -20px 20px -20px;
        display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
    }
    .progress-container { flex: 1; min-width: 200px; }
    .progress-bar-bg { background: #e0e0e0; border-radius: 8px; height: 20px; overflow: hidden; }
    .progress-bar-fill { background: #4CAF50; height: 100%; border-radius: 8px; transition: width 0.3s; }
    .progress-text { font-size: 0.9em; margin-top: 3px; }
    .download-btn {
        padding: 10px 24px; font-size: 1em; font-weight: bold; cursor: pointer;
        border: none; border-radius: 6px; color: white;
    }
    .download-btn.ready { background: #2196F3; }
    .download-btn.ready:hover { background: #1976D2; }
    .download-btn.disabled { background: #bdbdbd; cursor: not-allowed; }
    .save-btn { background: #ff9800; }
    .save-btn:hover { background: #f57c00; }
</style>
"""


def generate_javascript(case_ids):
    """Generate client-side JS for progress tracking and CSV export."""
    ids_json = json.dumps(case_ids)
    return f"""
<script>
const CASE_IDS = {ids_json};

function getRadioValue(name) {{
    const el = document.querySelector('input[name="' + name + '"]:checked');
    return el ? el.value : '';
}}

function updateProgress() {{
    let completed = 0;
    CASE_IDS.forEach(function(cid) {{
        const a = getRadioValue(cid + '_a');
        const b = getRadioValue(cid + '_b');
        const p = getRadioValue(cid + '_pref');
        const badge = document.getElementById('badge_' + cid);
        if (a && b && p) {{
            completed++;
            badge.textContent = 'Complete';
            badge.className = 'completion-badge complete';
        }} else {{
            badge.textContent = 'Incomplete';
            badge.className = 'completion-badge incomplete';
        }}
    }});

    const pct = Math.round(100 * completed / CASE_IDS.length);
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('progress-text').textContent = completed + '/' + CASE_IDS.length + ' cases completed (' + pct + '%)';

    const btn = document.getElementById('download-btn');
    if (completed === CASE_IDS.length) {{
        btn.className = 'download-btn ready';
        btn.disabled = false;
        btn.textContent = 'Download Ratings CSV';
    }} else {{
        btn.className = 'download-btn ready';
        btn.disabled = false;
        btn.textContent = 'Download CSV (' + completed + '/' + CASE_IDS.length + ')';
    }}
}}

function downloadCSV() {{
    let rows = ['case_id,plan_a_likert,plan_b_likert,preference,comments'];
    CASE_IDS.forEach(function(cid) {{
        const a = getRadioValue(cid + '_a');
        const b = getRadioValue(cid + '_b');
        const p = getRadioValue(cid + '_pref');
        const commentEl = document.getElementById(cid + '_comment');
        const comment = commentEl ? commentEl.value.replace(/"/g, '""').replace(/\\n/g, ' ') : '';
        rows.push(cid + ',' + a + ',' + b + ',' + p + ',"' + comment + '"');
    }});

    const csv = rows.join('\\n');
    const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 's51a_rating_form.csv';
    a.click();
    URL.revokeObjectURL(url);
}}

// Auto-save to localStorage
function saveToLocalStorage() {{
    const state = {{}};
    CASE_IDS.forEach(function(cid) {{
        state[cid + '_a'] = getRadioValue(cid + '_a');
        state[cid + '_b'] = getRadioValue(cid + '_b');
        state[cid + '_pref'] = getRadioValue(cid + '_pref');
        const commentEl = document.getElementById(cid + '_comment');
        state[cid + '_comment'] = commentEl ? commentEl.value : '';
    }});
    localStorage.setItem('s51a_ratings', JSON.stringify(state));
    document.getElementById('save-status').textContent = 'Saved at ' + new Date().toLocaleTimeString();
}}

function loadFromLocalStorage() {{
    const saved = localStorage.getItem('s51a_ratings');
    if (!saved) return;
    const state = JSON.parse(saved);
    Object.keys(state).forEach(function(key) {{
        if (key.endsWith('_comment')) {{
            const el = document.getElementById(key);
            if (el) el.value = state[key];
        }} else if (state[key]) {{
            const el = document.querySelector('input[name="' + key + '"][value="' + state[key] + '"]');
            if (el) el.checked = true;
        }}
    }});
    updateProgress();
}}

// Auto-save every 30 seconds
setInterval(saveToLocalStorage, 30000);

// Also save on every rating change
document.addEventListener('change', function(e) {{
    if (e.target.type === 'radio' || e.target.tagName === 'TEXTAREA') {{
        saveToLocalStorage();
    }}
}});

// Load saved state on page load
window.addEventListener('load', function() {{
    loadFromLocalStorage();
    updateProgress();
}});
</script>
"""


def generate_instructions(n_cases):
    return f"""
    <div class="instructions">
        <h3>Instructions for Clinician Reviewers</h3>
        <p>Thank you for participating in this evaluation of AI-recommended treatment plans for ICU patients.</p>
        <ul>
            <li><strong>Context:</strong> Each case presents an ICU patient from the MIMIC-III database with their current diastolic blood pressure and two AI-recommended treatment plans for the next 3 hours.</li>
            <li><strong>Treatments:</strong> Vasopressor dose (normalized 0–1) and ventilation (Yes/No) at each hourly step.</li>
            <li><strong>Your task:</strong> For each case, rate both plans independently on a 1–5 scale, then indicate your overall preference.</li>
            <li><strong>Scale:</strong>
                <ol>
                    <li><strong>Dangerous</strong> – would likely cause harm</li>
                    <li><strong>Questionable</strong> – significant clinical concerns</li>
                    <li><strong>Acceptable</strong> – reasonable but not preferred</li>
                    <li><strong>Good</strong> – clinically sound</li>
                    <li><strong>Excellent</strong> – matches best practice</li>
                </ol>
            </li>
            <li><strong>Blinding:</strong> Plans A and B are randomly assigned. You do not know which method generated each plan.</li>
            <li><strong>Saving:</strong> Your progress is auto-saved in the browser. You can close the page and return later — your ratings will be restored.</li>
            <li><strong>When done:</strong> Click the <strong>"Download CSV"</strong> button at the top to export your ratings, then email the CSV file back.</li>
            <li><strong>Time:</strong> ~1–2 minutes per case, {n_cases} cases total (~30 minutes).</li>
        </ul>
    </div>
    """


def main():
    with open(INPUT_PATH) as f:
        data = json.load(f)

    tau = data['config']['tau']
    cases = data['cases']
    case_ids = [c['case_id'] for c in cases]

    print(f"Generating cards for {len(cases)} cases at τ={tau}")

    cards_html = ""
    for case in cases:
        cards_html += generate_case_card(case, tau)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>S5.1a Clinician Plausibility Assessment</title>
    {generate_css()}
</head>
<body>
    <h1>Clinician Plausibility Assessment: AI Treatment Plans for ICU Patients</h1>

    <div class="toolbar no-print">
        <div class="progress-container">
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" id="progress-fill" style="width:0%"></div>
            </div>
            <div class="progress-text" id="progress-text">0/{len(cases)} cases completed (0%)</div>
        </div>
        <button class="download-btn disabled" id="download-btn" onclick="downloadCSV()" disabled>
            Download CSV (0/{len(cases)})
        </button>
        <button class="download-btn save-btn" onclick="saveToLocalStorage()">Save Progress</button>
        <span id="save-status" style="font-size:0.8em;color:#666;"></span>
    </div>

    {generate_instructions(len(cases))}

    {cards_html}

    <div style="text-align:center; margin:30px; padding:20px; background:#e8f5e9; border-radius:8px;">
        <p style="font-size:1.2em;"><strong>All done?</strong></p>
        <button class="download-btn ready" onclick="downloadCSV()" style="font-size:1.1em; padding:12px 32px;">
            Download Ratings CSV
        </button>
        <p style="margin-top:10px; color:#555;">Please email the downloaded <code>s51a_rating_form.csv</code> file back to the researcher.</p>
    </div>

    {generate_javascript(case_ids)}
</body>
</html>"""

    with open(OUTPUT_PATH, 'w') as f:
        f.write(html)

    print(f"Generated {OUTPUT_PATH} ({len(html)//1024} KB)")


if __name__ == '__main__':
    main()
