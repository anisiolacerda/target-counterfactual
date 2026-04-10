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
CONTEXT_PATH = Path("s51a_patient_context.json")
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


def fmt_val(v, unit=""):
    """Format a value, handling None."""
    if v is None:
        return "—"
    return f"{v}{unit}"


def fmt_age(age):
    """Format MIMIC age (300 = >89 years, anonymized)."""
    if age is None or age < 0:
        return "Unknown"
    if age >= 300:
        return ">89"
    return str(int(round(age)))


def render_clinical_context(ctx):
    """Render the clinical context panel HTML."""
    if ctx is None:
        return '<p style="color:#999;"><em>Clinical context not available for this patient.</em></p>'

    # Demographics
    demo = ctx.get('demographics') or {}
    demo_html = f"""
    <table class="context-table" style="margin-bottom:8px;">
        <tr>
            <td><strong>Age:</strong> {fmt_age(demo.get('age'))}</td>
            <td><strong>Gender:</strong> {demo.get('gender', '—')}</td>
            <td><strong>Ethnicity:</strong> {demo.get('ethnicity', '—')}</td>
        </tr>
        <tr>
            <td colspan="2"><strong>Admission Dx:</strong> {demo.get('diagnosis', '—')}</td>
            <td><strong>ICU:</strong> {demo.get('icu_type', '—')} ({fmt_val(demo.get('los_icu_days'))} days)</td>
        </tr>
    </table>"""

    # Labs
    labs = ctx.get('labs') or {}
    lab_items = []
    lab_defs = [
        ('creatinine', 'mg/dL'), ('bicarbonate', 'mEq/L'), ('anion gap', 'mEq/L'),
        ('hemoglobin', 'g/dL'), ('glucose', 'mg/dL'), ('platelets', 'K/uL'),
    ]
    for name, unit in lab_defs:
        v = labs.get(name)
        lab_items.append(f"<td><strong>{name.title()}:</strong> {fmt_val(v, f' {unit}')}</td>")

    labs_html = f"""
    <table class="context-table" style="margin-bottom:8px;">
        <tr><td colspan="6" style="background:#f5f5f5;"><strong>Most Recent Labs</strong></td></tr>
        <tr>{''.join(lab_items[:3])}</tr>
        <tr>{''.join(lab_items[3:])}</tr>
    </table>"""

    # Vitals history
    vitals = ctx.get('vitals_history') or {}
    vitals_defs = [
        ('heart rate', 'bpm'), ('mean blood pressure', 'mmHg'),
        ('respiratory rate', '/min'), ('glascow coma scale total', ''),
        ('diastolic blood pressure', 'mmHg'),
    ]
    vitals_rows = ""
    for name, unit in vitals_defs:
        vals = vitals.get(name, [])
        cells = "".join(f"<td>{fmt_val(v)}</td>" for v in vals[-6:])
        # Pad if fewer than 6 values
        cells += "<td>—</td>" * max(0, 6 - len(vals))
        vitals_rows += f"<tr><td><strong>{name.title()}</strong> ({unit})</td>{cells}</tr>\n"

    n_hours = min(6, len(vitals.get('heart rate', [])))
    hour_headers = "".join(f"<th>-{6-i}h</th>" for i in range(6))
    vitals_html = f"""
    <table class="vitals-table" style="margin-bottom:8px;">
        <thead><tr><th>Vital Sign</th>{hour_headers}</tr></thead>
        <tbody>{vitals_rows}</tbody>
    </table>"""

    # Treatment history
    tx = ctx.get('treatment_history') or {}
    tx_html = ""
    if tx:
        vaso_vals = tx.get('vaso', [])
        vent_vals = tx.get('vent', [])
        vaso_cells = "".join(f"<td>{fmt_val(v)}</td>" for v in vaso_vals[-6:])
        vent_cells = "".join(
            f"<td>{'Yes' if v and v > 0.5 else 'No' if v is not None else '—'}</td>"
            for v in vent_vals[-6:]
        )
        vaso_cells += "<td>—</td>" * max(0, 6 - len(vaso_vals))
        vent_cells += "<td>—</td>" * max(0, 6 - len(vent_vals))
        tx_html = f"""
        <table class="vitals-table">
            <thead><tr><th>Treatment</th>{hour_headers}</tr></thead>
            <tbody>
                <tr><td><strong>Vasopressor</strong></td>{vaso_cells}</tr>
                <tr><td><strong>Ventilation</strong></td>{vent_cells}</tr>
            </tbody>
        </table>"""

    return demo_html + labs_html + vitals_html + tx_html


def generate_case_card(case, tau, patient_contexts=None):
    case_id = case['case_id']
    obs_dbp = case['observed_dbp']
    n_feas = case['n_feasible']
    plan_a = case['plan_a']
    plan_b = case['plan_b']

    # Get clinical context if available
    ind_id = str(case['individual_id'])
    ctx = patient_contexts.get(ind_id) if patient_contexts else None
    clinical_panel = render_clinical_context(ctx)

    return f"""
    <div class="case-card" id="{case_id}">
        <div class="case-header">
            <h2>Case {case_id}</h2>
            <span class="completion-badge" id="badge_{case_id}">Incomplete</span>
        </div>

        <div class="patient-context">
            <h3>Patient Context</h3>
            {clinical_panel}
            <table class="context-table" style="margin-top:8px; background:#eaf2f8; padding:6px;">
                <tr>
                    <td><strong>Current DBP:</strong> {obs_dbp:.0f} mmHg</td>
                    <td><strong>Target:</strong> 60–90 mmHg</td>
                    <td><strong>Safety:</strong> 40–120 mmHg</td>
                    <td><strong>Horizon:</strong> {tau}h</td>
                    <td><strong>Feasible:</strong> {n_feas}/100</td>
                </tr>
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
    .vitals-table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
    .vitals-table th, .vitals-table td { border: 1px solid #ddd; padding: 4px 6px; text-align: center; }
    .vitals-table th { background: #ecf0f1; font-size: 0.9em; }
    .vitals-table td:first-child { text-align: left; white-space: nowrap; }
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

function getReviewerName() {{
    const el = document.getElementById('reviewer_name');
    return el ? el.value.trim() : '';
}}

function sanitizeName(name) {{
    return name.replace(/\\s+/g, '_').replace(/[^a-zA-Z0-9_\\-]/g, '');
}}

function downloadCSV() {{
    const name = getReviewerName();
    if (!name) {{
        alert('Please enter your name at the top of the page before downloading.');
        document.getElementById('reviewer_name').focus();
        return;
    }}

    let rows = ['case_id,reviewer,plan_a_likert,plan_b_likert,preference,comments'];
    CASE_IDS.forEach(function(cid) {{
        const a = getRadioValue(cid + '_a');
        const b = getRadioValue(cid + '_b');
        const p = getRadioValue(cid + '_pref');
        const commentEl = document.getElementById(cid + '_comment');
        const comment = commentEl ? commentEl.value.replace(/"/g, '""').replace(/\\n/g, ' ') : '';
        rows.push(cid + ',"' + name + '",' + a + ',' + b + ',' + p + ',"' + comment + '"');
    }});

    // Append overall feedback as special rows
    const feedbackFields = ['feedback_realism', 'feedback_patterns', 'feedback_info', 'feedback_other'];
    feedbackFields.forEach(function(fid) {{
        const el = document.getElementById(fid);
        const val = el ? el.value.replace(/"/g, '""').replace(/\\n/g, ' ') : '';
        if (val) {{
            rows.push(fid + ',"' + name + '",,,,\"' + val + '\"');
        }}
    }});

    const csv = rows.join('\\n');
    const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 's51a_ratings_' + sanitizeName(name) + '.csv';
    a.click();
    URL.revokeObjectURL(url);
}}

// Auto-save to localStorage
function saveToLocalStorage() {{
    const state = {{}};
    state['_reviewer_name'] = getReviewerName();
    CASE_IDS.forEach(function(cid) {{
        state[cid + '_a'] = getRadioValue(cid + '_a');
        state[cid + '_b'] = getRadioValue(cid + '_b');
        state[cid + '_pref'] = getRadioValue(cid + '_pref');
        const commentEl = document.getElementById(cid + '_comment');
        state[cid + '_comment'] = commentEl ? commentEl.value : '';
    }});
    ['feedback_realism', 'feedback_patterns', 'feedback_info', 'feedback_other'].forEach(function(fid) {{
        const el = document.getElementById(fid);
        if (el) state[fid] = el.value;
    }});
    localStorage.setItem('s51a_ratings', JSON.stringify(state));
    document.getElementById('save-status').textContent = 'Saved at ' + new Date().toLocaleTimeString();
}}

function loadFromLocalStorage() {{
    const saved = localStorage.getItem('s51a_ratings');
    if (!saved) return;
    const state = JSON.parse(saved);
    if (state['_reviewer_name']) {{
        document.getElementById('reviewer_name').value = state['_reviewer_name'];
    }}
    ['feedback_realism', 'feedback_patterns', 'feedback_info', 'feedback_other'].forEach(function(fid) {{
        if (state[fid]) {{
            const el = document.getElementById(fid);
            if (el) el.value = state[fid];
        }}
    }});
    Object.keys(state).forEach(function(key) {{
        if (key.startsWith('_') || key.startsWith('feedback_')) return;
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

function clearAll() {{
    if (!confirm('This will erase ALL your ratings, comments, feedback, and saved progress. Are you sure?')) return;
    localStorage.removeItem('s51a_ratings');
    document.getElementById('reviewer_name').value = '';
    CASE_IDS.forEach(function(cid) {{
        document.querySelectorAll('input[name="' + cid + '_a"]').forEach(function(r) {{ r.checked = false; }});
        document.querySelectorAll('input[name="' + cid + '_b"]').forEach(function(r) {{ r.checked = false; }});
        document.querySelectorAll('input[name="' + cid + '_pref"]').forEach(function(r) {{ r.checked = false; }});
        var c = document.getElementById(cid + '_comment');
        if (c) c.value = '';
    }});
    ['feedback_realism', 'feedback_patterns', 'feedback_info', 'feedback_other'].forEach(function(fid) {{
        var el = document.getElementById(fid);
        if (el) el.value = '';
    }});
    updateProgress();
    document.getElementById('save-status').textContent = 'Cleared at ' + new Date().toLocaleTimeString();
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

    # Load patient clinical context if available
    patient_contexts = None
    if CONTEXT_PATH.exists():
        with open(CONTEXT_PATH) as f:
            patient_contexts = json.load(f)
        print(f"Loaded clinical context for {len(patient_contexts)} patients")
    else:
        print(f"No clinical context file ({CONTEXT_PATH}), generating cards without it")

    print(f"Generating cards for {len(cases)} cases at τ={tau}")

    cards_html = ""
    for case in cases:
        cards_html += generate_case_card(case, tau, patient_contexts)

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
        <button class="download-btn" style="background:#e53935;" onclick="clearAll()">Clear All</button>
        <span id="save-status" style="font-size:0.8em;color:#666;"></span>
    </div>

    <div class="name-box" style="background:#e8eaf6; border:2px solid #5c6bc0; border-radius:8px; padding:15px; margin:20px 0;">
        <label for="reviewer_name" style="font-size:1.1em; font-weight:bold;">Reviewer name (required):</label>
        <input type="text" id="reviewer_name" placeholder="e.g. Maria Silva"
            style="margin-left:10px; padding:6px 12px; font-size:1em; border:1px solid #999; border-radius:4px; width:300px;"
            oninput="updateProgress()">
        <p style="margin:5px 0 0 0; font-size:0.85em; color:#555;">Your name will be included in the downloaded CSV filename.</p>
    </div>

    {generate_instructions(len(cases))}

    {cards_html}

    <div class="case-card" style="background:#f3e5f5; border-color:#9c27b0;">
        <h2 style="color:#6a1b9a; border-bottom-color:#ce93d8;">Overall Feedback</h2>
        <p>Please share your overall impressions after reviewing all cases. Your qualitative feedback is very valuable for interpreting the quantitative results.</p>

        <p><strong>1. How realistic were the patient scenarios and treatment options presented?</strong></p>
        <textarea id="feedback_realism" rows="3" style="width:100%; box-sizing:border-box;"
            placeholder="e.g., Were the treatment options clinically plausible? Was the information sufficient to make a judgment?"></textarea>

        <p><strong>2. In general, did one type of plan tend to feel more clinically appropriate than the other? Any patterns you noticed?</strong></p>
        <textarea id="feedback_patterns" rows="3" style="width:100%; box-sizing:border-box;"
            placeholder="e.g., Plans with/without vasopressors seemed more appropriate for..."></textarea>

        <p><strong>3. What additional patient information would have helped you make better assessments?</strong></p>
        <textarea id="feedback_info" rows="3" style="width:100%; box-sizing:border-box;"
            placeholder="e.g., Lab values, fluid balance, MAP, comorbidities, reason for ICU admission..."></textarea>

        <p><strong>4. Any other comments on the evaluation process, the AI-recommended plans, or suggestions for improvement?</strong></p>
        <textarea id="feedback_other" rows="3" style="width:100%; box-sizing:border-box;"
            placeholder="Free-form comments..."></textarea>
    </div>

    <div style="text-align:center; margin:30px; padding:20px; background:#e8f5e9; border-radius:8px;">
        <p style="font-size:1.2em;"><strong>All done?</strong></p>
        <button class="download-btn ready" onclick="downloadCSV()" style="font-size:1.1em; padding:12px 32px;">
            Download Ratings CSV
        </button>
        <p style="margin-top:10px; color:#555;">Please email the downloaded CSV file back to the researcher. Thank you!</p>
    </div>

    {generate_javascript(case_ids)}
</body>
</html>"""

    with open(OUTPUT_PATH, 'w') as f:
        f.write(html)

    print(f"Generated {OUTPUT_PATH} ({len(html)//1024} KB)")


if __name__ == '__main__':
    main()
