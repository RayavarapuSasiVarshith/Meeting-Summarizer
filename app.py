"""Minimal Flask web UI for the extractive meeting summarizer."""
from flask import Flask, request, render_template_string, jsonify, redirect, url_for
from summarizer import summarize, abstractive_summarize, transcribe_audio, summarize_and_actions_with_llm
import os
from pathlib import Path
from db import init_db, create_meeting, update_transcript, update_summary_actions, get_meeting, list_meetings

UPLOAD_DIR = Path(__file__).parent / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)
init_db()

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<title>Meeting Summarizer</title>
<h1>Meeting Summarizer</h1>
<form method=post action="/summarize">
    <label for=text>Paste meeting transcript or notes (or upload audio below):</label><br>
    <textarea name=text rows=15 cols=100></textarea><br>
    <label for=num>Number of sentences (extractive):</label>
    <input type=number name=num value=3 min=1 max=20>
    <label for=mode>Mode:</label>
    <select name=mode>
        <option value="extractive">Extractive (fast, picks sentences)</option>
        <option value="abstractive">Abstractive (paraphrase in own words)</option>
    </select>
    <input type=submit value=Summarize>
</form>
<h2>Upload audio</h2>
<form method=post action="/upload" enctype="multipart/form-data">
    <input type=file name=audio accept="audio/*">
    <input type=submit value=Upload>
</form>
<h2>Saved meetings</h2>
<ul>
{% for m in meetings %}
    <li><a href="/meet/{{m['id']}}">{{m['filename']}}</a> - {{m['uploaded_at']}}</li>
{% endfor %}
</ul>
{% if summary %}
<h2>Summary</h2>
<div style="white-space:pre-wrap;border:1px solid #ddd;padding:10px">{{summary}}</div>
{% endif %}
"""


@app.route('/', methods=['GET'])
def index():
    meetings = list_meetings()
    return render_template_string(INDEX_HTML, meetings=meetings)


@app.route('/summarize', methods=['POST'])
def summarize_route():
    text = request.form.get('text', '')
    try:
        num = int(request.form.get('num', 3))
    except ValueError:
        num = 3
    mode = request.form.get('mode', 'extractive')
    if mode == 'abstractive':
        try:
            summary = abstractive_summarize(text)
        except Exception:
            # fallback to extractive if abstractive fails
            summary = summarize(text, num_sentences=num)
    else:
        summary = summarize(text, num_sentences=num)
    return render_template_string(INDEX_HTML, summary=summary)


@app.route('/upload', methods=['POST'])
def upload_audio():
    f = request.files.get('audio')
    if not f:
        return redirect(url_for('index'))
    fname = f.filename
    dest = UPLOAD_DIR / fname
    f.save(dest)
    mid = create_meeting(fname)
    # attempt transcription
    transcript = transcribe_audio(str(dest))
    if transcript:
        update_transcript(mid, transcript)
        summary, actions = summarize_and_actions_with_llm(transcript)
        update_summary_actions(mid, summary, actions)
    return redirect(url_for('meet', mid=mid))


@app.route('/meet/<int:mid>')
def meet(mid:int):
    m = get_meeting(mid)
    if not m:
        return 'Not found', 404
    return render_template_string('''
<h1>Meeting {{m.filename}}</h1>
<h2>Transcript</h2>
<pre>{{m.transcript}}</pre>
<h2>Summary</h2>
<pre>{{m.summary}}</pre>
<h2>Actions</h2>
<pre>{{m.actions}}</pre>
<a href="/">Back</a>
''', m=m)


@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    data = request.get_json() or {}
    text = data.get('text', '')
    num = int(data.get('num_sentences', 3))
    mode = data.get('mode', 'extractive')
    if mode == 'abstractive':
        try:
            from summarizer import abstractive_summarize
            summary = abstractive_summarize(text)
        except Exception:
            summary = summarize(text, num_sentences=num)
    else:
        summary = summarize(text, num_sentences=num)
    return jsonify({'summary': summary})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
