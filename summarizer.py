"""
Lightweight extractive meeting summarizer.

Implements a TextRank-like algorithm: split into sentences, compute sentence
similarity using TF vectors and cosine similarity, run PageRank, and select
top-ranked sentences. Designed to work offline with only Python standard
libraries (no heavy ML libraries required).

Functions:
- summarize(text, num_sentences=3): returns an extractive summary (string)

"""
from typing import List, Tuple
import math
import re
from collections import Counter, defaultdict

_WORD_RE = re.compile(r"\w+")


def _sent_tokenize(text: str) -> List[str]:
    # Simple sentence splitter based on punctuation. Keeps abbreviations naive.
    text = text.strip()
    if not text:
        return []
    # Normalize newlines
    text = text.replace('\n', ' ')
    # Split on sentence boundaries (., ?, !) followed by space and capital letter or EOL
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(])', text)
    # Fallback: if only one sentence returned, try splitting on period-space
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def _tokenize(sent: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(sent)]


def _sentence_vectors(sents: List[str]) -> List[Counter]:
    vecs = []
    for s in sents:
        tokens = _tokenize(s)
        vecs.append(Counter(tokens))
    return vecs


def _cosine(a: Counter, b: Counter) -> float:
    # cosine similarity between two counters
    if not a or not b:
        return 0.0
    common = set(a.keys()) & set(b.keys())
    num = sum(a[w] * b[w] for w in common)
    denom = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    if denom == 0:
        return 0.0
    return num / denom


def _build_similarity_matrix(vecs: List[Counter]) -> List[List[float]]:
    n = len(vecs)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine(vecs[i], vecs[j])
            mat[i][j] = sim
            mat[j][i] = sim
    return mat


def _pagerank(sim_matrix: List[List[float]], d=0.85, max_iter=100, tol=1e-6) -> List[float]:
    n = len(sim_matrix)
    if n == 0:
        return []
    # Build adjacency with row-normalization
    scores = [1.0 / n] * n
    # Precompute row sums
    row_sums = [sum(sim_matrix[i]) for i in range(n)]
    for iteration in range(max_iter):
        new_scores = [ (1 - d) / n ] * n
        for i in range(n):
            for j in range(n):
                if row_sums[j] == 0:
                    continue
                new_scores[i] += d * (sim_matrix[i][j] * scores[j] / row_sums[j])
        diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores
        if diff < tol:
            break
    return scores


def summarize(text: str, num_sentences: int = 3) -> str:
    """Return an extractive summary selecting the top `num_sentences` sentences.

    The summarizer preserves the original sentence order in the returned
    summary. If input has fewer sentences than requested, returns the whole
    text.
    """
    sents = _sent_tokenize(text)
    if not sents:
        return ""
    if num_sentences >= len(sents):
        return ' '.join(sents)
    vecs = _sentence_vectors(sents)
    sim = _build_similarity_matrix(vecs)
    scores = _pagerank(sim)
    # select top indices by score
    idx_scores = list(enumerate(scores))
    idx_scores.sort(key=lambda x: x[1], reverse=True)
    top_idx = sorted(i for i, _ in idx_scores[:num_sentences])
    selected = [sents[i] for i in top_idx]
    return ' '.join(selected)


def abstractive_summarize(text: str, max_length: int = 130, min_length: int = 30) -> str:
    """Try to produce a short abstractive summary in own words.

    Strategy:
    1. If OpenAI is available and an API key is set, use it (ChatCompletion) to produce a concise paraphrase.
    2. Else, if `transformers` is installed, use a summarization pipeline (BART/PEGASUS-like).
    3. Otherwise, fall back to an extractive summary and perform a small heuristic rewrite to shorten and paraphrase.
    """
    text = (text or '').strip()
    if not text:
        return ''

    # Attempt OpenAI ChatCompletion (if openai package and OPENAI_API_KEY present)
    try:
        import os
        if 'OPENAI_API_KEY' in os.environ:
            try:
                import openai
                openai.api_key = os.environ.get('OPENAI_API_KEY')
                prompt = (
                    "Summarize the following meeting transcript in 2-3 short sentences in clear, natural English, "
                    "using your own words (do not quote sentences verbatim). Be concise and highlight key decisions and next steps:\n\n" + text
                )
                # Use ChatCompletion if available
                try:
                    resp = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo',
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens= max(60, max_length // 2),
                        temperature=0.3,
                    )
                    ans = resp['choices'][0]['message']['content'].strip()
                    if ans:
                        return ans
                except Exception:
                    # fallback to older completions API
                    resp = openai.Completion.create(
                        engine='text-davinci-003',
                        prompt=prompt,
                        max_tokens=max_length,
                        temperature=0.3,
                    )
                    ans = resp['choices'][0]['text'].strip()
                    if ans:
                        return ans
            except Exception:
                # openai not usable, continue to transformers
                pass
    except Exception:
        pass

    # Attempt transformers summarization pipeline
    try:
        from transformers import pipeline
        # choose a smaller model by default to reduce memory usage
        model_name = 'sshleifer/distilbart-cnn-12-6'
        summarizer = pipeline('summarization', model=model_name)
        # Some models expect shorter inputs; chunk if necessary
        # We'll cap input length to something reasonable; longer inputs will be truncated by the pipeline.
        out = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        if out and isinstance(out, list):
            return out[0].get('summary_text', '').strip()
    except Exception:
        # transformers not available or model failed
        pass

    # Fallback: extractive + heuristic rewrite (simple paraphrase)
    ext = summarize(text, num_sentences=3)
    # simple heuristics: shorten, remove filler phrases, replace common connectors
    replacements = [
        (r"\baction items?\b", 'actions'),
        (r"\bwe will\b", "team will"),
        (r"\bwe'll\b", "team will"),
        (r"\bwill be\b", "is"),
        (r"\bdiscussed the\b", "discussed"),
        (r"\bin order to\b", "to"),
    ]
    s = ext
    for pat, rep in replacements:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    # collapse repeated spaces
    s = re.sub(r"\s+", ' ', s).strip()
    # Ensure it's short — if still long, truncate to first 2 sentences
    s_sents = _sent_tokenize(s)
    if len(s_sents) > 2:
        s = ' '.join(s_sents[:2])
    return s


def transcribe_audio(filepath: str) -> str:
    """Transcribe audio using OpenAI Whisper (if available) or return empty string.

    This function will try OpenAI if `OPENAI_API_KEY` is set and the `openai`
    package is installed. Otherwise it's a placeholder that returns an empty
    transcript (you can plug any ASR here).
    """
    import os
    import requests
    import time

    # Try AssemblyAI if key is present
    assembly_key = os.environ.get('ASSEMBLYAI_API_KEY') or os.environ.get('ASSEMBLYAI_KEY')
    if assembly_key:
        try:
            headers = {'authorization': assembly_key}
            # Upload file
            with open(filepath, 'rb') as f:
                data = f.read()
            up = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=data, timeout=60)
            if up.status_code == 200:
                upload_url = up.json().get('upload_url')
                # create transcript
                body = {"audio_url": upload_url}
                tr = requests.post('https://api.assemblyai.com/v2/transcript', headers={**headers, 'content-type': 'application/json'}, json=body)
                if tr.status_code in (200,201):
                    tid = tr.json().get('id')
                    # poll for completion
                    for _ in range(120):
                        time.sleep(2)
                        stat = requests.get(f'https://api.assemblyai.com/v2/transcript/{tid}', headers=headers)
                        if stat.status_code != 200:
                            continue
                        j = stat.json()
                        status = j.get('status')
                        if status == 'completed':
                            return j.get('text', '') or ''
                        if status == 'error':
                            break
            # fallback to other methods if upload failed
        except Exception:
            pass

    # Try OpenAI if available
    if 'OPENAI_API_KEY' in os.environ:
        try:
            import openai
            openai.api_key = os.environ.get('OPENAI_API_KEY')
            # openai.Audio.transcriptions available in some SDK versions; try recommended method
            with open(filepath, 'rb') as f:
                resp = openai.Audio.transcribe('gpt-4o-transcribe', f)
                # resp may vary by SDK; try to extract text
                if isinstance(resp, dict):
                    return resp.get('text') or resp.get('transcript') or ''
                return str(resp)
        except Exception:
            # if OpenAI isn't usable, fall through
            pass
    # No ASR available — return empty to force user-provided transcript
    return ''


def summarize_and_actions_with_llm(transcript: str) -> Tuple[str, str]:
    """Generate an abstractive summary and action items from a transcript.

    Tries OpenAI (if key present), otherwise uses a simple heuristic fallback.
    Returns (summary, actions_text).
    """
    import os
    prompt = (
        "Summarize this meeting transcript into a concise summary (2-4 sentences) "
        "highlighting key decisions, and then list clear action items with owners and deadlines if present.\n\n"
        + transcript
    )
    # Try OpenAI ChatCompletion
    if 'OPENAI_API_KEY' in os.environ:
        try:
            import openai
            openai.api_key = os.environ.get('OPENAI_API_KEY')
            resp = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2,
            )
            text = resp['choices'][0]['message']['content'].strip()
            # crude split: assume actions after a heading 'Action' or 'Actions' or 'Action items'
            actions = ''
            summary = text
            for sep in ['\nAction items', '\nActions', '\nAction item', '\nAction items:']:
                if sep in text:
                    parts = text.split(sep, 1)
                    summary = parts[0].strip()
                    actions = parts[1].strip()
                    break
            return summary, actions
        except Exception:
            pass

    # Fallback heuristic: use extractive summarizer + find action sentences
    summary = summarize(transcript, num_sentences=3)
    # action sentences often start with verbs or contain 'will'/'action'/'due'
    actions = []
    sents = _sent_tokenize(transcript)
    for s in sents:
        if re.search(r"\b(will|shall|action|due|deadline|assign|owner|task)\b", s, re.IGNORECASE):
            actions.append(s)
    actions_text = '\n'.join(actions)
    return summary, actions_text


if __name__ == '__main__':
    sample = (
        """Today we discussed the Q3 roadmap. The engineering team will focus on
        performance improvements. Marketing will prepare the new campaign. Next
        week we'll reconvene to review progress."""
    )
    print(summarize(sample, 2))
