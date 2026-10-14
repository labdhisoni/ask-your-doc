# ZeroGPU support: must be imported before torch. Falls back to a no-op on Colab / CPU.
try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f
import os, re, shutil
import numpy as np
import torch
import gradio as gr
from pypdf import PdfReader
import pypdfium2 as pdfium
import pytesseract
from PIL import Image
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OCR_AVAILABLE = shutil.which("tesseract") is not None
MAX_OCR_PAGES = int(os.environ.get("MAX_OCR_PAGES", "60"))
print("Device:", DEVICE, "| OCR available:", OCR_AVAILABLE)

# ---- Embedding model (retrieval) ----
EMBED_NAME = "BAAI/bge-small-en-v1.5"
embedder = SentenceTransformer(EMBED_NAME, device=DEVICE)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---- Answer model (generation) ----
# Override with the LLM_NAME environment variable (e.g. a Space "Variable").
DEFAULT_LLM = "Qwen/Qwen2.5-1.5B-Instruct" if DEVICE == "cuda" else "Qwen/Qwen2.5-0.5B-Instruct"
LLM_NAME = os.environ.get("LLM_NAME") or DEFAULT_LLM
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
llm = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)
llm.eval()
print("Models loaded:", EMBED_NAME, "+", LLM_NAME)

IMAGE_EXTS = (".png", ".jpg", ".jpeg")
MIN_TEXT_CHARS = 40          # pages with less text than this are treated as scanned
PARAGRAPH_TARGET_WORDS = 400  # size of a "section" for DOCX/TXT files


def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def ocr_image(img):
    return pytesseract.image_to_string(img)


def group_paragraphs(paragraphs, target=PARAGRAPH_TARGET_WORDS):
    """Group paragraphs into ~target-word sections. Returns [(loc, text)]."""
    sections, cur, cur_words = [], [], 0

    def flush():
        nonlocal cur, cur_words
        if cur:
            sections.append(" ".join(cur))
            cur, cur_words = [], 0

    for para in paragraphs:
        para = clean(para)
        if not para:
            continue
        words = para.split()
        if len(words) > target * 2:           # giant block with no paragraph breaks
            flush()
            for i in range(0, len(words), target):
                sections.append(" ".join(words[i:i + target]))
            continue
        cur.append(para)
        cur_words += len(words)
        if cur_words >= target:
            flush()
    flush()
    return [(f"section {i}", t) for i, t in enumerate(sections, start=1)]


def extract_pdf(path):
    units, notes = [], []
    reader = PdfReader(path)
    pdf_doc = None
    ocr_done = ocr_skipped = 0

    for i, page in enumerate(reader.pages):
        text = clean(page.extract_text())
        if len(text) < MIN_TEXT_CHARS:                      # probably a scanned page
            if OCR_AVAILABLE and ocr_done < MAX_OCR_PAGES:
                try:
                    if pdf_doc is None:
                        pdf_doc = pdfium.PdfDocument(path)
                    img = pdf_doc[i].render(scale=2.0).to_pil()
                    ocr_text = clean(ocr_image(img))
                    ocr_done += 1
                    if len(ocr_text) > len(text):
                        text = ocr_text
                except Exception as e:
                    notes.append(f"OCR failed on page {i + 1} ({e})")
            elif not text:
                ocr_skipped += 1
        if text:
            units.append((f"page {i + 1}", text))

    if pdf_doc is not None:
        pdf_doc.close()
    if ocr_done:
        notes.append(f"used OCR on {ocr_done} scanned page(s)")
    if ocr_skipped:
        why = "OCR is not installed" if not OCR_AVAILABLE else f"OCR limit of {MAX_OCR_PAGES} pages reached"
        notes.append(f"skipped {ocr_skipped} image-only page(s) ({why})")
    return units, notes


def extract_docx(path):
    doc = Document(path)
    blocks = []
    for child in doc.element.body.iterchildren():      # keeps paragraphs and tables in order
        if child.tag.endswith("}p"):
            blocks.append(Paragraph(child, doc).text)
        elif child.tag.endswith("}tbl"):
            for row in Table(child, doc).rows:
                cells = [clean(c.text) for c in row.cells]
                blocks.append(" | ".join(c for c in cells if c))
    return group_paragraphs(blocks), []


def extract_txt(path):
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return group_paragraphs(re.split(r"\n\s*\n", text)), []


def extract_image(path):
    if not OCR_AVAILABLE:
        raise RuntimeError("OCR (tesseract) is not installed")
    text = clean(ocr_image(Image.open(path)))
    return ([("image", text)] if text else []), ["read text from image using OCR"]


def extract_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(path)
    if ext == ".docx":
        return extract_docx(path)
    if ext in (".txt", ".md"):
        return extract_txt(path)
    if ext in IMAGE_EXTS:
        return extract_image(path)
    raise ValueError(f"unsupported file type '{ext}'")

def split_text(text, chunk_words=150, overlap=30):
    """Sentence-aware chunking; very long 'sentences' are split by word count."""
    pieces = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        words = s.split()
        if len(words) > chunk_words:
            pieces += [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]
        elif words:
            pieces.append(s)

    out, cur, cur_len = [], [], 0
    for s in pieces:
        n = len(s.split())
        if cur and cur_len + n > chunk_words:
            out.append(" ".join(cur))
            tail, tail_len = [], 0              # keep the last sentences as overlap
            for prev in reversed(cur):
                tail_len += len(prev.split())
                tail.insert(0, prev)
                if tail_len >= overlap:
                    break
            if tail_len + n > chunk_words:       # overlap would make the chunk too big
                tail, tail_len = [], 0
            cur, cur_len = tail, tail_len
        cur.append(s)
        cur_len += n
    if cur:
        out.append(" ".join(cur))
    return [c for c in out if c.strip()]


@gpu(duration=120)
def process_files(files):
    """Returns: status text, new index, cleared chat, cleared sources."""
    if not files:
        return "Please upload at least one file.", None, [], ""

    new_chunks, notes = [], []
    for f in files:
        path = f if isinstance(f, str) else getattr(f, "name", str(f))
        name = os.path.basename(path)
        try:
            units, file_notes = extract_file(path)
            notes += [f"{name}: {n}" for n in file_notes]
            before = len(new_chunks)
            for loc, text in units:
                for c in split_text(text):
                    new_chunks.append({"text": c, "source": name, "loc": loc})
            if len(new_chunks) == before:
                notes.append(f"{name}: no readable text found")
        except Exception as e:
            notes.append(f"{name}: failed ({e})")

    if not new_chunks:
        return "No readable text found.\n" + "\n".join(notes), None, [], ""

    texts = [c["text"] for c in new_chunks]
    emb = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                          batch_size=32, show_progress_bar=False)
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        mat = vec.fit_transform(texts)
    except ValueError:                           # e.g. only numbers / stop-words
        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        mat = vec.fit_transform(texts)

    index = {"chunks": new_chunks, "dense": emb, "tfidf": vec, "mat": mat}
    msg = f"Done. Indexed {len(files)} file(s) into {len(new_chunks)} chunks."
    if notes:
        msg += "\n" + "\n".join(notes)
    return msg, index, [], ""


BROAD_PATTERN = re.compile(
    r"\b(summar(y|ize|ise)|overview|what is (this|the) (document|pdf|file) about|"
    r"main (points|topics|ideas)|key (points|takeaways)|tl;?dr)\b", re.I)


def retrieve(question, index, top_k=6):
    chunks = index["chunks"]

    # Broad questions: sample chunks spread across the whole document
    if BROAD_PATTERN.search(question):
        idx = np.linspace(0, len(chunks) - 1, num=min(top_k + 2, len(chunks))).astype(int)
        return [dict(chunks[i], score=1.0) for i in sorted(set(idx))]

    q_emb = embedder.encode([QUERY_PREFIX + question], convert_to_numpy=True,
                            normalize_embeddings=True)[0]
    dense = index["dense"] @ q_emb
    sparse = (index["mat"] @ index["tfidf"].transform([question]).T).toarray().ravel()

    def norm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)

    score = 0.65 * norm(dense) + 0.35 * norm(sparse)   # meaning + exact keywords
    order = np.argsort(score)[::-1][:top_k]
    return [dict(chunks[i], score=float(score[i])) for i in order]

def msg_text(content):
    """Chat content can be a string or (in newer Gradio) a list of content blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or ""
    if isinstance(content, (list, tuple)):
        return " ".join(msg_text(c) for c in content)
    return str(content)


def norm_history(history):
    out = []
    for m in history or []:
        if isinstance(m, dict):
            out.append({"role": m.get("role", "user"), "content": msg_text(m.get("content"))})
        elif isinstance(m, (list, tuple)) and len(m) == 2:      # old (user, bot) pairs
            if m[0]:
                out.append({"role": "user", "content": msg_text(m[0])})
            if m[1]:
                out.append({"role": "assistant", "content": msg_text(m[1])})
    return out


@torch.no_grad()
def generate(messages, max_new_tokens=350):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(DEVICE)
    out = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def condense_question(history, question):
    """Turn a follow-up ('what about the second one?') into a standalone question."""
    if not history:
        return question
    convo = "\n".join(f"{m['role'].capitalize()}: {m['content'][:500]}" for m in history[-6:])
    messages = [
        {"role": "system", "content": "Rewrite the user's last question so it is fully "
         "self-contained, using the conversation for context. If it already is, repeat it "
         "unchanged. Output ONLY the rewritten question."},
        {"role": "user", "content": f"Conversation:\n{convo}\n\nLast question: {question}\n\nStandalone question:"},
    ]
    rewritten = generate(messages, max_new_tokens=64).strip().split("\n")[0].strip()
    return rewritten if 5 <= len(rewritten) <= 400 else question


def build_messages(question, passages, history):
    context = "\n\n".join(
        f"[Source {i}: {p['source']}, {p['loc']}]\n{p['text']}"
        for i, p in enumerate(passages, start=1)
    )
    system = (
        "You answer questions about the user's uploaded documents. "
        "Use ONLY the provided context. Be accurate and concise. "
        "If the context does not contain the answer, say you could not find it in the documents. "
        "Do not make up facts. Mention the page or section when helpful."
    )
    messages = [{"role": "system", "content": system}]
    recent = history[-6:]
    while recent and recent[0]["role"] != "user":
        recent = recent[1:]
    for m in recent:
        messages.append({"role": m["role"], "content": m["content"][:800]})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    return messages


@gpu(duration=90)
def chat(question, history, index):
    """Returns: updated chat, cleared textbox, source passages."""
    history = norm_history(history)
    if not question or not question.strip():
        return history, "", ""

    previous = list(history)
    history.append({"role": "user", "content": question})

    if index is None:
        history.append({"role": "assistant", "content": "Please upload and process a document first."})
        return history, "", ""

    standalone = condense_question(previous, question)
    passages = retrieve(standalone, index)
    answer = generate(build_messages(standalone, passages, previous))
    history.append({"role": "assistant", "content": answer})

    sources = "\n\n".join(
        f"**{i}. {p['source']} - {p['loc']}**\n\n{p['text'][:350]}..."
        for i, p in enumerate(passages, start=1)
    )
    if standalone != question:
        sources = f"*Searched for:* {standalone}\n\n" + sources
    return history, "", sources

import html

EMPTY_STATS = """<div class="stats-empty">No documents loaded yet.<br>Upload files and click <b>Process documents</b>.</div>"""


def render_stats(index):
    if not index:
        return EMPTY_STATS
    chunks = index["chunks"]
    names = sorted({c["source"] for c in chunks})
    words = sum(len(c["text"].split()) for c in chunks)
    chips = "".join(f'<span class="file-chip">{html.escape(n)}</span>' for n in names)
    return f"""
<div class="stats-grid">
  <div class="stat"><div class="stat-num">{len(names)}</div><div class="stat-label">Files</div></div>
  <div class="stat"><div class="stat-num">{len(chunks)}</div><div class="stat-label">Passages</div></div>
  <div class="stat"><div class="stat-num">{words:,}</div><div class="stat-label">Words (approx.)</div></div>
</div>
<div class="chips">{chips}</div>
"""

import inspect

THEME = gr.themes.Soft(
    primary_hue="indigo", secondary_hue="purple", neutral_hue="slate", radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

CSS = """
.gradio-container { max-width: 1200px !important; margin: auto !important; }
.hero {
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
  color: #fff; border-radius: 20px; padding: 28px 32px; margin-bottom: 8px;
  box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
}
.hero h1 { margin: 6px 0 4px; font-size: 2.1rem; font-weight: 800; color: #fff; }
.hero p { margin: 0; opacity: 0.92; font-size: 1.02rem; color: #fff; }
.hero-badge {
  display: inline-block; background: rgba(255,255,255,0.18); border-radius: 999px;
  padding: 4px 12px; font-size: 0.8rem; font-weight: 600; letter-spacing: .02em; color: #fff;
}
.pills { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
.pill {
  background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28);
  border-radius: 999px; padding: 3px 12px; font-size: 0.82rem; color: #fff;
}
.stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.stat {
  background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.25);
  border-radius: 14px; padding: 12px 8px; text-align: center;
}
.stat-num { font-size: 1.5rem; font-weight: 800; color: #6366f1; line-height: 1.1; }
.stat-label { font-size: 0.75rem; opacity: 0.75; margin-top: 2px; }
.chips { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }
.file-chip {
  background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.3);
  border-radius: 999px; padding: 2px 10px; font-size: 0.78rem;
}
.stats-empty {
  border: 1.5px dashed rgba(99,102,241,0.4); border-radius: 14px; padding: 14px;
  text-align: center; font-size: 0.9rem; opacity: 0.85;
}
.footer-note { text-align: center; font-size: 0.8rem; opacity: 0.6; margin-top: 6px; }
"""

HERO = """
<div class="hero">
  <span class="hero-badge">RAG &middot; OCR &middot; Chat memory</span>
  <h1>Ask your documents</h1>
  <p>Drop in files, then chat with them. Every answer comes with the passages it was based on.</p>
  <div class="pills">
    <span class="pill">PDF</span><span class="pill">Word (DOCX)</span>
    <span class="pill">TXT / MD</span><span class="pill">Scans &amp; images (OCR)</span>
  </div>
</div>
"""

# Newer Gradio versions take theme/css in launch(); older ones in gr.Blocks().
_style = {"theme": THEME, "css": CSS}
if "theme" in inspect.signature(gr.Blocks.__init__).parameters:
    BLOCKS_KW, LAUNCH_KW = {"title": "Ask your documents", **_style}, {}
else:
    BLOCKS_KW, LAUNCH_KW = {"title": "Ask your documents"}, _style


def make_chatbot():
    attempts = [
        dict(type="messages", height=500, label="Conversation", show_copy_button=True,
             placeholder="### Upload a document, then ask me anything about it"),
        dict(type="messages", height=500, label="Conversation"),
        dict(height=500, label="Conversation"),   # newer Gradio: messages is the only format
    ]
    for kwargs in attempts:
        try:
            return gr.Chatbot(**kwargs)
        except TypeError:
            continue


with gr.Blocks(**BLOCKS_KW) as demo:
    index_state = gr.State(None)           # one index per browser session

    gr.HTML(HERO)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 1. Add documents")
            files = gr.File(label="Drop files here", file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg"],
                            type="filepath")
            process_btn = gr.Button("Process documents", variant="primary", size="lg")
            stats_html = gr.HTML(EMPTY_STATS)
            status = gr.Textbox(label="Processing notes", interactive=False, lines=3)

        with gr.Column(scale=2):
            gr.Markdown("### 2. Ask questions")
            chatbot = make_chatbot()
            with gr.Row():
                question = gr.Textbox(show_label=False, lines=1, scale=5, container=False,
                                      placeholder="Ask something about your documents...")
                ask_btn = gr.Button("Ask", variant="primary", scale=1)
            gr.Examples(
                examples=[["Summarize this document"],
                          ["What are the key points?"],
                          ["List all names, dates and numbers mentioned"],
                          ["What are the main conclusions or recommendations?"]],
                inputs=question, label="Try asking")
            with gr.Row():
                clear_btn = gr.Button("Clear chat", size="sm")
            with gr.Accordion("Source passages for the last answer", open=False):
                sources_out = gr.Markdown()

    gr.HTML('<div class="footer-note">Answers are generated from your documents by a small AI model. '
            'Check important details against the sources.</div>')

    process_btn.click(process_files, files, [status, index_state, chatbot, sources_out]) \
               .then(render_stats, index_state, stats_html)
    ask_btn.click(chat, [question, chatbot, index_state], [chatbot, question, sources_out])
    question.submit(chat, [question, chatbot, index_state], [chatbot, question, sources_out])
    clear_btn.click(lambda: ([], ""), None, [chatbot, sources_out])


# ---------- Login (username + password) ----------
# Credentials come from a secret named APP_USERS, formatted:  alice:pass1,bob:pass2
def load_auth():
    pairs = []
    for item in os.environ.get("APP_USERS", "").split(","):
        if ":" in item:
            user, pwd = item.split(":", 1)
            user, pwd = user.strip(), pwd.strip()
            if user and pwd:
                pairs.append((user, pwd))
    return pairs


AUTH_PAIRS = load_auth()
if AUTH_PAIRS:
    LAUNCH_KW["auth"] = AUTH_PAIRS
    print(f"Login enabled for {len(AUTH_PAIRS)} user(s).")
elif os.environ.get("SPACE_ID"):
    # On a Hugging Face Space, refuse to start without credentials rather than run open.
    raise RuntimeError(
        "No login configured. In Space Settings > Variables and secrets, add a secret "
        "named APP_USERS with value  username:password  (several users: a:pw1,b:pw2), "
        "then restart the Space."
    )
else:
    print("APP_USERS not set: running WITHOUT login (fine for Colab testing).")

demo.launch(**LAUNCH_KW)
