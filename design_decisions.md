# Design decisions — viva prep

This file records *why* each technical choice was made. If you can explain
these, you can answer most assessment questions.

---

## Overall architecture

**Q. Why four separate modules instead of one big model?**
Each module is a well-scoped, separately-trainable component with its own best
data and failure mode — the brief explicitly frames this as an integrated
pipeline of modules. This also makes deployment simple and lets us swap any
stage (e.g. replace TF-IDF with a transformer) without touching the rest.
All stages share a config (`support_bot/config.py`) so labels, paths and
mapping can never drift between training notebooks and the serving app.

**Q. Why the response policy (small talk / RAG / escalate)?**
Routing on intent + sentiment + language means we never run RAG when it is
wasteful or unsafe:
* *small talk* → no retrieval needed (fast, warm reply);
* *order / invoice / account / delivery* → RAG, grounded in the KB;
* *complaint* and *talk-to-a-human* → **escalation**, not an auto-generated
  policy answer (a complaint should reach a person; generating a reply risks
  more frustration);
* *frustrated tone anywhere* → an explicit apology/acknowledgment is prepended;
* *no confident retrieval* → an honest "I can't find that / shall I connect you
  to a human?" instead of a hallucinated answer.

---

## 01 — Language detection (traditional NLP)

**Q. Why TF-IDF and character n-grams rather than words or a transformer?**
Language identification is basically an *orthographic* problem: each language
has a distinctive script / diacritic / n-gram distribution. Character n-grams
(`char_wb`, 2–4) capture that even for words the model has never seen, and work
uniformly across Latin, Cyrillic, Arabic, Devanagari and CJK scripts. Words
would fail for Chinese/Japanese (no spaces). A transformer is unnecessary for a
90k-sample, ~99%-achievable task and is slower to serve.

**Q. Why SGD with `log_loss`?**
It is a linear logistic classifier trained by SGD — fast on high-dimensional
sparse input at 90k rows — and, unlike a plain SVM or `LinearSVC`, it exposes
`predict_proba`. Those probabilities are used downstream as a **confidence
gate** (and for top-3 diagnostics).

**Q. What enhancements did you add beyond a bare CountVectorizer baseline?**
1. `char_wb` character n-grams instead of a word bag (above);
2. `sublinear_tf` + `min_df`/`max_features` to keep the matrix well-conditioned;
3. shared `clean()` pre-processing identical at train and serve time;
4. probability outputs + per-language report + confusion-pair analysis rather
   than a single accuracy number;
5. a SMOKE/full switch + fixed seed for reproducible runs.

**Q. How do you handle a language not in the 20?**
The classifier still returns its best guess with a confidence; the serving
logic treats low confidence as "uncertain language" and falls back to English
framing rather than guessing wrong.

---

## 02 — Sentiment / emotion (RNN)

**Q. Why an RNN from scratch rather than a pretrained transformer?**
The brief allows either. A small BiLSTM trained from scratch is fully
transparent, has a tiny footprint, runs on CPU in minutes, and is a good
fit for short messages. A pretrained transformer would be more accurate but
needs a big download, is slower on CPU, and is harder to explain every layer
of in a viva.

**Q. Why train on 6 emotions and then collapse to 3 tone buckets?**
The 6-class model keeps the dataset's information; the routing stage only needs
`negative/neutral/positive`. Collapsing after prediction is a deterministic
map (`sadness/anger/fear → negative`, `joy/love → positive`,
`surprise → neutral`) that we also evaluate on. Training a 3-class model from
the start would throw away the emotion distinctions and still need the same
mapping logic at the end.

**Q. Why bidirectional + masked mean pooling?**
* Bidirectional: sentiment often depends on words after the target word
  ("I am **not** happy").
* Masked mean pooling over time instead of the last hidden state: robust to
  variable-length padded inputs and avoids relying on the (padding) final
  state. Padding is explicitly masked so `<pad>` positions contribute nothing.

**Q. Domain shift — the emotion data is Twitter, not customer support.**
Acknowledged in notebook 02. The notebook ends with a **qualitative check** on
hand-written support-style messages (as the brief suggests), and tone is used
only to adjust tone/escalation — the RAG grounding is what carries the factual
answer, so a slightly-off emotion cannot produce wrong policy info.

---

## 03 — Intent classifier (traditional ML on gold labels)

**Q. Why not zero/few-shot LLM prompting for intent?**
The Bitext corpus ships **gold intent labels** (27 fine intents), so this is a
supervised problem. A trained model is deterministic, free, instant at
inference, and its probabilities give a confidence gate. An LLM would add
latency, cost and variance for no gain on labelled data.

**Q. The brief suggests 7 routes; you trained on a different set — why?**
Two of the brief's suggested routes, `greeting/goodbye/gratitude` and
`out_of_scope`, have **no rows** in Bitext, so they cannot be supervised
classes. We handle them *outside* the model: small talk via a keyword rule,
out-of-scope via low classifier confidence / no-retrieval. Meanwhile Bitext
contains ~1k-row `shipping_address` and ~2k-row `contact/human-agent` intents
that a real bot must route, so we gave them dedicated routes. Result: **all 27
fine intents are covered** (checked programmatically) and every route maps to
one policy (RAG vs escalate vs no-RAG). See `FINE_TO_ROUTE` in config.

**Q. Why word **and** character TF-IDF features (`FeatureUnion`)?**
Word 1–2 grams carry the topic ("cancel", "refund"); character 2–4 grams make
the model robust to the corpus's deliberate typos ("oorder", "puchase") and
unknown wording. Two classifiers (MultinomialNB and SGD logistic) were compared
on a stratified held-out set; the better one by macro-F1 is kept. Reported
metrics are on a held-out 10% the final model never trained on (no leakage).

**Q. Can the classifier ever override a decisive "I want a human" request?**
We add a lightweight **escalation-phrase rule** on top of the classifier: if the
customer explicitly asks to talk to / be connected to a person, manager or
agent, the route is forced to `contact_support` no matter what the model says
(e.g. *"I was double-charged **and I want a manager**"* must not fall into plain
RAG billing handling). This mirrors how real support systems treat "escalate"
as a hard signal rather than a soft prediction.

**Q. Why is recall on `complaint`/`contact_support` important?**
Mis-routing those changes the response *policy* (escalation) rather than just
the wording, so we watch their per-class recall in the report, not only the
overall accuracy.

---

## 04 — Q&A RAG

**Q. Why FAISS locally instead of cloud Qdrant?**
Both are fine; we chose a local **FAISS `IndexFlatIP`** so the project runs with
no external account. Because embeddings are L2-normalised, inner product =
cosine similarity. 27k chunks fit trivially in a flat (exact) index — no ANN
approximation to explain away. Swapping to Qdrant later only replaces the
`index.add` / `index.search` calls.

**Q. What exactly is a knowledge-base "chunk"?**
A full Bitext `(instruction, response)` pair. We embed the *instruction*
(customer question) for retrieval and feed the paired *response* into the
generation prompt as grounding context. Keeping `intent`/`category` as metadata
lets us show *what* a retrieved hit is about and why the answer is grounded.

**Q. How do you stop hallucination / out-of-scope answers?**
Three mechanisms: (1) the generation system prompt is strict — answer **only**
from the retrieved context, and if the context doesn't cover the question say
so and offer a human; (2) a similarity threshold — below it we do **not** feed
off-topic chunks to the LLM and instead return the honest "can't help"
message; (3) complaint/human routes are escalated rather than generated.
The Bitext responses contain `{{placeholders}}` (slots); the retrieval-only
fallback path renders them into readable text (`deplaceholder`).

**Q. Why reply in the detected language? Where does language detection matter?**
Language detection decides the framing language (greetings, apologies, the
"can't help" message) and, with the LLM, we ask it to keep facts identical to
the English context but render them in the customer's language. Caveat (kept
honest in docs): the knowledge base itself is English-only, so fully grounded
multilingual answers would require a multilingual KB — noted as a limitation.

**Q. What if there is no Groq key?**
`GroqClient.generate` returns `ok=False` and the pipeline falls back to the
best retrieved support response (with deplaceholder rendering + tone
apology). The demo therefore never breaks and lights up the moment a key is
added — no code change.

**Q. How is the whole thing served?**
`Flask` (chosen over FastAPI because it is already lightweight and dependency-
free for this repo). `POST /chat` calls the *same* `SupportBot.answer()` the
notebooks use, so notebook behaviour == served behaviour. `/health` lists which
trained modules are present; if any are missing the API returns a 503 that says
which notebook to run.

---

## Reproducibility / environment notes

* Developed & smoke-verified on **Windows, Python 3.14, CPU-only PyTorch**.
* All runs seed everything with `SEED = 42`.
* Every notebook has a `SMOKE` switch: set `True` (or `PROJECT_SMOKE=1`) for a
  <1-minute check; default is the full run producing final artifacts.
* Notebooks are built from plain-text sources in `notebooks/_src/` via
  `tools/build_notebooks.py`, so diffs and review are easy; the `.ipynb` files
  are the deliverable and can be re-generated any time.
