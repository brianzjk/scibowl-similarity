const state = {
  corpus: { manifest: null, manifestUrl: null, categoryIndex: new Map(), categories: new Map() },
  upload: { bundle: null, embeddings: null },
  results: { corpus: [], within: [], filtered: [], currentView: "corpus", currentIndex: 0 },
  busy: false,
};

function setStatus(message, tone = "info") {
  const banner = document.getElementById("status-banner");
  banner.className = `status-banner${tone === "info" ? "" : ` ${tone}`}`;
  document.getElementById("status-message").textContent = message;
}

function setText(id, value) {
  document.getElementById(id).textContent = value ?? "";
}

function renderChoices(id, choices) {
  const element = document.getElementById(id);
  if (!choices || choices.length === 0) {
    element.textContent = "";
    element.style.display = "none";
    return;
  }
  element.textContent = choices.map((choice) => `${choice.label}) ${choice.text}`).join("\n");
  element.style.display = "block";
}

function getThreshold() {
  const value = Number.parseFloat(document.getElementById("threshold-input").value);
  return Number.isFinite(value) ? Math.min(Math.max(value, 0), 1) : 0.9;
}

function getTopK() {
  const value = Number.parseInt(document.getElementById("top-k-input").value, 10);
  return Number.isFinite(value) && value > 0 ? value : 10;
}

function setBusy(isBusy) {
  state.busy = isBusy;
  ["load-corpus", "analyze-button", "prev-button", "next-button"].forEach((id) => {
    document.getElementById(id).disabled = isBusy;
  });
}

function normalizeAnswerText(answerText) {
  return String(answerText || "")
    .replace(/^ANSWER:\s*/i, "")
    .replace(/^[WXYZ]\s*[\)\.]\s*/i, "")
    .trim()
    .toLowerCase();
}

function lexicalOverlapScore(textA, textB) {
  const tokensA = new Set((String(textA || "").toLowerCase().match(/[a-z0-9]+/g)) || []);
  const tokensB = new Set((String(textB || "").toLowerCase().match(/[a-z0-9]+/g)) || []);
  if (!tokensA.size || !tokensB.size) {
    return 0;
  }
  let overlap = 0;
  for (const token of tokensA) {
    if (tokensB.has(token)) {
      overlap += 1;
    }
  }
  return Number((overlap / (tokensA.size + tokensB.size - overlap)).toFixed(3));
}

function buildRoundLabel(question) {
  if (!question) {
    return null;
  }
  if (question.source_metadata && question.source_metadata.round != null) {
    return `Round ${question.source_metadata.round}`;
  }
  const rawFile = question.provenance && question.provenance.raw_file;
  if (!rawFile) {
    return null;
  }
  const filename = rawFile.split(/[\\/]/).pop() || rawFile;
  let stem = filename.replace(/\.[^.]+$/, "");
  if (stem.includes("__")) {
    stem = stem.split("__").slice(1).join("__");
  }
  return stem.replace(/_/g, " ").toUpperCase();
}

function questionOrigin(question) {
  return {
    tournament: (question.source_metadata && question.source_metadata.tournament) || question.source_id || null,
    round: buildRoundLabel(question),
  };
}

function flattenEmbeddings(rows, dimension) {
  const flat = new Float32Array(rows.length * dimension);
  rows.forEach((row, index) => {
    if (!Array.isArray(row) || row.length !== dimension) {
      throw new Error(`Embedding row ${index} does not match expected dimension ${dimension}.`);
    }
    flat.set(row, index * dimension);
  });
  return flat;
}

function dotProduct(flatA, indexA, flatB, indexB, dimension) {
  let score = 0;
  let offsetA = indexA * dimension;
  let offsetB = indexB * dimension;
  for (let d = 0; d < dimension; d += 1) {
    score += flatA[offsetA + d] * flatB[offsetB + d];
  }
  return score;
}

function insertTopMatch(best, entry, topK) {
  let inserted = false;
  for (let i = 0; i < best.length; i += 1) {
    if (entry.score > best[i].score) {
      best.splice(i, 0, entry);
      inserted = true;
      break;
    }
  }
  if (!inserted) {
    best.push(entry);
  }
  if (best.length > topK) {
    best.length = topK;
  }
}

function buildCandidate(questionA, questionB, score, preserveOrder) {
  const ordered = preserveOrder
    ? [questionA, questionB]
    : [questionA, questionB].sort((left, right) => String(left.question_id).localeCompare(String(right.question_id)));
  const first = ordered[0];
  const second = ordered[1];
  return {
    pair_id: `dup__${first.question_id}__${second.question_id}`,
    question_id_a: first.question_id,
    question_id_b: second.question_id,
    source_id_a: first.source_id,
    source_id_b: second.source_id,
    category: first.category,
    subcategory_a: first.subcategory,
    subcategory_b: second.subcategory,
    question_type_a: first.question_type,
    question_type_b: second.question_type,
    answer_mode_a: first.answer_mode,
    answer_mode_b: second.answer_mode,
    question_text_a: first.question_text,
    question_text_b: second.question_text,
    answer_text_a: first.answer_text,
    answer_text_b: second.answer_text,
    embedding_similarity: Number(score.toFixed(4)),
    lexical_overlap: lexicalOverlapScore(first.question_text, second.question_text),
    same_subcategory: String(first.subcategory || "").toLowerCase() === String(second.subcategory || "").toLowerCase(),
    same_question_type: first.question_type === second.question_type,
    same_answer_mode: first.answer_mode === second.answer_mode,
    same_normalized_answer: normalizeAnswerText(first.answer_text) === normalizeAnswerText(second.answer_text),
    question_a: first,
    question_b: second,
    origin_a: questionOrigin(first),
    origin_b: questionOrigin(second),
  };
}

function candidateLabel(view) {
  return view === "corpus" ? "upload_vs_corpus" : "within_upload";
}

function currentCandidate() {
  return state.results.filtered[state.results.currentIndex] ?? null;
}

function updateSummaryPills() {
  setText("corpus-status", state.corpus.manifest ? "yes" : "no");
  setText("upload-status", state.upload.bundle ? "yes" : "no");
  setText("corpus-match-count", state.results.corpus.length);
  setText("within-match-count", state.results.within.length);
  setText("hero-model", state.corpus.manifest ? state.corpus.manifest.model_name : "not loaded");
  setText("hero-count", state.corpus.manifest ? state.corpus.manifest.question_count : 0);
}

function populateCategoryFilter() {
  const select = document.getElementById("category-filter");
  const categories = new Set();
  for (const result of state.results[state.results.currentView]) {
    categories.add(result.category);
  }
  const previous = select.value;
  select.innerHTML = "";
  const allOption = document.createElement("option");
  allOption.value = "all";
  allOption.textContent = "All categories";
  select.appendChild(allOption);
  Array.from(categories).sort().forEach((category) => {
    const option = document.createElement("option");
    option.value = category;
    option.textContent = category;
    select.appendChild(option);
  });
  if (Array.from(select.options).some((option) => option.value === previous)) {
    select.value = previous;
  }
}

function renderEmptyCandidate() {
  setText("current-index", 0);
  setText("current-total", 0);
  setText("embedding-score", "0.0000");
  setText("lexical-score", "0.000");
  setText("same-answer", "false");
  setText("pair-type", "none");
  [
    "qid-a", "origin-a-tournament", "origin-a-round", "source-a", "category-a", "type-a", "answer-mode-a",
    "question-a", "answer-a", "qid-b", "origin-b-tournament", "origin-b-round", "source-b", "category-b",
    "type-b", "answer-mode-b", "question-b", "answer-b",
  ].forEach((id) => setText(id, ""));
  renderChoices("choices-a", []);
  renderChoices("choices-b", []);
}

function renderCurrentCandidate() {
  const candidate = currentCandidate();
  if (!candidate) {
    renderEmptyCandidate();
    return;
  }
  setText("current-index", state.results.filtered.length ? state.results.currentIndex + 1 : 0);
  setText("current-total", state.results.filtered.length);
  setText("embedding-score", candidate.embedding_similarity.toFixed(4));
  setText("lexical-score", candidate.lexical_overlap.toFixed(3));
  setText("same-answer", String(candidate.same_normalized_answer));
  setText("pair-type", candidateLabel(state.results.currentView));

  setText("qid-a", candidate.question_id_a);
  setText("origin-a-tournament", candidate.origin_a.tournament);
  setText("origin-a-round", candidate.origin_a.round);
  setText("source-a", candidate.source_id_a);
  setText("category-a", candidate.category);
  setText("type-a", candidate.question_type_a);
  setText("answer-mode-a", candidate.answer_mode_a);
  setText("question-a", candidate.question_text_a);
  setText("answer-a", candidate.answer_text_a);
  renderChoices("choices-a", candidate.question_a.choices || []);

  setText("qid-b", candidate.question_id_b);
  setText("origin-b-tournament", candidate.origin_b.tournament);
  setText("origin-b-round", candidate.origin_b.round);
  setText("source-b", candidate.source_id_b);
  setText("category-b", candidate.category);
  setText("type-b", candidate.question_type_b);
  setText("answer-mode-b", candidate.answer_mode_b);
  setText("question-b", candidate.question_text_b);
  setText("answer-b", candidate.answer_text_b);
  renderChoices("choices-b", candidate.question_b.choices || []);
}

function renderResultList() {
  const container = document.getElementById("result-list");
  const note = document.getElementById("result-list-note");
  container.innerHTML = "";

  if (state.results.filtered.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No matches satisfy the current filters.";
    container.appendChild(empty);
    note.textContent = "";
    return;
  }

  const limit = Math.min(state.results.filtered.length, 250);
  for (let index = 0; index < limit; index += 1) {
    const candidate = state.results.filtered[index];
    const button = document.createElement("button");
    button.className = `result-row${index === state.results.currentIndex ? " active" : ""}`;
    button.type = "button";
    button.addEventListener("click", () => {
      state.results.currentIndex = index;
      renderResultList();
      renderCurrentCandidate();
    });

    const head = document.createElement("div");
    head.className = "result-head";
    head.innerHTML = `<strong>${index + 1}. ${candidate.embedding_similarity.toFixed(4)}</strong><span class="result-sub">${candidate.category}</span>`;

    const sub = document.createElement("div");
    sub.className = "result-sub";
    sub.textContent = `${candidate.question_id_a} <-> ${candidate.question_id_b}`;

    const snippet = document.createElement("div");
    snippet.className = "result-sub";
    snippet.textContent = candidate.question_text_a.slice(0, 150);

    button.appendChild(head);
    button.appendChild(sub);
    button.appendChild(snippet);
    container.appendChild(button);
  }

  note.textContent = state.results.filtered.length > limit
    ? `Showing the first ${limit} filtered results. Use Prev/Next to continue through the full set.`
    : "";
}

function applyFilters() {
  const threshold = getThreshold();
  const category = document.getElementById("category-filter").value;
  const base = state.results[state.results.currentView];
  state.results.filtered = base.filter((candidate) => {
    if (candidate.embedding_similarity < threshold) {
      return false;
    }
    if (category !== "all" && candidate.category !== category) {
      return false;
    }
    return true;
  });
  state.results.currentIndex = Math.min(state.results.currentIndex, Math.max(state.results.filtered.length - 1, 0));
  renderResultList();
  renderCurrentCandidate();
}

async function yieldToBrowser() {
  await new Promise((resolve) => window.requestAnimationFrame(() => resolve()));
}

async function loadCorpusManifest() {
  const manifestUrl = document.getElementById("manifest-url").value.trim();
  if (!manifestUrl) {
    throw new Error("Corpus manifest URL is required.");
  }
  setStatus("Loading corpus manifest...");
  const response = await fetch(manifestUrl);
  if (!response.ok) {
    throw new Error(`Could not load corpus manifest: ${response.status}`);
  }
  const manifest = await response.json();
  if (!manifest || !Array.isArray(manifest.categories)) {
    throw new Error("Corpus manifest is missing its categories array.");
  }
  state.corpus.manifest = manifest;
  state.corpus.manifestUrl = new URL(manifestUrl, window.location.href).toString();
  state.corpus.categoryIndex = new Map();
  state.corpus.categories = new Map();
  manifest.categories.forEach((entry) => {
    state.corpus.categoryIndex.set(entry.category, entry);
  });
  window.localStorage.setItem("scibowl-similarity:manifest-url", manifestUrl);
  updateSummaryPills();
  setStatus("Corpus manifest loaded.");
}

async function loadCorpusCategory(category) {
  if (state.corpus.categories.has(category)) {
    return state.corpus.categories.get(category);
  }
  const manifestEntry = state.corpus.categoryIndex.get(category);
  if (!manifestEntry) {
    throw new Error(`Corpus manifest does not contain category ${category}.`);
  }
  const questionsUrl = new URL(manifestEntry.questions_path, state.corpus.manifestUrl).toString();
  const embeddingsUrl = new URL(manifestEntry.embeddings_path, state.corpus.manifestUrl).toString();
  const [questionsResponse, embeddingsResponse] = await Promise.all([fetch(questionsUrl), fetch(embeddingsUrl)]);
  if (!questionsResponse.ok || !embeddingsResponse.ok) {
    throw new Error(`Failed to load corpus data for ${category}.`);
  }
  const [questions, buffer] = await Promise.all([questionsResponse.json(), embeddingsResponse.arrayBuffer()]);
  const embeddings = new Float32Array(buffer);
  const dimension = state.corpus.manifest.embedding_dimension;
  if (embeddings.length !== manifestEntry.count * dimension) {
    throw new Error(`Embedding shard for ${category} has an unexpected length.`);
  }
  const payload = { questions, embeddings, count: manifestEntry.count };
  state.corpus.categories.set(category, payload);
  return payload;
}

async function readUploadBundle() {
  const file = document.getElementById("upload-bundle").files?.[0];
  if (!file) {
    throw new Error("Choose a local upload bundle JSON file first.");
  }
  setStatus(`Reading ${file.name}...`);
  const bundle = JSON.parse(await file.text());
  if (!bundle || !Array.isArray(bundle.questions) || !Array.isArray(bundle.embeddings)) {
    throw new Error("Upload bundle is missing questions or embeddings.");
  }
  state.upload.bundle = bundle;
  state.upload.embeddings = flattenEmbeddings(bundle.embeddings, bundle.embedding_dimension);
  updateSummaryPills();
  setStatus(`Loaded upload bundle with ${bundle.question_count} questions.`);
}

function validateBundleCompatibility() {
  if (!state.corpus.manifest) {
    throw new Error("Load the corpus manifest first.");
  }
  if (!state.upload.bundle) {
    throw new Error("Load a local upload bundle first.");
  }
  const bundle = state.upload.bundle;
  const manifest = state.corpus.manifest;
  if (bundle.model_name !== manifest.model_name) {
    throw new Error(`Model mismatch: upload bundle uses ${bundle.model_name}, corpus uses ${manifest.model_name}.`);
  }
  if (Boolean(bundle.include_answer) !== Boolean(manifest.include_answer)) {
    throw new Error("Upload bundle include-answer setting does not match the corpus bundle.");
  }
  if (bundle.embedding_dimension !== manifest.embedding_dimension) {
    throw new Error("Upload bundle embedding dimension does not match the corpus bundle.");
  }
}

function groupUploadIndicesByCategory(questions) {
  const groups = new Map();
  questions.forEach((question, index) => {
    const list = groups.get(question.category) || [];
    list.push(index);
    groups.set(question.category, list);
  });
  return groups;
}

async function computeWithinUploadMatches(questions, embeddings, dimension, threshold, topK) {
  const groups = groupUploadIndicesByCategory(questions);
  const pairMap = new Map();
  let processed = 0;

  for (const indices of groups.values()) {
    for (const indexA of indices) {
      const best = [];
      for (const indexB of indices) {
        if (indexA === indexB) {
          continue;
        }
        const score = dotProduct(embeddings, indexA, embeddings, indexB, dimension);
        if (score < threshold) {
          continue;
        }
        insertTopMatch(best, { index: indexB, score }, topK);
      }
      for (const match of best) {
        const questionA = questions[indexA];
        const questionB = questions[match.index];
        const key = [questionA.question_id, questionB.question_id].sort().join("__");
        if (!pairMap.has(key)) {
          pairMap.set(key, buildCandidate(questionA, questionB, match.score, false));
        }
      }
      processed += 1;
      if (processed % 25 === 0) {
        setStatus(`Computing within-upload matches... ${processed} / ${questions.length}`);
        await yieldToBrowser();
      }
    }
  }

  return Array.from(pairMap.values()).sort((left, right) => right.embedding_similarity - left.embedding_similarity);
}

async function computeCorpusMatches(questions, embeddings, dimension, threshold, topK) {
  const matches = [];
  let processed = 0;

  for (let indexA = 0; indexA < questions.length; indexA += 1) {
    const questionA = questions[indexA];
    const corpusCategory = await loadCorpusCategory(questionA.category);
    const best = [];
    for (let indexB = 0; indexB < corpusCategory.count; indexB += 1) {
      const score = dotProduct(embeddings, indexA, corpusCategory.embeddings, indexB, dimension);
      if (score < threshold) {
        continue;
      }
      insertTopMatch(best, { index: indexB, score }, topK);
    }
    for (const match of best) {
      matches.push(buildCandidate(questionA, corpusCategory.questions[match.index], match.score, true));
    }
    processed += 1;
    if (processed % 10 === 0) {
      setStatus(`Computing upload-vs-corpus matches... ${processed} / ${questions.length}`);
      await yieldToBrowser();
    }
  }

  return matches.sort((left, right) => right.embedding_similarity - left.embedding_similarity);
}

async function analyzeUpload() {
  if (!state.corpus.manifest) {
    await loadCorpusManifest();
  }
  if (!state.upload.bundle) {
    await readUploadBundle();
  }
  validateBundleCompatibility();

  const neededCategories = Array.from(new Set(state.upload.bundle.questions.map((question) => question.category))).sort();
  setStatus(`Loading ${neededCategories.length} corpus category shard(s)...`);
  for (const category of neededCategories) {
    await loadCorpusCategory(category);
  }

  const threshold = getThreshold();
  const topK = getTopK();
  const dimension = state.upload.bundle.embedding_dimension;
  const questions = state.upload.bundle.questions;
  const embeddings = state.upload.embeddings;

  state.results.currentIndex = 0;
  state.results.within = [];
  state.results.corpus = [];
  state.results.filtered = [];
  renderEmptyCandidate();
  renderResultList();

  state.results.within = await computeWithinUploadMatches(questions, embeddings, dimension, threshold, topK);
  state.results.corpus = await computeCorpusMatches(questions, embeddings, dimension, threshold, topK);
  updateSummaryPills();
  populateCategoryFilter();
  applyFilters();
  setStatus(
    `Analysis finished. Found ${state.results.corpus.length} upload-vs-corpus matches and ${state.results.within.length} within-upload matches.`
  );
}

function setView(view) {
  state.results.currentView = view;
  state.results.currentIndex = 0;
  populateCategoryFilter();
  applyFilters();
}

function move(delta) {
  if (!state.results.filtered.length) {
    return;
  }
  state.results.currentIndex = Math.min(Math.max(state.results.currentIndex + delta, 0), state.results.filtered.length - 1);
  renderResultList();
  renderCurrentCandidate();
}

async function handleLoadCorpus() {
  setBusy(true);
  try {
    await loadCorpusManifest();
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
}

async function handleAnalyze() {
  setBusy(true);
  try {
    await analyzeUpload();
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
}

document.getElementById("load-corpus").addEventListener("click", () => handleLoadCorpus());
document.getElementById("analyze-button").addEventListener("click", () => handleAnalyze());
document.getElementById("view-select").addEventListener("change", (event) => setView(event.target.value));
document.getElementById("category-filter").addEventListener("change", () => {
  state.results.currentIndex = 0;
  applyFilters();
});
document.getElementById("threshold-input").addEventListener("change", () => {
  state.results.currentIndex = 0;
  applyFilters();
});
document.getElementById("prev-button").addEventListener("click", () => move(-1));
document.getElementById("next-button").addEventListener("click", () => move(1));
document.getElementById("upload-bundle").addEventListener("change", async () => {
  setBusy(true);
  try {
    await readUploadBundle();
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

document.addEventListener("keydown", (event) => {
  if (state.busy) {
    return;
  }
  if (event.key.toLowerCase() === "j") {
    move(1);
  }
  if (event.key.toLowerCase() === "k") {
    move(-1);
  }
});

const savedManifestUrl = window.localStorage.getItem("scibowl-similarity:manifest-url");
if (savedManifestUrl) {
  document.getElementById("manifest-url").value = savedManifestUrl;
}

updateSummaryPills();
renderEmptyCandidate();
