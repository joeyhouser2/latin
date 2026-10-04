/* Latin Library — single-page frontend.
 *
 * No build step and no framework on purpose: this runs on one machine next to
 * a GPU, and a toolchain that has to be reinstalled before you can look at your
 * own corpus is a liability. Every view is a function that returns HTML; events
 * are delegated from one listener on <main>.
 */

const $ = (sel, root = document) => root.querySelector(sel);
const main = $('#main');

const state = {
  view: 'library',
  facets: null,
  filters: { q: '', language: '', stage: '', source_prefix: '', status: '', sort: 'author' },
  page: { offset: 0, limit: 50, total: 0 },
  docs: [],
  selected: new Set(),
  doc: null,            // open document (metadata + sections)
  segs: null,           // open document's loaded segments
  readerView: 'literal',
  readerSection: '',
  files: { root: 'data', path: '', entries: [], preview: null },
  catalog: { source: 'treatises', query: 'usury', items: [], loading: false },
  jobs: [],
  jobLog: null,         // { id, text }
  search: { q: '', items: [], message: '' },
  summ: { q: '', level: '', items: [], mode: '', note: '', stats: null, loading: false },
  docSumm: null,        // open document's summaries: { document, parts }
  readerOffset: 0,      // reader can open mid-document (a summary part's start)
  gpus: [],
  workerActive: true,
  llmModels: null,
  // Job options shared by every "queue this" button.
  opts: { batch_size: 16, max_length: 256, chunk: 200, preset: 'victorian_prose',
          backend: 'llm', skip_translated: false, not_before: '', gpu: 'any',
          summary_model: '', summary_source: 'both' },
};

// ---------------------------------------------------------------------------
// plumbing
// ---------------------------------------------------------------------------

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* not json */ }
    throw new Error(detail);
  }
  return res.json();
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function toast(message, bad = false) {
  const el = document.createElement('div');
  el.className = 'toast' + (bad ? ' bad' : '');
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), bad ? 7000 : 3500);
}

function num(n) { return (n ?? 0).toLocaleString(); }

function bar(done, total) {
  const pct = total ? Math.round(100 * done / total) : 0;
  const cls = pct >= 100 ? '' : (pct > 0 ? ' partial' : ' none');
  return `<div class="bar${cls}" title="${num(done)} / ${num(total)}">
            <span style="width:${pct}%"></span></div>`;
}

/* The scheduling controls, shared by every view that can queue work. The
 * datetime-local value is local time; the API wants UTC ISO-8601. */
function gpuControl() {
  const opts = [['any', 'auto (freest card)']];
  if (state.gpus.length > 1) opts.push(['alternate', 'alternate cards (one doc per card)']);
  state.gpus.forEach(g => opts.push([g.index,
    `GPU ${g.index} · ${g.name.replace('NVIDIA GeForce ', '')}`]));
  return `<label class="inline" title="Which card a queued job runs on. Two jobs on different single documents can run at the same time, one per card.">GPU
            <select data-opt="gpu">${opts.map(([v, t]) =>
              `<option value="${esc(v)}" ${String(state.opts.gpu) === v ? 'selected' : ''}>${esc(t)}</option>`).join('')}
            </select></label>`;
}

function scheduleControls() {
  return `${gpuControl()}<label class="inline">start after
            <input type="datetime-local" data-opt="not_before" value="${esc(state.opts.not_before)}">
          </label>`;
}

function notBefore() {
  if (!state.opts.not_before) return null;
  return new Date(state.opts.not_before).toISOString();
}

async function queueJob(kind, params, label = '') {
  if (params.gpu === 'alternate') params = { ...params, gpu: undefined };   // only meaningful in bulk
  try {
    const job = await api('/api/jobs', {
      method: 'POST',
      body: { kind, params, label, not_before: notBefore() },
    });
    toast(`Queued #${job.id}: ${job.label}`);
    refreshJobs();
  } catch (e) { toast(e.message, true); }
}

async function queueBulk(kind, docIds, params = {}) {
  try {
    const out = await api('/api/jobs/bulk', {
      method: 'POST',
      body: { kind, doc_ids: docIds, params, not_before: notBefore() },
    });
    toast(`Queued ${out.created} job(s)`);
    refreshJobs();
  } catch (e) { toast(e.message, true); }
}

// ---------------------------------------------------------------------------
// view: documents
// ---------------------------------------------------------------------------

async function loadDocs() {
  const params = new URLSearchParams({
    ...state.filters, offset: state.page.offset, limit: state.page.limit,
  });
  const data = await api('/api/documents?' + params);
  state.docs = data.items;
  state.page.total = data.total;
}

function viewLibrary() {
  const f = state.filters, facets = state.facets || {};
  const opt = (list, value, fmt = v => v) => list.map(x =>
    `<option value="${esc(x.value ?? x)}" ${(x.value ?? x) === value ? 'selected' : ''}>
       ${esc(fmt(x))}</option>`).join('');

  const rows = state.docs.map(d => `
    <tr data-doc="${d.id}">
      <td><input type="checkbox" data-select="${d.id}" ${state.selected.has(d.id) ? 'checked' : ''}></td>
      <td class="title">
        <a data-open="${d.id}">${esc(d.title)}</a>
        <div class="sub">${esc(d.author || 'Anon.')}${d.century ? ' · ' + centuryLabel(d.century) : ''}
             · ${esc((d.source || '').slice(0, 42))}</div>
      </td>
      <td><span class="badge ${esc(d.language)}">${d.language === 'grc' ? 'Greek' : 'Latin'}</span></td>
      <td class="muted">${esc((d.language_stage || '').replace('_', ' '))}</td>
      <td class="num">${num(d.segments)}</td>
      <td>${bar(d.translated, d.segments)}</td>
      <td class="num muted">${d.segments ? Math.round(d.percent) + '%' : '—'}</td>
      <td class="num muted">${num(d.styled)}</td>
      <td><button class="btn ghost" data-queue-doc="${d.id}">Translate</button></td>
    </tr>`).join('');

  const from = state.page.offset + 1;
  const to = Math.min(state.page.offset + state.page.limit, state.page.total);

  return `
    <h2>Documents</h2>
    <div class="row">
      <input type="search" data-filter="q" value="${esc(f.q)}" placeholder="title, author or source…">
      <select data-filter="language">
        <option value="">any language</option>
        ${opt((facets.languages || []), f.language, x => (x.value === 'grc' ? 'Greek' : 'Latin') + ` (${x.n})`)}
      </select>
      <select data-filter="stage">
        <option value="">any era</option>
        ${opt((facets.stages || []), f.stage, x => x.value.replace('_', ' ') + ` (${x.n})`)}
      </select>
      <select data-filter="source_prefix">
        <option value="">any source</option>
        ${opt((facets.sources || []), f.source_prefix, x => `${x.value} (${x.n})`)}
      </select>
      <select data-filter="status">
        <option value="">any progress</option>
        ${opt([{ value: 'untranslated' }, { value: 'partial' }, { value: 'translated' },
               { value: 'unstyled' }, { value: 'empty' }], f.status, x => x.value)}
      </select>
      <select data-filter="sort">
        ${opt([{ value: 'author' }, { value: 'title' }, { value: 'id' }, { value: 'newest' }],
              f.sort, x => 'sort: ' + x.value)}
      </select>
    </div>

    <div class="row">
      <b>${num(state.page.total)}</b> <span class="muted">documents match</span>
      <span class="muted">·</span>
      <button class="btn ghost" data-select-page>select page</button>
      <button class="btn ghost" data-select-none>clear (${state.selected.size})</button>
      <span class="muted">·</span>
      <button class="btn" data-queue-selected="translate">Translate selected</button>
      <button class="btn ghost" data-queue-selected="stylize">Stylize selected</button>
      <button class="btn ghost" data-summarize-selected>Summarize selected</button>
      <button class="btn ghost" data-regerman-selected
              title="Re-translate the German editorial notes (e.g. Analecta Hymnica apparatus) that were translated as if they were Latin. Latin segments are not touched.">Re-translate German in selected</button>
      <button class="btn" data-queue-filter="translate">Translate everything matching</button>
      ${scheduleControls()}
    </div>

    <table class="grid">
      <thead><tr>
        <th></th><th>Work</th><th>Lang</th><th>Era</th><th class="num">Segs</th>
        <th>Translated</th><th class="num">%</th><th class="num">Styled</th><th></th>
      </tr></thead>
      <tbody>${rows || '<tr><td colspan="9" class="muted">No documents match.</td></tr>'}</tbody>
    </table>

    <div class="row" style="margin-top:0.7rem">
      <button class="btn ghost" data-page="-1" ${state.page.offset === 0 ? 'disabled' : ''}>← prev</button>
      <span class="muted">${num(from)}–${num(to)} of ${num(state.page.total)}</span>
      <button class="btn ghost" data-page="1" ${to >= state.page.total ? 'disabled' : ''}>next →</button>
      <select data-page-size>
        ${[25, 50, 100, 200].map(n => `<option ${n === state.page.limit ? 'selected' : ''}>${n}</option>`).join('')}
      </select>
    </div>

    <p class="muted" style="margin-top:0.8rem">
      “Translate everything matching” queues <b>one</b> job covering the whole filter when the
      filter is one the batch script understands (language and/or source); otherwise it queues
      one job per document, which is slower to start but scoped exactly.
    </p>`;
}

function centuryLabel(c) { return c < 0 ? `${Math.abs(c)}c BCE` : `${c}c CE`; }

// ---------------------------------------------------------------------------
// view: reader
// ---------------------------------------------------------------------------

async function openDoc(docId, sectionId = '', offset = 0) {
  state.doc = await api(`/api/documents/${docId}`);
  state.readerSection = sectionId;
  state.readerOffset = sectionId ? 0 : offset;
  const q = new URLSearchParams({ limit: 400, offset: state.readerOffset });
  if (sectionId) q.set('section_id', sectionId);
  const [segs, summ] = await Promise.all([
    api(`/api/documents/${docId}/segments?` + q),
    api(`/api/documents/${docId}/summaries`).catch(() => null),
  ]);
  state.segs = segs;
  state.docSumm = summ;
  state.view = 'reader';
  render();
  window.scrollTo(0, 0);
}

/* The document's summary, above the text, with its parts as jump links. */
function summaryCard(d) {
  const s = state.docSumm;
  if (!s || !s.document) {
    return `<div class="card summary-card muted">No summary yet.
      ${d.translated ? '<button class="btn ghost" data-summarize-doc>Summarize</button>'
                     : `<button class="btn ghost" data-summarize-preview
                          title="Reads a sample of the original text only; no translation needed.">Preview: what is this work?</button>
                        <span class="muted">Reads a sample of the original and says what the work is or may be.</span>`}
    </div>`;
  }
  const doc = s.document;
  const preview = doc.source_mode === 'original';
  const stale = preview
    ? `<span class="badge queued" title="Written from a sample of the untranslated original, with hedging. Replaced by a full summary once the work is translated.">preview</span>`
    : doc.translated_count !== d.translated
    ? `<span class="badge failed" title="More segments have been translated since this was written">out of date</span>` : '';
  const parts = (s.parts || []).map(p => `
    <li><a data-jump="${p.seg_offset_first}">Part ${p.part_index + 1}</a>
      <span class="muted">· segments ${num(p.seg_offset_first + 1)}–${num(p.seg_offset_last + 1)}</span>
      — ${esc(p.summary)}</li>`).join('');
  return `
    <div class="card summary-card">
      <h4>Summary <span class="muted">· ${esc(doc.model)} · ${esc((doc.created_at || '').slice(0, 10))}</span>
        ${stale} <button class="btn ghost" data-summarize-doc>re-summarize</button></h4>
      <p>${esc(doc.summary)}</p>
      <div>${(doc.topics || []).map(t => `<span class="chip" data-topic="${esc(t)}">${esc(t)}</span>`).join('')}</div>
      ${parts ? `<details><summary>${s.parts.length} part summaries</summary><ol class="parts">${parts}</ol></details>` : ''}
    </div>`;
}

function viewReader() {
  const d = state.doc;
  if (!d) return `<h2>Reader</h2><p class="muted">Pick a document from the Documents tab.</p>`;
  const styled = state.readerView === 'styled';

  let lastSection = null;
  const rows = (state.segs?.items || []).map(s => {
    let head = '';
    if (s.section !== lastSection) {
      lastSection = s.section;
      head = `<tr><td colspan="2" class="secmark">${esc(s.section)}</td></tr>`;
    }
    let en;
    if (styled && s.styled) en = `<td class="en styled">${esc(s.styled)}</td>`;
    else if (s.english) en = `<td class="en">${esc(s.english)}</td>`;
    else en = `<td class="en"><span class="untranslated">— not yet translated —</span></td>`;
    const scan = s.scansion ? `<span class="scansion">${esc(s.scansion)}</span>` : '';
    return head + `<tr><td class="la">${esc(s.latin)}${scan}</td>${en}</tr>`;
  }).join('');

  const secs = d.sections || [];
  const sectionOptions = secs.map(s =>
    `<option value="${s.id}" ${String(s.id) === String(state.readerSection) ? 'selected' : ''}>
       ${s.number}. ${esc(s.label)} (${num(s.translated)}/${num(s.segments)})</option>`).join('');

  const loaded = state.segs ? state.segs.items.length + state.readerOffset : 0;
  const total = state.segs ? state.segs.total : 0;

  return `
    <h2>${esc(d.title)}</h2>
    <p class="muted">${esc(d.author || 'Anon.')} · ${esc((d.language_stage || '').replace('_', ' '))}
       · ${esc(d.source || '')} · ${num(d.segments)} segments, ${num(d.translated)} translated
       · <a href="/api/documents/${d.id}/export?column=both" target="_blank">export .txt</a></p>
    ${translationLine(d)}

    <div class="row">
      <select data-reader-view>
        <option value="literal" ${styled ? '' : 'selected'}>literal English</option>
        <option value="styled" ${styled ? 'selected' : ''}>stylized English</option>
      </select>
      <select data-reader-section>
        <option value="">whole work (${secs.length} sections)</option>
        ${sectionOptions}
      </select>
      <button class="btn" data-translate-doc>Translate this document</button>
      ${secs.length > 1 ? `
        <label class="inline">sections
          <input type="number" min="1" max="${secs.length}" value="1" data-sec-first style="width:4.5rem">
          –
          <input type="number" min="1" max="${secs.length}" value="${Math.min(5, secs.length)}" data-sec-last style="width:4.5rem">
        </label>
        <button class="btn ghost" data-translate-range>Translate range</button>` : ''}
      <button class="btn ghost" data-stylize-doc>Stylize</button>
      ${d.translated ? `<button class="btn ghost" data-summarize-doc>Summarize</button>` : `<button class="btn ghost" data-summarize-preview>Preview summary</button>`}
      ${d.language === 'la' ? `<button class="btn ghost" data-regerman-doc
          title="Re-translate German editorial notes that were translated as if they were Latin. Latin segments are not touched.">Re-translate German</button>` : ''}
      ${scheduleControls()}
    </div>

    ${summaryCard(d)}
    ${state.readerOffset ? `<p class="muted">Starting at segment ${num(state.readerOffset + 1)}.
        <a data-jump="0">Go to the beginning</a></p>` : ''}
    <div class="reader"><table><tbody>${rows}</tbody></table></div>
    <div class="row" style="margin-top:0.6rem">
      <span class="muted">showing ${num(loaded)} of ${num(total)} segments</span>
      ${loaded < total ? '<button class="btn ghost" data-more>load more</button>' : ''}
    </div>`;
}

/* Whether an English translation already exists, and on what evidence. The
 * evidence is shown, not just the verdict: "untranslated" is a claim this
 * library makes, and the reader should be able to check it. */
function translationLine(d) {
  const labels = {
    translated: 'English translation exists (freely available)',
    translated_paywalled: 'English translation exists (in copyright)',
    untranslated: 'No English translation known',
    unknown: 'Translation status unknown',
  };
  const cls = { translated: 'done', translated_paywalled: 'queued',
                untranslated: 'running', unknown: '' }[d.translation_status] || '';
  return `<p class="muted" style="margin-top:-0.4rem">
    <span class="badge ${cls}">${esc(labels[d.translation_status] || d.translation_status)}</span>
    ${d.translation_evidence ? ' ' + esc(d.translation_evidence) : ''}</p>`;
}

async function loadMoreSegments() {
  const q = new URLSearchParams({ limit: 400, offset: state.readerOffset + state.segs.items.length });
  if (state.readerSection) q.set('section_id', state.readerSection);
  const next = await api(`/api/documents/${state.doc.id}/segments?` + q);
  state.segs.items = state.segs.items.concat(next.items);
  render();
}

// ---------------------------------------------------------------------------
// view: files
// ---------------------------------------------------------------------------

async function loadFiles(root = state.files.root, path = '') {
  const data = await api(`/api/files?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`);
  state.files.root = root;
  state.files.path = path;
  state.files.entries = data.entries || [];
  state.files.roots = data.roots || [root];
}

function viewFiles() {
  const f = state.files;
  const crumbs = ['', ...f.path.split('/').filter(Boolean)];
  const trail = crumbs.map((c, i) => {
    const target = crumbs.slice(1, i + 1).join('/');
    return `<a data-cd="${esc(target)}">${esc(c || f.root)}</a>`;
  }).join(' / ');

  const rows = f.entries.map(e => `
    <tr>
      <td class="title">
        ${e.is_dir
          ? `<a data-cd="${esc(e.path)}">📁 ${esc(e.name)}</a>`
          : `<a data-peek="${esc(e.path)}">${esc(e.name)}</a>`}
      </td>
      <td class="muted">${esc(e.kind)}</td>
      <td class="num muted">${e.is_dir ? '' : humanSize(e.size)}</td>
      <td class="muted">${esc(e.modified.replace('T', ' ').replace('+00:00', ''))}</td>
      <td>${e.downloadable
            ? `<a class="btn ghost" href="/api/files/download?root=${encodeURIComponent(f.root)}&path=${encodeURIComponent(e.path)}">download</a>`
            : '<span class="muted">—</span>'}</td>
    </tr>`).join('');

  const preview = f.preview ? `
    <div class="card">
      <h4>${esc(f.preview.path || '')}
        ${f.preview.truncated ? '<span class="muted">(first 200 KB)</span>' : ''}</h4>
      ${f.preview.error ? `<p class="error">${esc(f.preview.error)}</p>` : ''}
      ${f.preview.text ? `<pre class="preview">${esc(f.preview.text.slice(0, 60000))}</pre>` : ''}
    </div>` : '<p class="muted">Click a file to preview it.</p>';

  return `
    <h2>Files</h2>
    <div class="row">
      ${(f.roots || ['data']).map(r =>
        `<button class="chip ${r === f.root ? 'active' : ''}" data-root="${esc(r)}">${esc(r)}/</button>`).join('')}
    </div>
    <div class="crumbs">${trail}</div>
    <div class="split">
      <div class="left">
        <table class="grid">
          <thead><tr><th>Name</th><th>Kind</th><th class="num">Size</th><th>Modified (UTC)</th><th></th></tr></thead>
          <tbody>${rows || '<tr><td colspan="5" class="muted">Empty.</td></tr>'}</tbody>
        </table>
        <p class="muted" style="margin-top:0.6rem">
          SQLite and FAISS files are listed but never served — copying <code>corpus.db</code>
          while a job holds it open produces a torn file. Use the sqlite backup API instead.
        </p>
      </div>
      <div class="right">${preview}</div>
    </div>`;
}

function humanSize(n) {
  if (n < 1024) return n + ' B';
  if (n < 1048576) return (n / 1024).toFixed(0) + ' KB';
  if (n < 1073741824) return (n / 1048576).toFixed(1) + ' MB';
  return (n / 1073741824).toFixed(2) + ' GB';
}

// ---------------------------------------------------------------------------
// view: catalog (find texts to ingest)
// ---------------------------------------------------------------------------

/* Quick queries per source. For the two sources that verify translation
 * status, "untranslated" is the useful one: it lists only works with no known
 * English translation, each with the evidence for that claim. */
const THEMES = {
  treatises: ['usury', 'money', 'exchange', 'commerce', 'tax', 'weights', 'accounting', 'economy'],
  capitularia: ['untranslated', 'pre814', 'ldf', 'post840', 'all'],
  celt: ['untranslated', 'all'],
  sutton: ['Filelfo', 'Biondo', 'Salutati', 'Poliziano', 'Pontano', 'Bruni', 'epistolography', 'fetchable'],
  pg_corpus: ['chrysostom', 'genesis', 'psalms', 'all'],
};
const THEME_LABELS = { pre814: 'to 814', ldf: 'Louis the Pious 814–840', post840: 'after 840' };

function viewCatalog() {
  const c = state.catalog;
  const rows = c.items.map((r, i) => `
    <tr>
      <td class="title">
        ${r.url ? `<a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.title || r.identifier)}</a>`
                : esc(r.title || r.identifier)}
        <div class="sub">${esc(r.author || '')}${r.publisher ? ' · ' + esc(r.publisher) : ''}
          ${r.note ? ' · <i>' + esc(r.note) + '</i>' : ''}</div>
      </td>
      <td class="muted">${esc(r.catalogue || '')}</td>
      <td class="num muted">${esc(r.year ?? '')}</td>
      <td>${r.fetchable ? '<span class="badge done">text</span>'
                        : '<span class="badge">catalogue only</span>'}</td>
      <td>${r.translation_status
            ? `<span class="badge ${{translated: 'done', translated_paywalled: 'queued',
                 untranslated: 'running'}[r.translation_status] || ''}"
                 title="${esc(r.translation_evidence || '')}">${esc(r.translation_status.replace('_', ' '))}</span>
               <div class="sub" style="max-width:340px">${esc((r.translation_evidence || '').slice(0, 140))}</div>`
            : ''}</td>
      <td>${r.fetchable
            ? `<button class="btn ghost" data-ingest="${i}">Queue ingest</button>`
            : ''}</td>
    </tr>`).join('');

  const fetchable = c.items.filter(r => r.fetchable).length;

  return `
    <h2>Find texts</h2>
    <p class="muted">Search a connector's catalogue without ingesting anything, then queue
       the ones you want. <b>capitularia</b> (Frankish royal legislation, 507–9th c.) and
       <b>celt</b> (Hiberno-Latin) check whether each work already has an English translation
       and show the evidence; pick <i>untranslated</i> to see only the ones that don't.
       <b>treatises</b> searches archive.org and Gallica (catalogue only). <b>sutton</b> searches Dana Sutton's bibliography of ~54,000 online Neo-Latin texts (only the archive.org entries have text; add <i>fetchable</i> to the query to see just those). <b>pg_corpus</b> lists Migne's Patrologia Graeca volumes, e.g. Chrysostom's homilies.</p>

    <div class="row">
      <select data-cat-source>
        ${(state.facets?.sources_available || ['treatises']).map(s =>
          `<option ${s === c.source ? 'selected' : ''}>${esc(s)}</option>`).join('')}
      </select>
      <input type="search" data-cat-query value="${esc(c.query)}" placeholder="theme name or free text…">
      <button class="btn" data-cat-go ${c.loading ? 'disabled' : ''}>
        ${c.loading ? 'searching…' : 'Search'}</button>
      <label class="inline">limit <input type="number" data-cat-limit value="25" style="width:4.5rem"></label>
    </div>
    ${THEMES[c.source] ? `<div class="row">${THEMES[c.source].map(t =>
        `<button class="chip ${t === c.query ? 'active' : ''}" data-theme="${t}">${esc(THEME_LABELS[t] || t)}</button>`).join('')}</div>` : ''}

    <div class="row">
      <b>${c.items.length}</b><span class="muted">results, ${fetchable} with text</span>
      ${fetchable ? '<button class="btn" data-ingest-all>Queue all fetchable</button>' : ''}
      ${scheduleControls()}
    </div>

    <table class="grid">
      <thead><tr><th>Work</th><th>Catalogue</th><th class="num">Year</th><th>Text</th>
        <th>English translation?</th><th></th></tr></thead>
      <tbody>${rows || `<tr><td colspan="6" class="muted">${c.loading ? 'Searching…' : 'No results yet.'}</td></tr>`}</tbody>
    </table>`;
}

async function runCatalogSearch() {
  state.catalog.loading = true; render();
  try {
    const limit = parseInt($('[data-cat-limit]')?.value || '25', 10);
    const data = await api('/api/discover', {
      method: 'POST',
      body: { source: state.catalog.source, query: state.catalog.query, limit },
    });
    state.catalog.items = data.items || [];
  } catch (e) {
    state.catalog.items = [];
    toast(e.message, true);
  } finally {
    state.catalog.loading = false; render();
  }
}

// ---------------------------------------------------------------------------
// view: jobs
// ---------------------------------------------------------------------------

async function refreshJobs() {
  try {
    const data = await api('/api/jobs?limit=60');
    const firstGpus = !state.gpus.length && (data.gpus || []).length;
    state.jobs = data.items;
    state.gpus = data.gpus || [];
    state.workerActive = data.worker_active !== false;
    renderChip();
    // Patch only the live regions (cards, table, log), never the whole view: a
    // full re-render would steal focus and wipe whatever is half-typed in the
    // defaults card, which used to freeze progress while any field was focused.
    if (state.view === 'jobs' && $('#jobs-table')) patchJobs();
    else if (firstGpus && !typing()) render();      // GPU pickers need the card list
  } catch (e) { /* the page still works without the queue */ }
}

function patchJobs() {
  $('#jobs-gpus').innerHTML = gpuPanel();
  $('#jobs-table').innerHTML = jobsTable();
  if (state.jobLog) refreshLog();
}

async function refreshLog() {
  const id = state.jobLog.id;
  try {
    const data = await api(`/api/jobs/${id}/log`);
    if (!state.jobLog || state.jobLog.id !== id) return;
    state.jobLog.text = data.log;
    const pre = $('#jobs-log pre.log');
    if (!pre) return;
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 24;
    pre.textContent = data.log || '(empty)';
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  } catch (e) { /* job row may have been cleared */ }
}

/* Top-right: what is running right now, on every page. */
function eta(j) {
  const t0 = Date.parse(j.started_at || '');
  if (!t0 || !j.done || !j.total) return '';
  const secs = (Date.now() - t0) / 1000;
  const left = (j.total - j.done) / (j.done / secs);
  if (!isFinite(left)) return '';
  if (left < 90) return '<1 min left';
  if (left < 5400) return Math.round(left / 60) + ' min left';
  if (left < 172800) return (left / 3600).toFixed(1) + ' h left';
  return Math.round(left / 86400) + ' d left';
}

function renderChip() {
  const el = $('#jobchip');
  if (!el) return;
  const running = state.jobs.filter(j => j.status === 'running');
  const queued = state.jobs.filter(j => j.status === 'queued').length;
  if (!running.length && !queued) { el.hidden = true; main.style.paddingTop = ''; return; }
  el.hidden = false;
  const verb = { translate: 'Translating', stylize: 'Stylizing', summarize: 'Summarizing',
                 ingest: 'Ingesting', reindex: 'Re-indexing' };
  el.innerHTML = running.map(j => `
      <div class="jc-job" data-view="jobs" title="Open the Jobs page">
        <div class="jc-title"><b>${esc(verb[j.kind] || j.kind)}</b> ${esc(j.title || j.label)}</div>
        ${bar(j.done, j.total)}
        <div class="muted jc-foot">${j.gpu != null ? `GPU ${esc(j.gpu)} · ` : ''}${j.total ? `${num(j.done)} / ${num(j.total)} · ${Math.floor(100 * j.done / j.total)}%` : 'starting…'}${eta(j) ? ' · ' + eta(j) : ''}</div>
      </div>`).join('') +
    (queued ? `<div class="muted jc-queued" data-view="jobs">${queued} queued${running.length ? '' : ' — waiting for a free slot'}</div>` : '');
  // Keep the page's own header row clear of the chip.
  main.style.paddingTop = Math.max(0, el.offsetHeight - 6) + 'px';
}

function typing() {
  const a = document.activeElement;
  return !!a && ['INPUT', 'SELECT', 'TEXTAREA'].includes(a.tagName) && main.contains(a);
}

function gpuPanel() {
  if (!state.gpus.length) return '<p class="muted">No NVIDIA GPU detected — jobs run on the CPU, one at a time.</p>';
  return `<div class="row">${state.gpus.map(g => {
    const job = state.jobs.find(j => j.id === g.job_id);
    return `<div class="card gpu">
      <b>GPU ${esc(g.index)}</b> <span class="muted">${esc(g.name.replace('NVIDIA GeForce ', ''))}</span><br>
      ${bar(g.memory_used, g.memory_total)}
      <span class="muted">${(g.memory_used / 1024).toFixed(1)} / ${Math.round(g.memory_total / 1024)} GB</span><br>
      ${job ? `running <a data-joblog="${job.id}">#${job.id}</a> ${esc(job.kind)}` : '<span class="muted">idle</span>'}
    </div>`;
  }).join('')}</div>`;
}

function jobsTable() {
  const rows = state.jobs.map(j => `
    <tr>
      <td class="num muted">#${j.id}</td>
      <td class="title"><a data-joblog="${j.id}">${esc(j.label)}</a>
        <div class="sub">${esc(j.kind)}${j.not_before ? ' · not before ' + esc(j.not_before.replace('T', ' ')) : ''}
          ${j.error ? ' · <span class="error">' + esc(j.error) + '</span>' : ''}
          ${j.note ? ' · <span class="muted">' + esc(j.note) + '</span>' : ''}</div></td>
      <td><span class="badge ${esc(j.status)}">${esc(j.status)}</span></td>
      <td class="muted">${j.gpu != null ? 'GPU ' + esc(j.gpu)
          : (j.params && j.params.gpu && j.status === 'queued' ? '→ GPU ' + esc(j.params.gpu) : '')}</td>
      <td>${j.total ? bar(j.done, j.total) : ''}</td>
      <td class="num muted">${j.total ? num(j.done) + '/' + num(j.total) : ''}</td>
      <td class="muted">${esc((j.started_at || j.created_at || '').replace('T', ' ').replace('+00:00', ''))}</td>
      <td>
        ${['queued', 'running'].includes(j.status)
          ? `<button class="btn ghost" data-cancel="${j.id}">cancel</button>`
          : `<button class="btn ghost" data-requeue="${j.id}">requeue</button>`}
      </td>
    </tr>`).join('');

  return `
    <table class="grid">
      <thead><tr><th class="num">#</th><th>Job</th><th>Status</th><th>On</th><th>Progress</th>
        <th class="num"></th><th>Started (UTC)</th><th></th></tr></thead>
      <tbody>${rows || '<tr><td colspan="8" class="muted">Nothing queued.</td></tr>'}</tbody>
    </table>`;
}

function jobLogCard() {
  return state.jobLog ? `
    <div class="card">
      <h4>Log — job #${state.jobLog.id} <span class="muted">(updates live)</span>
        <button class="btn ghost" data-joblog-close>close</button></h4>
      <pre class="log">${esc(state.jobLog.text || '(empty)')}</pre>
    </div>` : '';
}

function viewJobs() {
  return `
    <h2>Jobs</h2>
    <p class="muted">One job per GPU, pinned to its card, with one rule on top: only one job at a
       time may write the corpus. So translations queue behind each other, while a summary runs
       on whichever card the translation isn't using. Closing this page does not stop a job;
       closing the server's console window does — it comes back as <b>interrupted</b>, and
       requeueing resumes from the last commit.</p>
    ${state.workerActive ? '' : `<p class="error">Another copy of the app is running and owns the
       queue — jobs you add here will run there. Close the extra copy to avoid confusion.</p>`}
    <div id="jobs-gpus">${gpuPanel()}</div>
    <div class="row">
      <button class="btn ghost" data-jobs-refresh>refresh</button>
      <button class="btn ghost" data-jobs-clear>clear finished</button>
    </div>

    <div class="card">
      <h4>Defaults for jobs you queue from now on</h4>
      <div class="row">
        <label class="inline">batch size
          <input type="number" min="1" max="128" data-opt="batch_size"
                 value="${state.opts.batch_size}" style="width:5rem"></label>
        <label class="inline">max length
          <input type="number" min="32" max="1024" data-opt="max_length"
                 value="${state.opts.max_length}" style="width:5.5rem"></label>
        <label class="inline">commit every
          <input type="number" min="10" max="5000" data-opt="chunk"
                 value="${state.opts.chunk}" style="width:5.5rem"> segments</label>
        <label class="inline">stylizer
          <select data-opt="preset">
            ${(state.facets?.presets || ['victorian_prose']).map(p =>
              `<option ${p === state.opts.preset ? 'selected' : ''}>${esc(p)}</option>`).join('')}
          </select></label>
        <label class="inline">engine
          <select data-opt="backend">
            <option value="llm" ${state.opts.backend === 'llm' ? 'selected' : ''}>llm (rich)</option>
            <option value="t5" ${state.opts.backend === 't5' ? 'selected' : ''}>t5 (fast Victorian)</option>
          </select></label>
        ${gpuControl()}
        <label class="inline">summarizer
          <select data-opt="summary_model">
            <option value="">default (gemma4:12b)</option>
            ${(state.llmModels || []).map(m =>
              `<option value="${esc(m.name)}" ${m.name === state.opts.summary_model ? 'selected' : ''}>
                 ${esc(m.name)} (${m.size_gb} GB)</option>`).join('')}
          </select></label>
        <label class="inline">reads
          <select data-opt="summary_source">
            <option value="both" ${state.opts.summary_source === 'both' ? 'selected' : ''}>original + translation</option>
            <option value="english" ${state.opts.summary_source === 'english' ? 'selected' : ''}>translation only (faster)</option>
            <option value="original" ${state.opts.summary_source === 'original' ? 'selected' : ''}>original only — preview, no translation needed</option>
          </select></label>
      </div>
      <p class="muted" style="margin:0">Lower the batch size or max length if a job dies
         with a CUDA OOM; the commit interval is how much work an interrupted run repeats.</p>
    </div>
    <div id="jobs-table">${jobsTable()}</div>
    <div id="jobs-log">${jobLogCard()}</div>`;
}

// ---------------------------------------------------------------------------
// view: summaries
// ---------------------------------------------------------------------------

/* HTML-escape first, then turn the server's control-character highlight
 * sentinels into <mark>. Escaping cannot touch the sentinels, and nothing in
 * the text can forge a tag. */
function hl(text, marks) {
  const [o, c] = marks || ['\u0002', '\u0003'];
  return esc(text).split(o).join('<mark>').split(c).join('</mark>');
}

function viewSummaries() {
  const s = state.summ, st = s.stats || state.facets?.summaries || {};
  const items = s.items.map(h => {
    const where = h.level === 'part'
      ? `Part ${h.part_index + 1} of ${h.part_count} · segments ${num(h.seg_offset_first + 1)}–${num(h.seg_offset_last + 1)}`
      : 'Whole work';
    const body = h.snippet ? hl(h.snippet, s.hl) : esc(h.summary);
    return `
      <div class="card">
        <div class="muted" style="font-size:0.78rem">
          <a data-open-at="${h.doc_id}" data-offset="${h.level === 'part' ? h.seg_offset_first : 0}">
            ${esc(h.author || 'Anon.')} — ${esc(h.title)}</a>
          · ${where}
          ${h.match ? ` · <span title="how it matched">${esc(h.match)}</span>` : ''}</div>
        <p style="margin:0.35rem 0">${body}</p>
        <div>${(h.topics || []).map(t => `<span class="chip" data-topic="${esc(t)}">${esc(t)}</span>`).join('')}</div>
      </div>`;
  }).join('');

  const modeNote = s.mode === 'hybrid' ? 'keyword + meaning'
    : s.mode === 'keyword' ? 'keyword only' : s.mode === 'recent' ? 'most recently summarized' : '';
  return `
    <h2>Summaries</h2>
    <p class="muted">A local LLM (Ollama, on whichever GPU isn't translating) reads each translated
      document — original and translation side by side — and writes a summary of the whole work
      plus one per part. Exact words and names match by keyword; concepts (“lending at
      interest” finding usury) match by meaning.</p>
    <div class="row">
      <input type="search" data-summ-q value="${esc(s.q)}" placeholder="saints, feasts, places, subjects…" style="min-width:340px">
      <select data-summ-level>
        <option value="" ${s.level === '' ? 'selected' : ''}>works and parts</option>
        <option value="document" ${s.level === 'document' ? 'selected' : ''}>whole works only</option>
        <option value="part" ${s.level === 'part' ? 'selected' : ''}>parts only</option>
      </select>
      <button class="btn" data-summ-go ${s.loading ? 'disabled' : ''}>${s.loading ? 'searching…' : 'Search'}</button>
      <span class="muted">${num(st.docs)} works summarized · ${num(st.parts)} parts</span>
      <button class="btn ghost" data-summarize-all>Summarize everything translated</button>
    </div>
    ${modeNote ? `<p class="muted">${esc(modeNote)}${s.note ? ' — ' + esc(s.note) : ''}</p>` : ''}
    ${items || `<p class="muted">${s.q ? 'No matches.' : 'Nothing summarized yet — open a translated document and press Summarize, or summarize everything translated.'}</p>`}`;
}

async function runSummarySearch() {
  state.summ.loading = true; render();
  try {
    const data = await api('/api/summaries/search?' + new URLSearchParams(
      { q: state.summ.q, level: state.summ.level, limit: 30 }));
    Object.assign(state.summ, { items: data.items, mode: data.mode, note: data.note || '',
                                stats: data.stats, hl: data.hl });
  } catch (e) { toast(e.message, true); }
  state.summ.loading = false; render();
}

// ---------------------------------------------------------------------------
// view: search
// ---------------------------------------------------------------------------

function viewSearch() {
  const s = state.search;
  const hits = s.items.map(h => `
    <div class="card">
      <div class="muted" style="font-size:0.78rem">
        <a data-open="${h.doc_id}">${esc(h.author || 'Anon.')} — ${esc(h.title)}</a>
        ${h.source_loc ? ' · ' + esc(h.source_loc) : ''} · ${h.score.toFixed(3)}</div>
      <div style="font-family:var(--serif);margin:0.3rem 0">${esc(h.latin)}</div>
      <div>${h.english ? esc(h.english) : '<span class="untranslated">— not yet translated —</span>'}</div>
    </div>`).join('');

  return `
    <h2>Search</h2>
    <p class="muted">Cross-lingual semantic search over every segment. The embedder and the
       ~600 MB index load on the first query, so the first search takes a minute.</p>
    <div class="row">
      <input type="search" data-search-q value="${esc(s.q)}" placeholder="English or Latin…" style="min-width:340px">
      <button class="btn" data-search-go>Search</button>
    </div>
    ${s.message ? `<p class="muted">${esc(s.message)}</p>` : ''}
    ${hits}`;
}

async function runSearch() {
  state.search.message = 'Searching…'; render();
  try {
    const data = await api('/api/search?' + new URLSearchParams({ q: state.search.q, k: 12 }));
    state.search.items = data.items;
    state.search.message = data.state === 'ready'
      ? (data.items.length ? '' : 'No results.')
      : (data.message || data.error || '');
  } catch (e) {
    state.search.message = e.message;
  }
  render();
}

// ---------------------------------------------------------------------------
// render + events
// ---------------------------------------------------------------------------

function render() {
  const views = {
    library: viewLibrary, reader: viewReader, files: viewFiles,
    catalog: viewCatalog, jobs: viewJobs, search: viewSearch, summaries: viewSummaries,
  };
  main.innerHTML = views[state.view]();
  document.querySelectorAll('nav.side button[data-view]').forEach(b =>
    b.classList.toggle('active', b.dataset.view === state.view));

  const t = state.facets?.totals;
  if (t) {
    $('#totals').innerHTML = `
      <b>${num(t.documents)}</b> works<br>
      <b>${num(t.segments)}</b> segments<br>
      <b>${num(t.translated)}</b> translated<br>
      <b>${num(t.untranslated)}</b> pending<br>
      <b>${num(t.styled)}</b> stylized`;
  }
}

async function go(view) {
  state.view = view;
  if (view === 'library') await loadDocs();
  if (view === 'files') await loadFiles(state.files.root, state.files.path);
  if (view === 'jobs') {
    await refreshJobs();
    if (state.llmModels === null) {
      state.llmModels = [];
      api('/api/llm/models').then(r => {
        state.llmModels = r.models || [];
        if (state.view === 'jobs' && !typing()) render();
      }).catch(() => {});
    }
  }
  if (view === 'summaries' && !state.summ.items.length) { render(); return runSummarySearch(); }
  render();
}

document.addEventListener('click', async (ev) => {
  const el = ev.target.closest('[data-view],[data-open],[data-page],[data-cd],[data-peek],' +
    '[data-root],[data-theme],[data-cat-go],[data-ingest],[data-ingest-all],[data-joblog],' +
    '[data-joblog-close],[data-cancel],[data-requeue],[data-jobs-refresh],[data-jobs-clear],' +
    '[data-queue-doc],[data-queue-selected],[data-queue-filter],[data-select-page],' +
    '[data-select-none],[data-translate-doc],[data-translate-range],[data-stylize-doc],' +
    '[data-more],[data-search-go],[data-summarize-doc],[data-summarize-preview],[data-summarize-selected],' +
    '[data-summarize-all],[data-summ-go],[data-topic],[data-jump],[data-open-at],' +
    '[data-regerman-doc],[data-regerman-selected]');
  if (!el) return;
  const d = el.dataset;

  try {
    if (d.view) return go(d.view);
    if (d.open) return openDoc(parseInt(d.open, 10));

    if (d.page) {
      state.page.offset = Math.max(0, state.page.offset + parseInt(d.page, 10) * state.page.limit);
      await loadDocs(); return render();
    }
    if (d.selectPage !== undefined) {
      state.docs.forEach(x => state.selected.add(x.id)); return render();
    }
    if (d.selectNone !== undefined) { state.selected.clear(); return render(); }

    if (d.queueDoc) return queueJob('translate', { doc_id: parseInt(d.queueDoc, 10), ...jobOpts('translate') });
    if (d.queueSelected) {
      if (!state.selected.size) return toast('Nothing selected.', true);
      return queueBulk(d.queueSelected, [...state.selected], jobOpts(d.queueSelected));
    }
    if (d.queueFilter) return queueWholeFilter(d.queueFilter);

    if (d.translateDoc !== undefined)
      return queueJob('translate', { doc_id: state.doc.id, ...jobOpts('translate') });
    if (d.regermanDoc !== undefined)
      return queueJob('translate', { doc_id: state.doc.id, ...jobOpts('translate'),
                                     retranslate_german: true });
    if (d.regermanSelected !== undefined) {
      if (!state.selected.size) return toast('Nothing selected.', true);
      return queueBulk('translate', [...state.selected],
                       { ...jobOpts('translate'), retranslate_german: true });
    }
    if (d.stylizeDoc !== undefined)
      return queueJob('stylize', { doc_id: state.doc.id, ...jobOpts('stylize') });
    if (d.translateRange !== undefined) {
      const first = parseInt($('[data-sec-first]')?.value || '1', 10);
      const last = parseInt($('[data-sec-last]')?.value || '1', 10);
      return queueJob('translate', {
        doc_id: state.doc.id, section_first: first, section_last: last,
        ...jobOpts('translate'),
      });
    }
    if (d.more !== undefined) return loadMoreSegments();

    if (d.cd !== undefined) { await loadFiles(state.files.root, d.cd); state.files.preview = null; return render(); }
    if (d.root) { await loadFiles(d.root, ''); state.files.preview = null; return render(); }
    if (d.peek) {
      state.files.preview = await api(
        `/api/files/preview?root=${encodeURIComponent(state.files.root)}&path=${encodeURIComponent(d.peek)}`);
      state.files.preview.path = d.peek;
      return render();
    }

    if (d.theme) { state.catalog.query = d.theme; render(); return runCatalogSearch(); }
    if (d.catGo !== undefined) return runCatalogSearch();
    if (d.ingest) return queueIngest([state.catalog.items[parseInt(d.ingest, 10)]]);
    if (d.ingestAll !== undefined) return queueIngest(state.catalog.items.filter(r => r.fetchable));

    if (d.joblog) {
      const data = await api(`/api/jobs/${d.joblog}/log`);
      state.jobLog = { id: parseInt(d.joblog, 10), text: data.log };
      render();
      const pre = $('#jobs-log pre.log'); if (pre) pre.scrollTop = pre.scrollHeight;
      return;
    }
    if (d.joblogClose !== undefined) { state.jobLog = null; return render(); }
    if (d.cancel) { await api(`/api/jobs/${d.cancel}/cancel`, { method: 'POST' }); return refreshJobs(); }
    if (d.requeue) { await api(`/api/jobs/${d.requeue}/requeue`, { method: 'POST' }); return refreshJobs(); }
    if (d.jobsRefresh !== undefined) return refreshJobs();
    if (d.jobsClear !== undefined) { await api('/api/jobs/clear', { method: 'POST' }); return refreshJobs(); }
    if (d.searchGo !== undefined) {
      state.search.q = $('[data-search-q]').value; return runSearch();
    }

    if (d.summarizePreview !== undefined)
      return queueJob('summarize', { doc_ids: [state.doc.id], ...jobOpts('summarize'), source: 'original', force: true });
    if (d.summarizeDoc !== undefined)
      return queueJob('summarize', { doc_ids: [state.doc.id], ...jobOpts('summarize'), force: true });
    if (d.summarizeSelected !== undefined) {
      if (!state.selected.size) return toast('Nothing selected.', true);
      // One job for the lot: each job starts (and stops) its own model server,
      // so per-document jobs would pay that load time again for every one.
      return queueJob('summarize', { doc_ids: [...state.selected], ...jobOpts('summarize') });
    }
    if (d.summarizeAll !== undefined) {
      if (!confirm('Summarize every document that is at least half translated and not yet ' +
                   'summarized? On a large library this is many hours of GPU time (it runs ' +
                   'beside translations, on the other card).')) return;
      return queueJob('summarize', { all: true, ...jobOpts('summarize') });
    }
    if (d.summGo !== undefined) { state.summ.q = $('[data-summ-q]').value; return runSummarySearch(); }
    if (d.topic) { state.summ.q = d.topic; state.view = 'summaries'; return runSummarySearch(); }
    if (d.jump !== undefined) return openDoc(state.doc.id, '', parseInt(d.jump, 10));
    if (d.openAt) return openDoc(parseInt(d.openAt, 10), '', parseInt(d.offset || '0', 10));
  } catch (e) { toast(e.message, true); }
});

document.addEventListener('change', async (ev) => {
  const el = ev.target, d = el.dataset;
  if (d.filter) {
    state.filters[d.filter] = el.value;
    state.page.offset = 0;
    await loadDocs(); return render();
  }
  if (d.select) {
    const id = parseInt(d.select, 10);
    el.checked ? state.selected.add(id) : state.selected.delete(id);
    return;
  }
  if (d.pageSize !== undefined) {
    state.page.limit = parseInt(el.value, 10); state.page.offset = 0;
    await loadDocs(); return render();
  }
  if (d.readerView !== undefined) { state.readerView = el.value; return render(); }
  if (d.readerSection !== undefined) return openDoc(state.doc.id, el.value);
  if (d.catSource !== undefined) {
    state.catalog.source = el.value;
    state.catalog.items = [];
    // Start each source on its most useful query rather than the last one's.
    state.catalog.query = (THEMES[el.value] || [''])[0];
    return render();
  }
  if (d.summLevel !== undefined) { state.summ.level = el.value; return runSummarySearch(); }
  if (d.opt) { state.opts[d.opt] = el.value; return; }
});

document.addEventListener('input', (ev) => {
  const d = ev.target.dataset;
  if (d.catQuery !== undefined) state.catalog.query = ev.target.value;
  if (d.searchQ !== undefined) state.search.q = ev.target.value;
  if (d.summQ !== undefined) state.summ.q = ev.target.value;
});

// Enter submits the two search boxes.
document.addEventListener('keydown', (ev) => {
  if (ev.key !== 'Enter') return;
  const d = ev.target.dataset || {};
  if (d.catQuery !== undefined) runCatalogSearch();
  if (d.searchQ !== undefined) { state.search.q = ev.target.value; runSearch(); }
  if (d.summQ !== undefined) { state.summ.q = ev.target.value; runSummarySearch(); }
});

function jobOpts(kind) {
  const o = state.opts;
  const gpu = o.gpu && o.gpu !== 'any' ? o.gpu : undefined;
  if (kind === 'summarize') {
    return { model: o.summary_model || undefined, source: o.summary_source, gpu };
  }
  if (kind === 'stylize') {
    return { preset: o.preset, backend: o.backend, batch_size: 20, gpu };
  }
  return { batch_size: Number(o.batch_size), chunk: Number(o.chunk),
           max_length: Number(o.max_length), gpu };
}

/* "Translate everything matching" -- one job when translate_pending.py can
 * express the filter itself (it understands language and source prefix), one
 * job per document otherwise, since a filter it cannot express has to be
 * enumerated here. */
async function queueWholeFilter(kind) {
  const f = state.filters;
  const scriptCanFilter = !f.q && !f.stage && !f.status;
  const noFilterAtAll = scriptCanFilter && !f.language && !f.source_prefix;
  if (noFilterAtAll) {
    const pending = state.facets?.totals?.untranslated;
    if (!confirm(`This queues one job over the entire library — about ` +
                 `${num(pending)} pending segments, which is days of GPU time. ` +
                 `Queue it?`)) return;
  }
  if (scriptCanFilter) {
    return queueJob(kind, {
      language: f.language || undefined,
      source_prefix: f.source_prefix || undefined,
      ...jobOpts(kind),
    });
  }
  if (state.page.total > 400 &&
      !confirm(`That filter needs one job per document — ${state.page.total} of them. Queue anyway?`)) return;
  const params = new URLSearchParams({ ...f, offset: 0, limit: 500 });
  const data = await api('/api/documents?' + params);
  return queueBulk(kind, data.items.map(x => x.id), jobOpts(kind));
}

async function queueIngest(records) {
  const usable = records.filter(r => r && r.fetchable);
  if (!usable.length) return toast('Nothing fetchable to queue.', true);
  for (const r of usable) {
    await queueJob('ingest', {
      source: state.catalog.source,
      identifier: r.identifier,
      genre: r.genre || undefined,
      language: r.lang_code || undefined,
    }, `Ingest: ${(r.title || r.identifier).slice(0, 60)}`);
  }
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

(async function boot() {
  try {
    state.facets = await api('/api/facets');
  } catch (e) { toast('Could not reach the API: ' + e.message, true); }
  await go('library');
  refreshJobs();
  // Poll the queue: a running job's progress is the one thing on this page that
  // changes without the user doing anything.
  setInterval(refreshJobs, 4000);
})();
