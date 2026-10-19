/**
 * Lightweight retrieval over the FAQ articles.
 *
 * We implement a simplified BM25 because:
 *   - no native embedding vectors in a Netlify Function cold start budget,
 *   - FAQs are small (tens to low hundreds of articles),
 *   - keyword overlap is already very strong for this kind of domain.
 *
 * Multi-tenant safe: the BM25 index is cached in a Map keyed by an
 * explicit `cacheKey` (typically the FAQ URL). Without this key, two
 * different tenants whose FAQs happen to have the same number of
 * articles would silently share an index, which would corrupt the
 * scoring for the second tenant. Always pass a stable `cacheKey`.
 *
 * Steps:
 *   1. Normalize (lowercase + strip accents + collapse punctuation).
 *   2. Tokenize, drop French stopwords.
 *   3. Score each article with BM25 (title weighted x3 vs. body).
 *   4. Return top-K above a minimum score, else top-K regardless
 *      (we always give the LLM *something*, and the prompt forbids
 *      it from answering if none fits, that's the safety net).
 */

const STOPWORDS = new Set([
  "a","à","ai","aie","aient","aies","ainsi","ait","alors","au","aucun","aucune",
  "aussi","autre","autres","aux","avec","avoir","bien","ça","ce","ceci","cela",
  "celle","celles","celui","ces","cet","cette","ceux","chez","comme","comment",
  "d","dans","de","des","du","elle","elles","en","encore","entre","es","est",
  "et","étaient","était","été","être","étés","étiez","étions","eu","être","faire",
  "faisait","fait","font","il","ils","je","j","la","là","le","les","leur","leurs",
  "lui","ma","mais","me","mes","mien","mon","n","ne","ni","non","nos","notre",
  "nous","on","ou","où","par","pas","peu","peux","plus","pour","pourquoi","puis",
  "qu","que","quel","quelle","quels","qui","quoi","sa","sans","se","ses","si",
  "sinon","soi","son","sont","sous","soyez","suis","sur","ta","te","tes","ton",
  "tous","tout","toute","toutes","tu","un","une","vers","vos","votre","vous",
  "y","c","l","s","m","d","t",
  // EN common
  "the","a","an","is","are","was","were","and","or","of","to","in","for","on","at","by","with","as","this","that","it","be","from",
]);

function foldAccents(s) {
  return s.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

function tokenize(text) {
  return foldAccents((text || "").toLowerCase())
    .replace(/[^a-z0-9àâäéèêëîïôöùûüÿç'\s-]/g, " ")
    .split(/[\s'-]+/)
    .map((t) => t.trim())
    .filter((t) => t.length >= 2 && !STOPWORDS.has(t))
    .map(stem);
}

// Racinisation légère : "délais" et "délai", "données" et "donnée",
// "tableaux" et "tableau" deviennent le même mot.
function stem(t) {
  if (t.length > 4 && /aux$/.test(t)) return t.slice(0, -1);   // tableaux -> tableau
  if (t.length > 3 && /[sx]$/.test(t)) return t.slice(0, -1);  // delais -> delai
  return t;
}

// BM25 params
const K1 = 1.5;
const B = 0.75;
const TITLE_BOOST = 3.0;

// Per-tenant index cache. Keys = stable cacheKey (typically FAQ URL).
// LRU-ish eviction when we exceed CACHE_MAX_ENTRIES (drop oldest insert).
const INDEX_CACHE = new Map();
const CACHE_MAX_ENTRIES = 50;

function buildIndex(faq) {
  // Precompute per-doc term frequencies and doc length
  const docs = faq.map((a) => {
    // Le thème compte comme le titre : "Délais", "Tarif", "Sécurité"...
    const titleTokens = tokenize(a.title + " " + (a.category || ""));
    const bodyTokens = tokenize(a.content);
    const tf = new Map();
    for (const t of titleTokens)
      tf.set(t, (tf.get(t) || 0) + TITLE_BOOST);
    for (const t of bodyTokens) tf.set(t, (tf.get(t) || 0) + 1);
    return { tf, length: titleTokens.length * TITLE_BOOST + bodyTokens.length };
  });

  // Document frequency per term
  const df = new Map();
  for (const d of docs) {
    for (const t of d.tf.keys()) {
      df.set(t, (df.get(t) || 0) + 1);
    }
  }

  const N = docs.length;
  const avgdl = docs.reduce((s, d) => s + d.length, 0) / Math.max(N, 1);

  // IDF per term
  const idf = new Map();
  for (const [t, n] of df.entries()) {
    idf.set(t, Math.log(1 + (N - n + 0.5) / (n + 0.5)));
  }

  return { docs, idf, avgdl, faqLength: faq.length };
}

function score(queryTokens, doc, idf, avgdl) {
  let s = 0;
  for (const q of queryTokens) {
    const f = doc.tf.get(q);
    if (!f) continue;
    const iq = idf.get(q) || 0;
    const numerator = f * (K1 + 1);
    const denominator = f + K1 * (1 - B + (B * doc.length) / avgdl);
    s += iq * (numerator / denominator);
  }
  return s;
}

function getOrBuildIndex(faq, cacheKey) {
  // If no cacheKey is provided, fall back to a content-derived signature.
  // This keeps the function safe even if the caller forgets to pass one,
  // but callers SHOULD pass an explicit key (FAQ URL is the canonical choice).
  const key = cacheKey
    ? `k:${cacheKey}`
    : `c:${faq.length}:${faq.slice(0, 3).map(a => (a.title || "").slice(0, 40)).join("|")}`;

  // Signature du contenu : si une ligne de la FAQ change sans que le nombre
  // d'entrées change (Google Sheet édité), l'index est reconstruit.
  const sig = faq.length + ":" + faq.reduce(
    (h, a) => (h * 31 + (a.title || "").length * 7 + (a.content || "").length) % 1000000007, 7
  );

  let entry = INDEX_CACHE.get(key);
  if (entry && entry.sig === sig) {
    return entry;
  }

  entry = buildIndex(faq);
  entry.sig = sig;
  INDEX_CACHE.set(key, entry);

  // Simple LRU-ish eviction (drop oldest insert)
  if (INDEX_CACHE.size > CACHE_MAX_ENTRIES) {
    const oldestKey = INDEX_CACHE.keys().next().value;
    if (oldestKey) INDEX_CACHE.delete(oldestKey);
  }

  return entry;
}

function retrieve(query, faq, k = 5, cacheKey = null) {
  if (!Array.isArray(faq) || faq.length === 0) return [];

  const { docs, idf, avgdl } = getOrBuildIndex(faq, cacheKey);

  const qTokens = tokenize(query);
  if (qTokens.length === 0) {
    // Fallback: return first K
    return faq.slice(0, k);
  }

  const scores = docs.map((d, i) => ({ i, s: score(qTokens, d, idf, avgdl) }));
  scores.sort((a, b) => b.s - a.s);

  // If absolutely nothing scored, still return top-K so model can at least
  // say "I don't know" with some grounding.
  return scores.slice(0, k).map((x) => faq[x.i]);
}

// Variante qui renvoie aussi le score (utilisée par la réponse de secours).
function bestMatches(query, faq, k = 3, cacheKey = null) {
  if (!Array.isArray(faq) || faq.length === 0) return [];
  const { docs, idf, avgdl } = getOrBuildIndex(faq, cacheKey);
  const qTokens = tokenize(query);
  if (qTokens.length === 0) return [];
  return docs
    .map((d, i) => ({ article: faq[i], score: score(qTokens, d, idf, avgdl) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

module.exports = { retrieve, bestMatches };
