/* =========================================================================
   ThermoData Chatbot - Netlify Function (Gemini multi-model + retries)
   - Reçoit POST { messages: [...] } depuis les domaines autorisés
   - Appelle l'API Gemini (clé jamais exposée côté client)
   - CORS configuré pour autoriser thermodata.fr
   - Tente plusieurs modèles Gemini en cascade
   - Retry automatique sur erreurs temporaires
   - Si tous les modèles échouent => renvoie une erreur exploitable côté front
   ========================================================================= */

// Domaines autorisés à appeler la function
const ALLOWED_ORIGINS = [
  'https://thermodata.fr',
  'https://www.thermodata.fr',
  'http://localhost:3000',
  'http://localhost:8888',
  'http://127.0.0.1:5500'
];

/*
  Ordre conseillé :
  1) modèles les plus économiques / légers
  2) modèles plus costauds ensuite

  Remarque :
  - Tous ne sont pas garantis gratuits en permanence
  - Certains previews peuvent changer ou disparaître
  - Le code continue même si un modèle n'est pas dispo
*/
const MODELS = [
  'gemini-3.1-flash-lite-preview',
  'gemini-2.5-flash-lite',
  'gemini-2.5-flash'
];

const MAX_HISTORY_MESSAGES = 20;
const MAX_MESSAGE_CHARS = 2000;
const MAX_OUTPUT_TOKENS = 400;
const TEMPERATURE = 0.4;

const REQUEST_TIMEOUT_MS = 15000;
const MAX_RETRIES_PER_MODEL = 2;
const RETRY_BASE_DELAY_MS = 700;

const SYSTEM_PROMPT = `Tu es l'assistant officiel de ThermoData (https://thermodata.fr), une plateforme française qui fournit des plans de prospection porte-à-porte aux artisans RGE (chauffagistes, isolateurs, énergéticiens) à partir des données DPE de l'ADEME.

OFFRES :
- Découverte : 49€ pour 50 adresses
- Pro : 99€ pour 200 adresses (le plus populaire)
- Accès à vie : 499€ illimité, 96 départements, mises à jour 12 mois

PRODUIT :
- Adresses certifiées BAN (Base Adresse Nationale)
- Score commercial 0-100 par prospect
- Tournée GPS optimisée (algo 2-Opt sur l'offre Pro)
- Carte interactive
- 6 onglets Excel (vue terrain, données brutes, analyse, etc.)
- Filtres : maison/appartement, classes DPE F et G, rayon personnalisable

DONNÉES :
- Source : DPE officiels ADEME
- Conformité RGPD : adresses publiques, base légale = intérêt légitime
- Couverture : 96 départements France métropolitaine

RÈGLES :
- Réponds en français, ton professionnel mais chaleureux
- Maximum 3-4 phrases courtes
- Si tu ne sais pas, ou question commerciale spécifique (devis, partenariat, remboursement, délai) -> redirige vers contact@thermodata.fr
- Ne promets jamais de délais, remboursements ou partenariats sans validation
- N'invente jamais de fonctionnalités ou de chiffres
- Si la question est hors-sujet (politique, perso, etc.), recadre poliment vers ThermoData`;

exports.handler = async (event) => {
  const origin = event.headers.origin || event.headers.Origin || '';
  const allowOrigin = ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0];

  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 204,
      headers: corsHeaders(allowOrigin),
      body: ''
    };
  }

  if (event.httpMethod !== 'POST') {
    return reply(405, { error: 'method_not_allowed' }, allowOrigin);
  }

  const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    console.error('Missing GEMINI_API_KEY / GOOGLE_API_KEY');
    return reply(503, { error: 'config_missing' }, allowOrigin);
  }

  let userMessages;
  try {
    const body = JSON.parse(event.body || '{}');
    userMessages = body.messages;

    if (!Array.isArray(userMessages) || userMessages.length === 0) {
      return reply(400, { error: 'messages_required' }, allowOrigin);
    }

    if (userMessages.length > MAX_HISTORY_MESSAGES) {
      userMessages = userMessages.slice(-MAX_HISTORY_MESSAGES);
    }

    userMessages = userMessages.map((m) => ({
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: String(m.content || '').slice(0, MAX_MESSAGE_CHARS)
    }));
  } catch (e) {
    return reply(400, { error: 'invalid_json' }, allowOrigin);
  }

  const contents = toGeminiContents(userMessages);

  const errors = [];

  for (const model of MODELS) {
    for (let attempt = 0; attempt <= MAX_RETRIES_PER_MODEL; attempt++) {
      try {
        const result = await callGemini({
          apiKey,
          model,
          contents,
          systemPrompt: SYSTEM_PROMPT
        });

        if (result.ok && result.text) {
          console.log(`Gemini success with model=${model} attempt=${attempt + 1}`);
          return reply(
            200,
            {
              reply: result.text,
              model
            },
            allowOrigin
          );
        }

        errors.push({
          model,
          attempt: attempt + 1,
          status: result.status,
          error: result.error
        });

        console.warn(
          `Gemini failed model=${model} attempt=${attempt + 1} status=${result.status} error=${result.error}`
        );

        if (!shouldRetry(result.status)) {
          break;
        }

        if (attempt < MAX_RETRIES_PER_MODEL) {
          await sleep(backoffDelay(attempt));
        }
      } catch (err) {
        const message = err?.message || 'unknown_error';

        errors.push({
          model,
          attempt: attempt + 1,
          status: 0,
          error: message
        });

        console.warn(
          `Gemini exception model=${model} attempt=${attempt + 1} error=${message}`
        );

        if (attempt < MAX_RETRIES_PER_MODEL) {
          await sleep(backoffDelay(attempt));
          continue;
        }
      }
    }
  }

  console.error('All Gemini models failed', JSON.stringify(errors));

  const authOrQuotaFailure = errors.some((e) =>
    [400, 401, 403, 429].includes(e.status)
  );

  return reply(
    authOrQuotaFailure ? 429 : 503,
    {
      error: authOrQuotaFailure ? 'quota_or_auth' : 'unavailable',
      details: errors.slice(-10)
    },
    allowOrigin
  );
};

async function callGemini({ apiKey, model, contents, systemPrompt }) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(apiKey)}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        signal: controller.signal,
        body: JSON.stringify({
          systemInstruction: {
            parts: [{ text: systemPrompt }]
          },
          contents,
          generationConfig: {
            temperature: TEMPERATURE,
            maxOutputTokens: MAX_OUTPUT_TOKENS
          }
        })
      }
    );

    const rawText = await res.text();
    let data = null;

    try {
      data = rawText ? JSON.parse(rawText) : null;
    } catch (e) {
      data = null;
    }

    if (!res.ok) {
      return {
        ok: false,
        status: res.status,
        error: extractApiError(data) || rawText || 'api_error'
      };
    }

    const text = extractCandidateText(data);

    if (!text) {
      return {
        ok: false,
        status: 503,
        error: 'empty_response'
      };
    }

    return {
      ok: true,
      status: 200,
      text
    };
  } catch (err) {
    if (err.name === 'AbortError') {
      return {
        ok: false,
        status: 408,
        error: 'timeout'
      };
    }

    return {
      ok: false,
      status: 0,
      error: err.message || 'network_error'
    };
  } finally {
    clearTimeout(timeout);
  }
}

function toGeminiContents(messages) {
  return messages.map((m) => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: m.content }]
  }));
}

function extractCandidateText(data) {
  const parts = data?.candidates?.[0]?.content?.parts;
  if (!Array.isArray(parts)) return '';

  return parts
    .map((p) => (typeof p?.text === 'string' ? p.text : ''))
    .join('')
    .trim();
}

function extractApiError(data) {
  return (
    data?.error?.message ||
    data?.error?.status ||
    data?.error?.code ||
    ''
  );
}

function shouldRetry(status) {
  return [0, 408, 429, 500, 502, 503, 504].includes(status);
}

function backoffDelay(attempt) {
  return RETRY_BASE_DELAY_MS * Math.pow(2, attempt);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function reply(statusCode, body, origin) {
  return {
    statusCode,
    headers: corsHeaders(origin),
    body: JSON.stringify(body)
  };
}

function corsHeaders(origin) {
  return {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Vary': 'Origin'
  };
}
