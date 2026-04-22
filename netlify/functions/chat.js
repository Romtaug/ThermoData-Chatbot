/* =========================================================================
   ThermoData Chatbot - Netlify Function (OpenAI)
   - Reçoit POST { messages: [...] } depuis les domaines autorisés
   - Appelle l'API OpenAI (clé jamais exposée côté client)
   - CORS configuré pour autoriser thermodata.fr
   - Si quota / crédit épuisé => renvoie le code HTTP au front pour fallback email
   ========================================================================= */

// ⚠️ EDIT THIS : domaines autorisés à appeler la function
const ALLOWED_ORIGINS = [
  'https://thermodata.fr',
  'https://www.thermodata.fr',
  'http://localhost:3000',
  'http://localhost:8888',
  'http://127.0.0.1:5500'
];

// ⚠️ EDIT THIS : modèle OpenAI à utiliser
// - 'gpt-4o-mini'  : rapide + très économique - recommandé pour un chat support
// - 'gpt-4o'       : meilleure qualité (~10x plus cher)
// - 'gpt-4.1-mini' : variante récente, cost/quality similaire à gpt-4o-mini
const MODEL = 'gpt-4o-mini';

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
- Si tu ne sais pas, ou question commerciale spécifique (devis, partenariat, remboursement, délai) → redirige vers contact@thermodata.fr
- Ne promets jamais de délais, remboursements ou partenariats sans validation
- N'invente jamais de fonctionnalités ou de chiffres
- Si la question est hors-sujet (politique, perso, etc.), recadre poliment vers ThermoData`;

exports.handler = async (event) => {
  const origin = event.headers.origin || event.headers.Origin || '';
  const allowOrigin = ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0];

  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 204, headers: corsHeaders(allowOrigin), body: '' };
  }

  if (event.httpMethod !== 'POST') {
    return reply(405, { error: 'method_not_allowed' }, allowOrigin);
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.error('OPENAI_API_KEY not set');
    return reply(503, { error: 'config_missing' }, allowOrigin);
  }

  // Parse body
  let userMessages;
  try {
    const body = JSON.parse(event.body || '{}');
    userMessages = body.messages;
    if (!Array.isArray(userMessages) || userMessages.length === 0) {
      return reply(400, { error: 'messages_required' }, allowOrigin);
    }
    if (userMessages.length > 20) userMessages = userMessages.slice(-20);
    userMessages = userMessages.map(m => ({
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: String(m.content || '').slice(0, 2000)
    }));
  } catch (e) {
    return reply(400, { error: 'invalid_json' }, allowOrigin);
  }

  // Format OpenAI : system prompt en 1er message avec role "system"
  const messages = [
    { role: 'system', content: SYSTEM_PROMPT },
    ...userMessages
  ];

  try {
    const apiRes = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: MODEL,
        messages: messages,
        max_tokens: 400,
        temperature: 0.4
      })
    });

    // Codes "plus de crédit" / quota / auth => fallback email côté front
    if ([401, 402, 429].includes(apiRes.status)) {
      const errBody = await apiRes.text();
      console.warn('OpenAI credit/quota error:', apiRes.status, errBody);
      return reply(apiRes.status, { error: 'quota_or_auth' }, allowOrigin);
    }

    if (!apiRes.ok) {
      const errBody = await apiRes.text();
      console.error('OpenAI API error:', apiRes.status, errBody);
      return reply(503, { error: 'api_error' }, allowOrigin);
    }

    const data = await apiRes.json();
    const text = data.choices?.[0]?.message?.content || '';
    if (!text) {
      console.error('Empty response from OpenAI:', JSON.stringify(data));
      return reply(503, { error: 'empty_response' }, allowOrigin);
    }

    return reply(200, { reply: text }, allowOrigin);
  } catch (err) {
    console.error('Function error:', err.message);
    return reply(503, { error: 'unavailable' }, allowOrigin);
  }
};

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
