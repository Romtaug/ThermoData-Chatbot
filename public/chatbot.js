/* =========================================================================
   ThermoData Chatbot : widget (v3)
   Servi par thermodata-chatbot.netlify.app et embarqué sur thermodata.fr :
     <script src="https://thermodata-chatbot.netlify.app/chatbot.js" defer></script>
   (ligne inchangée sur le site : rien à modifier côté thermodata.fr)

   v3 :
     - email et liens cliquables, jamais de [x](mailto:x) affiché brut
     - relance cliquable dans la bulle + 2 suggestions de suivi
     - boutons « Voir les tarifs », « Voir la démo », « Nous contacter »
       qui amènent à la bonne section de la page
     - une panne ne bloque plus le chat : relance silencieuse + « Réessayer »
     - conversation conservée pendant la session, « nouvelle conversation »
     - accessibilité : aria-live, Échap pour fermer, pas de zoom iOS
   ========================================================================= */
(function () {
  'use strict';

  if (document.getElementById('tdc-chatbot') || document.getElementById('td-chatbot')) return; // double chargement

  // =========================================================================
  // CONFIGURATION
  // =========================================================================
  const SCRIPT_TAG = document.currentScript || (function () {
    const s = document.getElementsByTagName('script');
    for (let i = s.length - 1; i >= 0; i--) {
      if ((s[i].src || '').indexOf('widget.js') !== -1) return s[i];
    }
    return null;
  })();
  const D = SCRIPT_TAG ? SCRIPT_TAG.dataset : {};

  function darkenHex(hex, amount) {
    if (!/^#[0-9a-f]{6}$/i.test(hex || '')) return hex;
    const c = [1, 3, 5].map(i => Math.max(0, parseInt(hex.slice(i, i + 2), 16) - amount));
    return '#' + c.map(x => x.toString(16).padStart(2, '0')).join('');
  }
  function hexToRgba(hex, a) {
    const m = String(hex).replace('#', '').match(/^([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
    if (!m) return 'rgba(0,0,0,' + a + ')';
    return 'rgba(' + parseInt(m[1], 16) + ',' + parseInt(m[2], 16) + ',' + parseInt(m[3], 16) + ',' + a + ')';
  }

  // URL du service qui héberge ce widget et la fonction /chat (sans slash final).
  const SERVICE_BASE_URL = 'https://thermodata-chatbot.netlify.app';

  const MAIL_SUBJECT = 'Question depuis le chat ThermoData';
  const CONTACT_EMAIL = 'contact@thermodata.fr';
  const brand = '#E67E22';

  const CONFIG = {
    title: 'Assistant ThermoData',
    welcome: "Bonjour 👋 Je suis l'assistant ThermoData. Posez-moi vos questions sur nos offres, nos données DPE ou la prospection RGE !",
    placeholder: 'Votre message...',
    brand: brand,
    brandDark: '#D85A30',
    headerStart: '#0B1A2F',
    headerEnd: '#1E3A5F',
    maxTurns: 30,
    requestTimeout: 25000,
    storageKey: 'thermodata_chat_v3',
    teaser: '',
    teaserDelay: 12000,
    // Sections de thermodata.fr vers lesquelles mènent les boutons d'action
    sections: { pricing: '#pricing', demo: '#demo', contact: '#contact' }
  };
  const startQuestions = ['Quelles sont vos offres ?', "D'où viennent vos données ?", 'Comment ça marche ?'];
  const ENDPOINT = SERVICE_BASE_URL + '/.netlify/functions/chat';
  const PRIMARY = { type: 'mail', value: CONTACT_EMAIL, display: CONTACT_EMAIL };
  const TU = false;
  CONFIG.contacts = [PRIMARY];

  // =========================================================================
  // STYLES (préfixe tdc- : aucun conflit avec la page ni avec l'ancien widget td-)
  // =========================================================================
  const B = CONFIG.brand, BD = CONFIG.brandDark;
  const css = `
#tdc-chatbot, #tdc-chatbot * { box-sizing: border-box; font-family: 'Outfit', system-ui, -apple-system, 'Segoe UI', sans-serif; }
#tdc-toggle {
  position: fixed; bottom: 24px; right: 24px; width: 60px; height: 60px; border-radius: 50%;
  background: linear-gradient(135deg, ${B} 0%, ${BD} 100%); border: none; cursor: pointer; padding: 0;
  box-shadow: 0 8px 24px ${hexToRgba(B, 0.4)}; z-index: 99999;
  display: flex; align-items: center; justify-content: center; transition: transform .2s;
}
#tdc-toggle:hover { transform: scale(1.06); }
#tdc-toggle:focus-visible, #tdc-chatbot button:focus-visible, #tdc-chatbot a:focus-visible, #tdc-teaser:focus-visible { outline: 3px solid ${hexToRgba(B, 0.5)}; outline-offset: 2px; }
#tdc-toggle svg { width: 26px; height: 26px; fill: #fff; }
#tdc-toggle .tdc-ico-close { display: none; }
#tdc-toggle.open .tdc-ico-chat { display: none; }
#tdc-toggle.open .tdc-ico-close { display: block; }
#tdc-badge {
  position: absolute; top: -2px; right: -2px; background: #E4573D; color: #fff;
  font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 100px; border: 2px solid #fff;
}
#tdc-teaser {
  position: fixed; bottom: 96px; right: 24px; max-width: 270px; z-index: 99999;
  background: #fff; color: #1F2937; border: 1px solid #E5E7EB; border-radius: 16px 16px 4px 16px;
  box-shadow: 0 12px 32px rgba(0,0,0,.14); padding: 12px 34px 12px 14px;
  font-size: 13.5px; line-height: 1.45; cursor: pointer; display: none;
}
#tdc-teaser.show { display: block; animation: tdcUp .35s cubic-bezier(.16,1,.3,1); }
#tdc-teaser-close {
  position: absolute; top: 6px; right: 6px; width: 24px; height: 24px; border: none; background: none;
  color: #9CA3AF; cursor: pointer; font-size: 16px; line-height: 1; border-radius: 6px;
}
#tdc-teaser-close:hover { background: #F3F4F6; color: #4B5563; }
#tdc-panel {
  position: fixed; bottom: 96px; right: 24px; width: 390px; max-width: calc(100vw - 32px);
  height: 600px; max-height: calc(100vh - 130px); background: #fff; border-radius: 20px;
  box-shadow: 0 20px 60px rgba(0,0,0,.2); z-index: 99998; display: none; flex-direction: column;
  overflow: hidden; border: 1px solid rgba(0,0,0,.06);
}
#tdc-panel.open { display: flex; animation: tdcUp .3s cubic-bezier(.16,1,.3,1); }
@keyframes tdcUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
.tdc-header {
  background: linear-gradient(135deg, ${CONFIG.headerStart} 0%, ${CONFIG.headerEnd} 100%);
  padding: 14px 12px 14px 18px; color: #fff; display: flex; align-items: center; gap: 10px;
}
.tdc-avatar { width: 38px; height: 38px; border-radius: 50%; background: #fff; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }
.tdc-hinfo { flex: 1; min-width: 0; }
.tdc-hname { font-size: 14px; font-weight: 700; margin: 0; line-height: 1.2; color: #fff; }
.tdc-hstatus { font-size: 11px; opacity: .8; margin: 3px 0 0; display: flex; align-items: center; gap: 6px; color: #fff; }
.tdc-dot { width: 7px; height: 7px; border-radius: 50%; background: #22C55E; animation: tdcPulse 2s ease-out infinite; }
@keyframes tdcPulse { 0% { box-shadow: 0 0 0 0 rgba(34,197,94,.6); } 70% { box-shadow: 0 0 0 6px rgba(34,197,94,0); } 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); } }
.tdc-hbtn { background: none; border: none; color: #fff; cursor: pointer; padding: 6px; border-radius: 8px; opacity: .75; display: flex; }
.tdc-hbtn:hover { opacity: 1; background: rgba(255,255,255,.14); }
.tdc-messages { flex: 1; overflow-y: auto; padding: 18px; background: #F9FAFB; display: flex; flex-direction: column; gap: 10px; overscroll-behavior: contain; }
.tdc-msg { max-width: 86%; padding: 10px 14px; border-radius: 16px; font-size: 14px; line-height: 1.55; overflow-wrap: anywhere; }
.tdc-msg p { margin: 0; }
.tdc-msg p + p, .tdc-msg p + ul, .tdc-msg ul + p { margin-top: 8px; }
.tdc-msg ul { margin: 6px 0 0; padding-left: 18px; }
.tdc-msg li { margin: 2px 0; }
.tdc-msg strong { font-weight: 700; }
.tdc-bot { background: #fff; color: #1F2937; border: 1px solid #E5E7EB; align-self: flex-start; border-bottom-left-radius: 4px; }
.tdc-bot a { color: ${BD}; font-weight: 600; text-decoration: none; border-bottom: 1px solid ${hexToRgba(BD, 0.35)}; }
.tdc-msg a[href^="mailto:"] { white-space: nowrap; } /* l'adresse ne se coupe jamais en deux */
.tdc-error a { color: #991B1B; font-weight: 700; text-decoration: underline; }
.tdc-bot a:hover { color: ${B}; border-bottom-color: ${B}; }
.tdc-user { background: linear-gradient(135deg, ${B}, ${BD}); color: #fff; align-self: flex-end; border-bottom-right-radius: 4px; white-space: pre-wrap; }
.tdc-error { background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; align-self: stretch; max-width: 100%; font-size: 13px; }
.tdc-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.tdc-retry { background: #fff; border: 1px solid #FCA5A5; color: #991B1B; border-radius: 100px; padding: 5px 12px; font-size: 12px; font-weight: 600; cursor: pointer; }
.tdc-retry:hover { background: #FEE2E2; }
.tdc-relance {
  display: block; width: 100%; margin-top: 10px; padding: 8px 10px; text-align: left;
  background: ${hexToRgba(B, 0.07)}; border: 1px solid ${hexToRgba(B, 0.3)}; border-radius: 10px;
  color: ${BD}; font: inherit; font-size: 13.5px; font-weight: 600; line-height: 1.4; cursor: pointer; transition: background .15s, border-color .15s;
}
.tdc-relance:hover { background: ${hexToRgba(B, 0.14)}; border-color: ${B}; }
.tdc-relance::after { content: ' →'; }
.tdc-typing { display: flex; gap: 4px; padding: 12px 14px; background: #fff; border: 1px solid #E5E7EB; border-radius: 16px; border-bottom-left-radius: 4px; align-self: flex-start; }
.tdc-typing span { width: 7px; height: 7px; background: #9CA3AF; border-radius: 50%; animation: tdcBounce 1.6s ease-in-out infinite both; }
.tdc-typing span:nth-child(2) { animation-delay: .2s; }
.tdc-typing span:nth-child(3) { animation-delay: .4s; }
@keyframes tdcBounce { 0%, 80%, 100% { transform: scale(.8); opacity: .5; } 40% { transform: scale(1); opacity: 1; } }
.tdc-quick { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 18px 10px; background: #F9FAFB; }
.tdc-quick:empty { display: none; }
.tdc-chip { background: #fff; border: 1px solid #E5E7EB; color: #374151; padding: 6px 12px; border-radius: 100px; font-size: 12.5px; cursor: pointer; transition: all .15s; font-family: inherit; text-align: left; line-height: 1.3; }
.tdc-chip:hover { background: ${hexToRgba(B, 0.08)}; border-color: ${B}; color: ${BD}; }
.tdc-chip.tdc-cta { background: linear-gradient(135deg, ${B}, ${BD}); border-color: ${B}; color: #fff; font-weight: 600; }
.tdc-chip.tdc-cta:hover { filter: brightness(1.06); color: #fff; }
.tdc-input-area { padding: 12px 14px; background: #fff; border-top: 1px solid #E5E7EB; display: flex; gap: 8px; align-items: flex-end; }
.tdc-input { flex: 1; border: 1px solid #D1D5DB; border-radius: 12px; padding: 9px 12px; font-size: 14px; resize: none; max-height: 110px; min-height: 40px; outline: none; font-family: inherit; line-height: 1.4; color: #1F2937; background: #fff; }
.tdc-input:focus { border-color: ${B}; box-shadow: 0 0 0 3px ${hexToRgba(B, 0.14)}; }
.tdc-send { background: linear-gradient(135deg, ${B}, ${BD}); border: none; width: 40px; height: 40px; border-radius: 50%; cursor: pointer; flex-shrink: 0; display: flex; align-items: center; justify-content: center; transition: transform .2s; }
.tdc-send:hover:not(:disabled) { transform: scale(1.05); }
.tdc-send:disabled { opacity: .4; cursor: not-allowed; }
.tdc-send svg { width: 18px; height: 18px; fill: #fff; }
.tdc-footer { font-size: 11px; color: #9CA3AF; text-align: center; padding: 6px 10px 9px; background: #fff; }
.tdc-footer a, .tdc-footer button { color: #6B7280; text-decoration: none; background: none; border: none; padding: 0; font: inherit; cursor: pointer; }
.tdc-footer a:hover, .tdc-footer button:hover { text-decoration: underline; color: ${BD}; }
.tdc-sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
@media (max-width: 480px) {
  #tdc-panel { right: 8px; left: 8px; width: auto; max-width: none; bottom: 84px; height: calc(100vh - 100px); height: calc(100dvh - 100px); max-height: none; border-radius: 16px; }
  #tdc-toggle { right: 16px; bottom: 16px; }
  #tdc-teaser { right: 16px; bottom: 86px; max-width: calc(100vw - 32px); }
  .tdc-input { font-size: 16px; } /* évite le zoom automatique sur iOS */
}
@media (prefers-reduced-motion: reduce) {
  #tdc-panel.open, #tdc-teaser.show, .tdc-typing span, .tdc-dot { animation: none !important; }
  #tdc-toggle, .tdc-send, .tdc-chip { transition: none !important; }
}
`;
  const styleEl = document.createElement('style');
  styleEl.textContent = css;
  document.head.appendChild(styleEl);

  // =========================================================================
  // RENDU DU TEXTE
  // =========================================================================
  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, c => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[c]
    ));
  }

  function linkTag(href, label) {
    const ext = /^https?:/i.test(href);
    const anchor = href.charAt(0) === '#';
    return '<a href="' + href + '"' +
      (ext ? ' target="_blank" rel="noopener noreferrer"' : '') +
      (anchor ? ' data-tdc-anchor="1"' : '') + '>' + label + '</a>';
  }

  // Une seule passe sur le texte échappé : on ne recrée que nos propres balises.
  //   [texte](https://...)            -> lien externe
  //   [x@y.fr](mailto:x@y.fr)         -> lien mailto, sans crochets visibles
  //   [x@y.fr](x@y.fr) ou (#ancre)    -> lien propre
  //   <https://...> ou <x@y.fr>       -> chevrons retirés, lien propre
  //   **gras**, URL nues, emails nus  -> gras, lien, mailto
  // Jamais de crochets ni de parenthèses markdown affichés au visiteur.
  const EMAIL = '[\\w.+-]+@[\\w-]+(?:\\.[\\w-]+)+';
  const INLINE_RE = new RegExp(
    '\\[([^\\]\\n]+)\\]\\(\\s*(?:&lt;)?((?:[^\\s()]|\\([^\\s()]*\\))+?)(?:&gt;)?\\s*\\)' +   // 1-2 lien markdown
    '|&lt;((?:https?:\\/\\/|mailto:)[^\\s&]+|' + EMAIL + ')&gt;' +     // 3   <autolien>
    '|\\*\\*([^*\\n]+)\\*\\*' +                                        // 4   gras
    '|(https?:\\/\\/[^\\s<]*[^\\s<.,;:!?)\\]\'"&])' +                  // 5   URL nue
    '|(mailto:' + EMAIL + ')' +                                        // 6   mailto: nu
    '|(' + EMAIL + ')',                                                // 7   email nu
    'g'
  );

  function mailtoTag(target, label) {
    const addr = target.replace(/^mailto:/i, '').split('?')[0];
    const subject = target.indexOf('?') !== -1 ? '' : '?subject=' + encodeURIComponent(MAIL_SUBJECT);
    return linkTag('mailto:' + addr + subject, label || addr);
  }

  function hrefFor(target, label) {
    const emailOnly = new RegExp('^' + EMAIL + '$');
    if (/^mailto:/i.test(target) || emailOnly.test(target)) return mailtoTag(target, label);
    if (/^tel:/i.test(target)) return linkTag(target, label);
    if (/^(https?:\/\/|#)/i.test(target)) return linkTag(target, label);
    return label; // schéma inconnu (javascript:, data:...) : texte seul, sans crochets
  }

  function renderInline(escaped) {
    return escaped.replace(INLINE_RE, function (m, label, mdUrl, auto, bold, url, mailto, email) {
      if (label) {
        // [mailto:x@y](mailto:x@y) -> on affiche juste l'adresse
        const shown = label.replace(/^mailto:/i, '');
        return hrefFor(mdUrl, shown);
      }
      if (auto) return hrefFor(auto, auto.replace(/^mailto:/i, ''));
      if (bold) return '<strong>' + renderInline(bold) + '</strong>';
      if (url) return linkTag(url, url);
      if (mailto) return mailtoTag(mailto);
      if (email) return mailtoTag(email);
      return m;
    });
  }

  // Paragraphes + listes ("- ", "• ", "1. ").
  function renderBot(text) {
    const lines = escapeHtml(String(text || '').replace(/\r/g, '')).split('\n');
    let html = '', inList = false, para = [];
    function flushPara() {
      if (para.length) { html += '<p>' + para.map(renderInline).join('<br>') + '</p>'; para = []; }
    }
    lines.forEach(function (raw) {
      const line = raw.trim();
      const li = line.match(/^(?:[-•*]|\d+[.)])\s+(.*)$/);
      if (li) {
        flushPara();
        if (!inList) { html += '<ul>'; inList = true; }
        html += '<li>' + renderInline(li[1]) + '</li>';
      } else if (!line) {
        flushPara();
        if (inList) { html += '</ul>'; inList = false; }
      } else {
        if (inList) { html += '</ul>'; inList = false; }
        para.push(line);
      }
    });
    flushPara();
    if (inList) html += '</ul>';
    return html;
  }

  // ------------------ CONTACTS ------------------
  function contactHref(c) {
    if (c.type === 'tel') return 'tel:' + c.value;
    if (c.type === 'url') return c.value;
    return 'mailto:' + c.value + '?subject=' + encodeURIComponent(MAIL_SUBJECT);
  }
  function contactLink(c) {
    return linkTag(escapeHtml(contactHref(c)), escapeHtml(c.display));
  }
  function contactLabel(c) {
    if (c.type === 'tel') return TU ? 'Nous appeler →' : 'Nous appeler →';
    if (c.type === 'url') return 'Formulaire de contact →';
    return TU ? 'Nous écrire →' : 'Nous écrire →';
  }
  // "écrivez-nous à X, appelez-nous au Y ou contactez-nous via Z"
  function contactPhraseHtml() {
    const parts = CONFIG.contacts.map(function (c) {
      if (c.type === 'tel') return (TU ? 'appelle-nous au ' : 'appelez-nous au ') + contactLink(c);
      if (c.type === 'url') return (TU ? 'contacte-nous via ' : 'contactez-nous via ') + linkTag(escapeHtml(c.value), 'notre formulaire');
      return (TU ? 'écris-nous à ' : 'écrivez-nous à ') + contactLink(c);
    });
    if (parts.length === 1) return parts[0];
    return parts.slice(0, -1).join(', ') + ' ou ' + parts[parts.length - 1];
  }
  function openContact(c) {
    const href = contactHref(c || PRIMARY);
    if (/^https?:/i.test(href)) window.open(href, '_blank', 'noopener,noreferrer');
    else window.location.href = href;
  }

  // =========================================================================
  // DOM
  // =========================================================================
  const root = document.createElement('div');
  root.id = 'tdc-chatbot';
  root.innerHTML = `
    <div id="tdc-teaser" role="button" tabindex="0" aria-label="Ouvrir l'assistant">
      <span id="tdc-teaser-text"></span>
      <button type="button" id="tdc-teaser-close" aria-label="Masquer l'invitation">×</button>
    </div>
    <button id="tdc-toggle" type="button" aria-label="Ouvrir l'assistant" aria-expanded="false" aria-controls="tdc-panel">
      <svg class="tdc-ico-chat" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z"/></svg>
      <svg class="tdc-ico-close" viewBox="0 0 24 24" aria-hidden="true"><path d="M19 6.4L17.6 5 12 10.6 6.4 5 5 6.4 10.6 12 5 17.6 6.4 19 12 13.4 17.6 19 19 17.6 13.4 12z"/></svg>
      <span id="tdc-badge" aria-hidden="true">1</span>
    </button>
    <div id="tdc-panel" role="dialog" aria-label="${escapeHtml(CONFIG.title)}">
      <div class="tdc-header">
        <div class="tdc-avatar" aria-hidden="true">🤖</div>
        <div class="tdc-hinfo">
          <p class="tdc-hname">${escapeHtml(CONFIG.title)}</p>
          <p class="tdc-hstatus"><span class="tdc-dot"></span>En ligne · réponse immédiate</p>
        </div>
        <button class="tdc-hbtn" id="tdc-reset" type="button" aria-label="Nouvelle conversation" title="Nouvelle conversation">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M17.65 6.35A7.96 7.96 0 0 0 12 4a8 8 0 1 0 7.73 10h-2.08A6 6 0 1 1 12 6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>
        </button>
        <button class="tdc-hbtn" id="tdc-close" type="button" aria-label="Fermer l'assistant">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M19 6.4L17.6 5 12 10.6 6.4 5 5 6.4 10.6 12 5 17.6 6.4 19 12 13.4 17.6 19 19 17.6 13.4 12z"/></svg>
        </button>
      </div>
      <div class="tdc-messages" id="tdc-messages" aria-live="polite" aria-relevant="additions"></div>
      <div class="tdc-quick" id="tdc-quick"></div>
      <div class="tdc-input-area">
        <label for="tdc-input" class="tdc-sr">Votre question</label>
        <textarea class="tdc-input" id="tdc-input" placeholder="${escapeHtml(CONFIG.placeholder)}" rows="1" maxlength="1000"></textarea>
        <button class="tdc-send" id="tdc-send" type="button" aria-label="Envoyer">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg>
        </button>
      </div>
      <div class="tdc-footer">Assistant IA · ${linkTag(escapeHtml(contactHref(PRIMARY)), 'Nous écrire directement')}</div>
    </div>`;
  document.body.appendChild(root);

  const $ = id => document.getElementById(id);
  const toggle = $('tdc-toggle'), panel = $('tdc-panel'), messagesEl = $('tdc-messages');
  const quickEl = $('tdc-quick'), input = $('tdc-input'), sendBtn = $('tdc-send'), badge = $('tdc-badge');
  const teaser = $('tdc-teaser');

  // =========================================================================
  // ÉTAT (persisté pendant la session de navigation)
  // =========================================================================
  let state = { history: [], last: null };
  try {
    const saved = JSON.parse(sessionStorage.getItem(CONFIG.storageKey) || 'null');
    if (saved && Array.isArray(saved.history)) state = saved;
  } catch (e) { /* stockage indisponible */ }
  function save() {
    try {
      state.history = state.history.slice(-40);
      sessionStorage.setItem(CONFIG.storageKey, JSON.stringify(state));
    } catch (e) { /* ignore */ }
  }
  let isOpen = false, isLoading = false, rendered = false;

  // =========================================================================
  // RENDU DES BULLES
  // =========================================================================
  function isMobile() { return window.matchMedia('(max-width: 760px)').matches; }
  function scrollDown() { messagesEl.scrollTop = messagesEl.scrollHeight; }

  function bubble(role, text, relance) {
    const el = document.createElement('div');
    if (role === 'user') { el.className = 'tdc-msg tdc-user'; el.textContent = text; }
    else {
      el.className = 'tdc-msg tdc-bot';
      el.innerHTML = renderBot(text);
      if (relance && relance.question && relance.message) {
        const b = document.createElement('button');
        b.type = 'button'; b.className = 'tdc-relance'; b.textContent = relance.question;
        b.addEventListener('click', function () { ask(relance.message); });
        el.appendChild(b);
      }
    }
    messagesEl.appendChild(el);
    scrollDown();
    return el;
  }
  function clearRelances() { messagesEl.querySelectorAll('.tdc-relance').forEach(b => b.remove()); }

  function errorBubble(html, retryText) {
    const el = document.createElement('div');
    el.className = 'tdc-msg tdc-error';
    const p = document.createElement('p');
    p.innerHTML = html;
    el.appendChild(p);
    const actions = document.createElement('div');
    actions.className = 'tdc-actions';
    if (retryText) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'tdc-retry'; b.textContent = 'Réessayer';
      b.addEventListener('click', function () { el.remove(); ask(retryText, true); });
      actions.appendChild(b);
    }
    const c = document.createElement('button');
    c.type = 'button'; c.className = 'tdc-retry'; c.textContent = contactLabel(PRIMARY).replace(' →', '');
    c.addEventListener('click', function () { openContact(PRIMARY); });
    actions.appendChild(c);
    el.appendChild(actions);
    messagesEl.appendChild(el);
    scrollDown();
  }

  function typing(on) {
    let t = messagesEl.querySelector('.tdc-typing');
    if (on && !t) {
      t = document.createElement('div');
      t.className = 'tdc-typing';
      t.setAttribute('role', 'status');
      t.setAttribute('aria-label', "L'assistant écrit");
      t.innerHTML = '<span></span><span></span><span></span>';
      messagesEl.appendChild(t);
      scrollDown();
    } else if (!on && t) t.remove();
  }

  function renderQuick(list, cta) {
    quickEl.innerHTML = '';
    const done = state.history.filter(m => m.role === 'user').map(m => m.content.toLowerCase());
    (list || []).filter(q => done.indexOf(String(q).toLowerCase()) === -1).slice(0, 4).forEach(function (q) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'tdc-chip'; b.textContent = q;
      b.addEventListener('click', function () { ask(q); });
      quickEl.appendChild(b);
    });
    const CTA_LABELS = { pricing: 'Voir les tarifs →', demo: 'Voir la démo →', contact: 'Nous contacter →' };
    if (CTA_LABELS[cta]) {
      const a = document.createElement('button');
      a.type = 'button'; a.className = 'tdc-chip tdc-cta'; a.textContent = CTA_LABELS[cta];
      a.addEventListener('click', function () { goToSection(cta); });
      quickEl.appendChild(a);
    }
    scrollDown();
  }

  function renderAll() {
    messagesEl.innerHTML = '';
    bubble('bot', CONFIG.welcome);
    const lastIdx = state.history.length - 1;
    state.history.forEach((m, i) => bubble(m.role === 'user' ? 'user' : 'bot', m.content, i === lastIdx ? m.relance : null));
    if (state.last) renderQuick(state.last.suggestions, state.last.cta);
    else renderQuick(startQuestions);
    rendered = true;
  }

  // Amène à une section de thermodata.fr ; à défaut, ouvre l'email.
  function goToSection(key) {
    let target = null;
    try { target = document.querySelector(CONFIG.sections[key]); } catch (e) { target = null; }
    if (!target) { if (key === 'contact') openContact(PRIMARY); return; }
    if (isMobile()) closePanel();
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // Ancres internes de la page hôte : défilement doux, panneau fermé sur mobile.
  messagesEl.addEventListener('click', function (e) {
    const a = e.target.closest && e.target.closest('a[data-tdc-anchor]');
    if (!a) return;
    let target = null;
    try { target = document.querySelector(a.getAttribute('href')); } catch (err) { target = null; }
    if (!target) return;
    e.preventDefault();
    if (isMobile()) closePanel();
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  // =========================================================================
  // ENVOI
  // =========================================================================
  async function ask(text, isRetry) {
    text = String(text || '').trim();
    if (!text || isLoading) return;
    if (!isOpen) openPanel();

    if (!isRetry && state.history.filter(m => m.role === 'user').length >= CONFIG.maxTurns) {
      errorBubble('Nous avons beaucoup échangé ! Pour aller plus loin, ' + contactPhraseHtml() + '.');
      return;
    }

    quickEl.innerHTML = '';
    clearRelances();
    if (!isRetry) {
      bubble('user', text);
      state.history.push({ role: 'user', content: text });
      save();
    }
    input.value = '';
    autoGrow();
    isLoading = true;
    sendBtn.disabled = true;
    typing(true);

    const payload = JSON.stringify({
      messages: state.history.slice(-12).map(m => ({ role: m.role, content: m.content }))
    });

    // Jusqu'à 2 tentatives : une panne passagère est relancée en silence.
    let res = null, data = null, lastErr = null;
    try {
      for (let attempt = 1; attempt <= 2; attempt++) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), CONFIG.requestTimeout);
        try {
          res = await fetch(ENDPOINT, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload, signal: controller.signal });
          try { data = await res.json(); } catch (e) { data = null; }
          lastErr = null;
        } catch (err) {
          res = null; data = null; lastErr = err;
        } finally {
          clearTimeout(timer);
        }
        if (data && data.diag) console.info('[ThermoData chat]', data.model || data.error, data.diag);
        const retryable = !res || [500, 502, 503, 504].includes(res.status);
        if (!retryable || attempt === 2) break;
        await new Promise(r => setTimeout(r, 600));
      }

      typing(false);

      if (res && res.ok && data && data.reply) {
        const relance = data.relance && data.relance.question && data.relance.message ? data.relance : null;
        bubble('bot', data.reply, relance);
        state.history.push({ role: 'assistant', content: data.reply, relance: relance });
        state.last = { suggestions: Array.isArray(data.suggestions) ? data.suggestions : [], cta: data.cta || 'none' };
        save();
        renderQuick(state.last.suggestions, state.last.cta);
        return;
      }

      if (res && res.status === 429) {
        errorBubble(TU ? "Beaucoup de messages d'un coup : patiente une minute puis réessaie." : "Beaucoup de messages d'un coup : patientez une minute puis réessayez.", text);
      } else if (!res && lastErr && lastErr.name === 'AbortError') {
        errorBubble('La réponse prend trop de temps. ' + (TU ? 'Réessaie, ou ' : 'Réessayez, ou ') + contactPhraseHtml() + '.', text);
      } else if (!res) {
        errorBubble('Connexion impossible. ' + (TU ? 'Vérifie ton réseau et réessaie, ou ' : 'Vérifiez votre réseau et réessayez, ou ') + contactPhraseHtml() + '.', text);
      } else {
        errorBubble("😕 Le chat est momentanément indisponible. " + (TU ? 'Réessaie, ou ' : 'Réessayez, ou ') + contactPhraseHtml() + ', réponse sous 24h ouvrées.', text);
      }
    } finally {
      typing(false);
      isLoading = false;
      sendBtn.disabled = false;
      if (isOpen && !isMobile()) input.focus();
    }
  }

  // =========================================================================
  // OUVERTURE / FERMETURE / INVITATION
  // =========================================================================
  function hideTeaser(remember) {
    teaser.classList.remove('show');
    if (remember) { try { sessionStorage.setItem('thermodata_teaser_seen', '1'); } catch (e) { /* ignore */ } }
  }
  function openPanel() {
    isOpen = true;
    hideTeaser(true);
    panel.classList.add('open');
    toggle.classList.add('open');
    toggle.setAttribute('aria-expanded', 'true');
    toggle.setAttribute('aria-label', "Fermer l'assistant");
    if (badge) badge.style.display = 'none';
    if (!rendered) renderAll();
    scrollDown();
    if (!isMobile()) setTimeout(() => input.focus(), 80);
  }
  function closePanel() {
    if (!isOpen) return;
    isOpen = false;
    panel.classList.remove('open');
    toggle.classList.remove('open');
    toggle.setAttribute('aria-expanded', 'false');
    toggle.setAttribute('aria-label', "Ouvrir l'assistant");
  }
  function resetConversation() {
    if (isLoading) return;
    state = { history: [], last: null };
    save();
    renderAll();
    if (!isMobile()) input.focus();
  }
  function autoGrow() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 110) + 'px';
  }

  toggle.addEventListener('click', () => (isOpen ? closePanel() : openPanel()));
  $('tdc-close').addEventListener('click', () => { closePanel(); toggle.focus(); });
  $('tdc-reset').addEventListener('click', resetConversation);
  sendBtn.addEventListener('click', () => ask(input.value));
  input.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(input.value); }
  });
  input.addEventListener('input', autoGrow);
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && isOpen) { closePanel(); toggle.focus(); }
  });
  teaser.addEventListener('click', function (e) {
    if (e.target.id === 'tdc-teaser-close') { e.stopPropagation(); hideTeaser(true); return; }
    openPanel();
  });
  teaser.addEventListener('keydown', function (e) {
    if (e.target.id === 'tdc-teaser-close') return;
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openPanel(); }
  });

  let teaserSeen = false;
  try { teaserSeen = sessionStorage.getItem('thermodata_teaser_seen') === '1'; } catch (e) { /* ignore */ }
  if (CONFIG.teaser && !teaserSeen && !state.history.length) {
    $('tdc-teaser-text').textContent = CONFIG.teaser;
    setTimeout(function () { if (!isOpen) teaser.classList.add('show'); }, CONFIG.teaserDelay);
  }

  // API publique : window.ThermoDataChat.open('Quels sont vos tarifs ?')
  window.ThermoDataChat = {
    open: function (question) { openPanel(); if (question) ask(question); },
    close: closePanel
  };
})();
