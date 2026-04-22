/* =========================================================================
   ThermoData Chatbot Widget - version "service séparé"
   À héberger sur un site Netlify dédié (ex: thermodata-chatbot.netlify.app)
   et à embarquer sur thermodata.fr via :
     <script src="https://thermodata-chatbot.netlify.app/chatbot.js" defer></script>
   ========================================================================= */
(function () {
  'use strict';

  // ⚠️ EDIT THIS : URL du service où tu héberges ce widget + la function /chat
  // Si tu déploies sur Netlify, ce sera l'URL de ton site Netlify.
  // Pas de slash final.
  const SERVICE_BASE_URL = 'https://thermodata-chatbot.netlify.app';

  // ------------------ CONFIG ------------------
  const CONFIG = {
    apiEndpoint: SERVICE_BASE_URL + '/.netlify/functions/chat',
    contactEmail: 'contact@thermodata.fr',
    botName: 'Assistant ThermoData',
    welcomeMessage: "Bonjour 👋 Je suis l'assistant ThermoData. Posez-moi vos questions sur nos offres, nos données DPE ou la prospection RGE !",
    quickReplies: [
      'Quelles sont vos offres ?',
      "D'où viennent vos données ?",
      'Comment ça marche ?'
    ]
  };

  // Évite le double-chargement si le script est inclus deux fois
  if (document.getElementById('td-chatbot')) return;

  // ------------------ STYLES ------------------
  const css = `
#td-chatbot, #td-chatbot * { box-sizing: border-box; font-family: 'Outfit', system-ui, -apple-system, sans-serif; }

#td-chatbot-toggle {
  position: fixed; bottom: 24px; right: 24px;
  width: 60px; height: 60px; border-radius: 50%;
  background: linear-gradient(135deg, #E67E22 0%, #D85A30 100%);
  border: none; cursor: pointer;
  box-shadow: 0 8px 24px rgba(230, 126, 34, 0.4);
  z-index: 99999;
  display: flex; align-items: center; justify-content: center;
  transition: transform 0.2s;
}
#td-chatbot-toggle:hover { transform: scale(1.08); }
#td-chatbot-toggle svg { width: 26px; height: 26px; fill: white; transition: transform 0.2s; }
#td-chatbot-toggle.open svg { transform: rotate(180deg); }
#td-chatbot-toggle .td-badge {
  position: absolute; top: -2px; right: -2px;
  background: #16A34A; color: white;
  font-size: 10px; font-weight: 700;
  padding: 2px 6px; border-radius: 100px;
  border: 2px solid white;
}

#td-chatbot-panel {
  position: fixed; bottom: 96px; right: 24px;
  width: 380px; max-width: calc(100vw - 32px);
  height: 560px; max-height: calc(100vh - 140px);
  background: white; border-radius: 20px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
  z-index: 99998;
  display: none; flex-direction: column;
  overflow: hidden;
  border: 1px solid rgba(0, 0, 0, 0.06);
}
#td-chatbot-panel.open {
  display: flex;
  animation: tdSlideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes tdSlideUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}

.td-header {
  background: linear-gradient(135deg, #0B1A2F 0%, #1E3A5F 100%);
  padding: 14px 18px; color: white;
  display: flex; align-items: center; gap: 12px;
}
.td-avatar {
  width: 38px; height: 38px; border-radius: 50%;
  background: white; display: flex;
  align-items: center; justify-content: center;
  font-size: 18px; flex-shrink: 0;
}
.td-header-info { flex: 1; min-width: 0; }
.td-header-name { font-size: 14px; font-weight: 700; margin: 0; line-height: 1.2; color: white; }
.td-header-status { font-size: 11px; opacity: 0.75; margin: 3px 0 0; display: flex; align-items: center; gap: 6px; color: white; }
.td-status-dot { width: 7px; height: 7px; border-radius: 50%; background: #22C55E; }
.td-close {
  background: none; border: none; color: white;
  cursor: pointer; padding: 6px; border-radius: 8px;
  opacity: 0.7; display: flex;
}
.td-close:hover { opacity: 1; background: rgba(255, 255, 255, 0.1); }

.td-messages {
  flex: 1; overflow-y: auto; padding: 18px;
  background: #F9FAFB; display: flex;
  flex-direction: column; gap: 10px;
}
.td-msg {
  max-width: 85%; padding: 10px 14px;
  border-radius: 16px; font-size: 14px; line-height: 1.5;
  word-wrap: break-word;
}
.td-msg-bot {
  background: white; color: #1F2937;
  border: 1px solid #E5E7EB;
  align-self: flex-start;
  border-bottom-left-radius: 4px;
}
.td-msg-user {
  background: linear-gradient(135deg, #E67E22, #D85A30);
  color: white; align-self: flex-end;
  border-bottom-right-radius: 4px;
}
.td-msg-error {
  background: #FEF2F2; color: #991B1B;
  border: 1px solid #FECACA;
  align-self: stretch; max-width: 100%;
  font-size: 13px;
}
.td-msg-error a { color: #991B1B; font-weight: 700; text-decoration: underline; }

.td-typing {
  display: flex; gap: 4px;
  padding: 12px 14px; background: white;
  border: 1px solid #E5E7EB;
  border-radius: 16px; border-bottom-left-radius: 4px;
  align-self: flex-start; width: fit-content;
}
.td-typing span {
  width: 7px; height: 7px;
  background: #9CA3AF; border-radius: 50%;
  animation: tdBounce 1.4s infinite both;
}
.td-typing span:nth-child(2) { animation-delay: 0.16s; }
.td-typing span:nth-child(3) { animation-delay: 0.32s; }
@keyframes tdBounce {
  0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
  40%           { transform: scale(1);   opacity: 1;   }
}

.td-quick {
  display: flex; flex-wrap: wrap; gap: 6px;
  padding: 0 18px 8px; background: #F9FAFB;
}
.td-quick button {
  background: white; border: 1px solid #E5E7EB;
  color: #4B5563; padding: 6px 12px;
  border-radius: 100px; font-size: 12px;
  cursor: pointer; transition: all 0.2s;
  font-family: inherit;
}
.td-quick button:hover {
  background: #FFF7ED; border-color: #E67E22; color: #C2410C;
}

.td-input-area {
  padding: 12px 14px;
  background: white; border-top: 1px solid #E5E7EB;
  display: flex; gap: 8px; align-items: flex-end;
}
.td-input {
  flex: 1; border: 1px solid #D1D5DB;
  border-radius: 12px; padding: 9px 12px;
  font-size: 14px; resize: none;
  max-height: 100px; min-height: 40px;
  outline: none; font-family: inherit;
  line-height: 1.4; color: #1F2937;
  background: white;
}
.td-input:focus {
  border-color: #E67E22;
  box-shadow: 0 0 0 3px rgba(230, 126, 34, 0.1);
}
.td-send {
  background: linear-gradient(135deg, #E67E22, #D85A30);
  border: none; width: 40px; height: 40px;
  border-radius: 50%; cursor: pointer;
  flex-shrink: 0; display: flex;
  align-items: center; justify-content: center;
  transition: transform 0.2s;
}
.td-send:hover:not(:disabled) { transform: scale(1.05); }
.td-send:disabled { opacity: 0.4; cursor: not-allowed; }
.td-send svg { width: 18px; height: 18px; fill: white; }

.td-footer {
  font-size: 10px; color: #9CA3AF;
  text-align: center; padding: 6px 0 8px;
  background: white;
}
.td-footer a { color: #6B7280; text-decoration: none; }
.td-footer a:hover { text-decoration: underline; }

@media (max-width: 480px) {
  #td-chatbot-panel {
    right: 8px; left: 8px; width: auto;
    max-width: none; bottom: 86px;
  }
  #td-chatbot-toggle { right: 16px; bottom: 16px; }
}
`;

  const styleEl = document.createElement('style');
  styleEl.textContent = css;
  document.head.appendChild(styleEl);

  // ------------------ HTML ------------------
  const container = document.createElement('div');
  container.id = 'td-chatbot';
  container.innerHTML = `
    <button id="td-chatbot-toggle" aria-label="Ouvrir le chat ThermoData">
      <svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z"/></svg>
      <span class="td-badge" id="td-badge">1</span>
    </button>
    <div id="td-chatbot-panel" role="dialog" aria-label="Chat ThermoData">
      <div class="td-header">
        <div class="td-avatar">🤖</div>
        <div class="td-header-info">
          <p class="td-header-name">${CONFIG.botName}</p>
          <p class="td-header-status"><span class="td-status-dot"></span>En ligne · réponse en quelques secondes · 24h/24, 7j/7</p>
        </div>
        <button class="td-close" aria-label="Fermer le chat">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M19 6.4L17.6 5 12 10.6 6.4 5 5 6.4 10.6 12 5 17.6 6.4 19 12 13.4 17.6 19 19 17.6 13.4 12z"/></svg>
        </button>
      </div>
      <div class="td-messages" id="td-messages"></div>
      <div class="td-quick" id="td-quick"></div>
      <div class="td-input-area">
        <textarea class="td-input" id="td-input" placeholder="Votre message..." rows="1" aria-label="Tapez votre message"></textarea>
        <button class="td-send" id="td-send" aria-label="Envoyer le message">
          <svg viewBox="0 0 24 24"><path d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg>
        </button>
      </div>
      <div class="td-footer">Assistant IA · <a href="mailto:${CONFIG.contactEmail}">Nous écrire directement</a></div>
    </div>
  `;
  document.body.appendChild(container);

  // ------------------ STATE ------------------
   const messages = [];
   let isOpen = false;
   let isLoading = false;
   let hasFailed = false;
   let hasShownWelcome = false;

  const toggle    = document.getElementById('td-chatbot-toggle');
  const panel     = document.getElementById('td-chatbot-panel');
  const closeBtn  = container.querySelector('.td-close');
  const messagesEl = document.getElementById('td-messages');
  const quickEl   = document.getElementById('td-quick');
  const input     = document.getElementById('td-input');
  const sendBtn   = document.getElementById('td-send');
  const badge     = document.getElementById('td-badge');

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[c]
    ));
  }

  function addMessage(role, html, isError) {
    const msg = document.createElement('div');
    msg.className = 'td-msg td-msg-' + (isError ? 'error' : role);
    msg.innerHTML = html;
    messagesEl.appendChild(msg);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function showTyping() {
    if (document.getElementById('td-typing')) return;
    const t = document.createElement('div');
    t.className = 'td-typing';
    t.id = 'td-typing';
    t.innerHTML = '<span></span><span></span><span></span>';
    messagesEl.appendChild(t);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function hideTyping() {
    const t = document.getElementById('td-typing');
    if (t) t.remove();
  }

  function renderQuickReplies(replies) {
    quickEl.innerHTML = '';
    if (!replies || hasFailed) return;
    replies.forEach(r => {
      const btn = document.createElement('button');
      btn.textContent = r;
      btn.onclick = () => { input.value = r; sendMessage(); };
      quickEl.appendChild(btn);
    });
  }

  function showFallbackError() {
    hasFailed = true;
    quickEl.innerHTML = '';
    const subject = encodeURIComponent('Question depuis le chat ThermoData');
    addMessage(
      'error',
      `😕 Le chat est temporairement indisponible.<br><br>` +
      `Écrivez-nous à <a href="mailto:${CONFIG.contactEmail}?subject=${subject}">${CONFIG.contactEmail}</a> ` +
      `- réponse sous 24h ouvrées.`,
      true
    );
    sendBtn.disabled = true;
    input.disabled = true;
    input.placeholder = "Chat indisponible - utilisez l'email";
  }

   function openPanel() {
     isOpen = true;
     panel.classList.add('open');
     toggle.classList.add('open');
     if (badge) badge.style.display = 'none';
   
     if (!hasShownWelcome) {
       addMessage('bot', escapeHtml(CONFIG.welcomeMessage));
       renderQuickReplies(CONFIG.quickReplies);
       hasShownWelcome = true;
     }
   
     setTimeout(() => input.focus(), 100);
   }

  function closePanel() {
    isOpen = false;
    panel.classList.remove('open');
    toggle.classList.remove('open');
  }

  toggle.addEventListener('click', () => isOpen ? closePanel() : openPanel());
  closeBtn.addEventListener('click', closePanel);

  async function sendMessage() {
    const text = input.value.trim();
    if (!text || isLoading || hasFailed) return;

    quickEl.innerHTML = '';
    addMessage('user', escapeHtml(text));
    messages.push({ role: 'user', content: text });
    input.value = '';
    input.style.height = 'auto';
    isLoading = true;
    sendBtn.disabled = true;
    showTyping();

    try {
      const res = await fetch(CONFIG.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages })
      });

      hideTyping();

      // Codes "plus de crédit / API down" => fallback email
      if ([401, 402, 429, 500, 502, 503, 504].includes(res.status)) {
        showFallbackError();
        return;
      }

      if (!res.ok) throw new Error('HTTP ' + res.status);

      const data = await res.json();
      if (!data.reply) {
        showFallbackError();
        return;
      }

      addMessage('bot', escapeHtml(data.reply).replace(/\n/g, '<br>'));
      messages.push({ role: 'assistant', content: data.reply });
    } catch (err) {
      hideTyping();
      console.error('[ThermoData chatbot]', err);
      showFallbackError();
    } finally {
      isLoading = false;
      sendBtn.disabled = hasFailed;
      if (!hasFailed) input.focus();
    }
  }

  sendBtn.addEventListener('click', sendMessage);
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 100) + 'px';
  });
})();
