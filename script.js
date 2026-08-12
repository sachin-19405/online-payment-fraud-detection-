// The Ledger — client-side interaction
// Handles: health check, form submit -> /api/predict, dial animation,
// verdict card, and the running session ledger list.

const form = document.getElementById('predict-form');
const submitBtn = document.getElementById('submit-btn');
const errorEl = document.getElementById('form-error');
const dialArc = document.getElementById('dial-arc');
const dialNeedle = document.getElementById('dial-needle');
const dialValue = document.getElementById('dial-value');
const verdictCard = document.getElementById('verdict-card');
const ledger = document.getElementById('ledger');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

const ARC_LENGTH = 314; // matches stroke-dasharray in the SVG

async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data.status === 'ok') {
      statusDot.classList.add('ok');
      statusText.textContent = 'model ready';
    } else {
      statusDot.classList.add('bad');
      statusText.textContent = 'model unavailable';
    }
  } catch (e) {
    statusDot.classList.add('bad');
    statusText.textContent = 'offline';
  }
}
checkHealth();

function setDial(prob) {
  const angleDeg = -90 + prob * 180; // -90 (left) .. +90 (right)
  dialNeedle.setAttribute('transform', `rotate(${angleDeg} 120 130)`);
  const offset = ARC_LENGTH - prob * ARC_LENGTH;
  dialArc.style.strokeDashoffset = offset;
  dialValue.textContent = (prob * 100).toFixed(2) + '%';
}

function riskClass(prob) {
  if (prob >= 0.75) return 'danger';
  if (prob >= 0.25) return 'warn';
  return 'safe';
}

function updateVerdictCard(result) {
  const cls = riskClass(result.fraudProbability);
  verdictCard.className = 'verdict-card ' + cls;

  const labelEl = verdictCard.querySelector('.verdict-card__label') || document.createElement('p');
  const bodyEl = verdictCard.querySelector('.verdict-card__body') || document.createElement('p');
  labelEl.className = 'verdict-card__label';
  bodyEl.className = 'verdict-card__body';

  const labels = {
    danger: 'High risk — likely fraud',
    warn: 'Medium risk — review recommended',
    safe: 'Low risk — looks clean',
  };
  const bodies = {
    danger: 'The balance movement on this transaction matches patterns the model associates strongly with fraudulent transfers. Hold the transaction for manual review before it settles.',
    warn: 'Some balance signals here are unusual but not conclusive. Consider a second check against the account history before releasing funds.',
    safe: 'Balances reconcile in the way genuine transactions typically do. No action needed based on this score alone.',
  };
  labelEl.textContent = labels[cls];
  bodyEl.textContent = bodies[cls];

  verdictCard.innerHTML = '';
  verdictCard.appendChild(labelEl);
  verdictCard.appendChild(bodyEl);
}

function addLedgerEntry(type, amount, result) {
  const empty = ledger.querySelector('.ledger__empty');
  if (empty) empty.remove();

  const li = document.createElement('li');
  li.className = 'entry';
  const cls = riskClass(result.fraudProbability);
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  li.innerHTML = `
    <span class="entry__left">${time} &middot; <span class="entry__type">${type}</span> &middot; $${Number(amount).toLocaleString()}</span>
    <span class="entry__prob ${cls}">${(result.fraudProbability * 100).toFixed(2)}%</span>
  `;
  ledger.prepend(li);
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  errorEl.hidden = true;
  submitBtn.disabled = true;
  submitBtn.textContent = 'Scoring…';

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || 'Prediction failed.');
    }

    setDial(data.fraudProbability);
    updateVerdictCard(data);
    addLedgerEntry(payload.type, payload.amount, data);
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Score transaction';
  }
});

document.getElementById('fill-fraud').addEventListener('click', () => {
  form.type.value = 'TRANSFER';
  form.amount.value = 181;
  form.oldbalanceOrg.value = 181;
  form.newbalanceOrig.value = 0;
  form.oldbalanceDest.value = 0;
  form.newbalanceDest.value = 0;
  form.nameDest.value = 'C553264065';
});

document.getElementById('fill-clean').addEventListener('click', () => {
  form.type.value = 'PAYMENT';
  form.amount.value = 9839.64;
  form.oldbalanceOrg.value = 170136;
  form.newbalanceOrig.value = 160296.36;
  form.oldbalanceDest.value = 0;
  form.newbalanceDest.value = 0;
  form.nameDest.value = 'M1979787155';
});
