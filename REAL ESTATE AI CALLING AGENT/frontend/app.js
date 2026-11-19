/**
 * RealEstateAI Voice Agent — Clean Phone Call Screen Controller
 * Features:
 * - Direct AudioContext initialization on user gesture
 * - PcmAudioPlayer for jitter-free 16kHz audio playback
 * - Acoustic echo suppression: pauses mic upload while agent is speaking
 */

// Application State
const state = {
  sessionId: null,
  ws: null,
  isConnected: false,
  isMuted: false,
  isAgentSpeaking: false,
  callDuration: 0,
  callTimerInterval: null,
  micStream: null,
  scriptProcessor: null,
  audioCtx: null,
};

// DOM Elements
const elements = {
  statusDot: document.getElementById('statusDot'),
  callStatusText: document.getElementById('callStatusText'),
  avatarRing: document.getElementById('avatarRing'),
  phoneInput: document.getElementById('phoneInput'),
  callTimer: document.getElementById('callTimer'),
  waveContainer: document.getElementById('waveContainer'),
  captionsBox: document.getElementById('captionsBox'),
  captionSpeaker: document.getElementById('captionSpeaker'),
  captionText: document.getElementById('captionText'),
  btnToggleMic: document.getElementById('btnToggleMic'),
  btnCall: document.getElementById('btnCall'),
  callIcon: document.getElementById('callIcon'),
  btnToggleKeyboard: document.getElementById('btnToggleKeyboard'),
  keyboardContainer: document.getElementById('keyboardContainer'),
  textPromptInput: document.getElementById('textPromptInput'),
  btnSendText: document.getElementById('btnSendText'),
  interruptBadge: document.getElementById('interruptBadge'),
};

const CALL_ICON_SVG = `
  <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path>
`;

const HANGUP_ICON_SVG = `
  <path d="M10.68 13.31a16 16 0 0 0 3.41 2.6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7 2 2 0 0 1 1.72 2v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.42 19.42 0 0 1-3.33-2.67m-2.67-3.34a19.79 19.79 0 0 1-3.07-8.63A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91"></path>
  <line x1="23" y1="1" x2="1" y2="23"></line>
`;

// --- PCM AUDIO STREAM PLAYER ---
class PcmAudioPlayer {
  constructor() {
    this.audioCtx = null;
    this.nextTime = 0;
    this.isPlaying = false;
    this.sampleRate = 16000;
    this.pendingChunks = [];
    this.flushTimer = null;
    this.activeSources = [];
  }

  ensureContext() {
    if (!this.audioCtx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioCtx();
    }
    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
    return this.audioCtx;
  }

  feed(arrayBuffer) {
    this.ensureContext();
    this.isPlaying = true;
    setSpeakingVisual(true);
    showInterruptBadge(true);
    this.pendingChunks.push(new Int16Array(arrayBuffer));
    if (!this.flushTimer) {
      this.flushTimer = setTimeout(() => this.flush(), 40);
    }
  }

  flush() {
    this.flushTimer = null;
    if (this.pendingChunks.length === 0 || !this.audioCtx) return;

    let totalSamples = 0;
    for (const chunk of this.pendingChunks) {
      totalSamples += chunk.length;
    }

    const float32 = new Float32Array(totalSamples);
    let offset = 0;
    for (const chunk of this.pendingChunks) {
      for (let i = 0; i < chunk.length; i++) {
        float32[offset + i] = chunk[i] / (chunk[i] < 0 ? 0x8000 : 0x7fff);
      }
      offset += chunk.length;
    }
    this.pendingChunks = [];

    const audioBuffer = this.audioCtx.createBuffer(1, float32.length, this.sampleRate);
    audioBuffer.copyToChannel(float32, 0);

    const source = this.audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioCtx.destination);

    const now = this.audioCtx.currentTime;
    if (this.nextTime < now) {
      this.nextTime = now + 0.03;
    }

    source.start(this.nextTime);
    this.nextTime += audioBuffer.duration;
    this.activeSources.push(source);

    this.isPlaying = true;
    setSpeakingVisual(true);
    showInterruptBadge(true);

    source.onended = () => {
      const idx = this.activeSources.indexOf(source);
      if (idx !== -1) {
        this.activeSources.splice(idx, 1);
      }
      if (this.activeSources.length === 0 && !state.isAgentSpeaking) {
        this.isPlaying = false;
        setSpeakingVisual(false);
        showInterruptBadge(false);
      }
    };
  }

  stop() {
    this.pendingChunks = [];
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
    for (const s of this.activeSources) {
      try {
        s.stop();
      } catch (e) {}
    }
    this.activeSources = [];
    if (this.audioCtx) {
      this.nextTime = this.audioCtx.currentTime;
    }
    this.isPlaying = false;
    setSpeakingVisual(false);
    showInterruptBadge(false);
  }
}

const player = new PcmAudioPlayer();

// --- INITIALIZATION ---
window.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
});

function setupEventListeners() {
  elements.btnCall.addEventListener('click', () => {
    // Explicit user-gesture AudioContext resume
    player.ensureContext();

    if (state.isConnected) {
      endCall();
    } else {
      startCall();
    }
  });

  elements.btnToggleMic.addEventListener('click', toggleMute);

  elements.btnToggleKeyboard.addEventListener('click', () => {
    elements.keyboardContainer.classList.toggle('hidden');
    if (!elements.keyboardContainer.classList.contains('hidden')) {
      elements.textPromptInput.focus();
    }
  });

  elements.btnSendText.addEventListener('click', sendTextMessage);
  elements.textPromptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendTextMessage();
  });

  // Tap to interrupt Aria while she is speaking
  if (elements.interruptBadge) {
    elements.interruptBadge.addEventListener('click', (e) => {
      e.stopPropagation();
      interruptAgent();
    });
  }
  elements.avatarRing.style.cursor = 'pointer';
  elements.avatarRing.addEventListener('click', interruptAgent);
  elements.captionsBox.style.cursor = 'pointer';
  elements.captionsBox.addEventListener('click', interruptAgent);

  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target !== elements.phoneInput && e.target !== elements.textPromptInput) {
      if (player.isPlaying) {
        e.preventDefault();
        interruptAgent();
      }
    }
  });
}

function interruptAgent() {
  if (!state.isConnected) return;
  if (player.isPlaying) {
    player.stop();
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: 'user_interrupt' }));
    }
    updateCaption('You', 'Listening...');
  }
}

// --- CALL MANAGEMENT ---
async function startCall() {
  const phone = elements.phoneInput.value.trim() || '+919876543210';
  state.sessionId = `phone_call_${Date.now().toString(36)}`;

  elements.callStatusText.textContent = 'Connecting...';
  elements.statusDot.classList.remove('idle');
  updateCaption('Aria', 'Connecting call...');

  try {
    const res = await fetch('/api/calls/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        caller_phone: phone,
        caller_name: 'Caller',
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    connectAudioWebSocket(state.sessionId, phone);

    // Transform Call Button to Hangup Button
    elements.btnCall.classList.remove('btn-call');
    elements.btnCall.classList.add('btn-hangup');
    elements.btnCall.title = 'End Call';
    elements.callIcon.innerHTML = HANGUP_ICON_SVG;

    elements.btnToggleMic.disabled = false;
    elements.btnToggleMic.classList.add('active');
    elements.phoneInput.disabled = true;

    startTimer();
    startMicrophone();
  } catch (err) {
    console.error('Call start error:', err);
    elements.callStatusText.textContent = 'Connection Failed';
    elements.statusDot.classList.add('idle');
  }
}

async function endCall() {
  if (state.sessionId) {
    try {
      await fetch('/api/calls/end', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: state.sessionId }),
      });
    } catch (err) {
      console.warn('Call end request error:', err);
    }
  }

  stopMicrophone();
  player.stop();

  if (state.ws) {
    state.ws.close();
    state.ws = null;
  }

  state.isConnected = false;
  state.isMuted = false;

  // Transform Hangup Button back to Call Button
  elements.btnCall.classList.remove('btn-hangup');
  elements.btnCall.classList.add('btn-call');
  elements.btnCall.title = 'Call Aria';
  elements.callIcon.innerHTML = CALL_ICON_SVG;

  elements.btnToggleMic.disabled = true;
  elements.btnToggleMic.classList.remove('active', 'muted');
  elements.phoneInput.disabled = false;

  elements.callStatusText.textContent = 'Call Ended';
  elements.statusDot.classList.add('idle');
  setSpeakingVisual(false);

  stopTimer();
  updateCaption('Aria', 'Call ended. Click the button to start another call.');
}

// --- WEBSOCKET CONNECTION ---
function connectAudioWebSocket(sessionId, phone) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const encodedPhone = encodeURIComponent(phone);
  const wsUrl = `${protocol}//${window.location.host}/ws/audio/${sessionId}?phone=${encodedPhone}`;

  state.ws = new WebSocket(wsUrl);
  state.ws.binaryType = 'arraybuffer';

  state.ws.onopen = () => {
    state.isConnected = true;
    elements.callStatusText.textContent = 'On Call with Aria';
    player.ensureContext();
  };

  state.ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
      try {
        const payload = JSON.parse(event.data);
        handleControlMessage(payload);
      } catch (err) {
        console.warn('Non-JSON packet:', err);
      }
    } else if (event.data instanceof ArrayBuffer) {
      // Inbound 16kHz binary PCM chunk
      player.feed(event.data);
    }
  };

  state.ws.onclose = () => {
    if (state.isConnected) {
      endCall();
    }
  };

  state.ws.onerror = (err) => {
    console.error('Telephony WebSocket error:', err);
  };
}

function handleControlMessage(msg) {
  if (msg.type === 'agent_speech_start') {
    state.isAgentSpeaking = true;
    player.isPlaying = true;
    setSpeakingVisual(true);
    showInterruptBadge(true);
  } else if (msg.type === 'greeting_metadata') {
    updateCaption('Aria', msg.agent_response || "Hi, I'm Aria from Prime Realty. How can I help you find your ideal property today?");
    elements.callStatusText.textContent = 'On Call with Aria';
  } else if (msg.type === 'turn_transcription') {
    if (msg.transcript) {
      updateCaption('You', msg.transcript);
      elements.callStatusText.textContent = 'Aria is thinking...';
    }
  } else if (msg.type === 'turn_metadata') {
    if (msg.transcript) {
      updateCaption('You', msg.transcript);
    }
    if (msg.agent_response) {
      updateCaption('Aria', msg.agent_response);
      elements.callStatusText.textContent = 'On Call with Aria';
    }
  } else if (msg.type === 'playback_interrupted') {
    state.isAgentSpeaking = false;
    player.stop();
    showInterruptBadge(false);
  } else if (msg.type === 'turn_end' || msg.type === 'greeting_end') {
    const delay = Math.max(0, ((player.nextTime || 0) - (player.audioCtx?.currentTime || 0)) * 1000) + 150;
    setTimeout(() => {
      state.isAgentSpeaking = false;
      player.isPlaying = false;
      setSpeakingVisual(false);
      showInterruptBadge(false);
    }, delay);
  }
}

function showInterruptBadge(show) {
  if (elements.interruptBadge) {
    if (show) {
      elements.interruptBadge.classList.remove('hidden');
    } else {
      elements.interruptBadge.classList.add('hidden');
    }
  }
}

function updateCaption(speaker, text) {
  elements.captionSpeaker.textContent = speaker;
  elements.captionSpeaker.className = `caption-speaker ${speaker === 'You' ? 'caller' : ''}`;
  elements.captionText.textContent = text;
}

// --- MICROPHONE AUDIO (16kHz PCM Downsampling) ---
async function startMicrophone() {
  const audioCtx = player.ensureContext();
  try {
    state.micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    const source = audioCtx.createMediaStreamSource(state.micStream);
    const bufferSize = 4096;
    state.scriptProcessor = audioCtx.createScriptProcessor(bufferSize, 1, 1);

    const inputSampleRate = audioCtx.sampleRate;
    const targetSampleRate = 16000;

    state.scriptProcessor.onaudioprocess = (e) => {
      if (!state.isConnected || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
      if (state.isMuted) return;

      const inputData = e.inputBuffer.getChannelData(0);

      // Auto-clear isPlaying flag as soon as audio playback finishes in AudioContext
      if (player.isPlaying && player.audioCtx && player.audioCtx.currentTime >= player.nextTime) {
        player.isPlaying = false;
        setSpeakingVisual(false);
        showInterruptBadge(false);
      }

      // Acoustic echo suppression: Pause mic upload while agent is actively speaking through computer speakers
      // unless user clicks the on-screen interrupt button or presses spacebar
      if (state.isAgentSpeaking || player.isPlaying) {
        return;
      }

      // Measure RMS audio energy
      let sum = 0;
      for (let i = 0; i < inputData.length; i++) {
        sum += inputData[i] * inputData[i];
      }
      const rms = Math.sqrt(sum / inputData.length);

      if (rms > 0.02) {
        setSpeakingVisual(true);
      } else {
        setSpeakingVisual(false);
      }

      // Downsample Float32 to 16kHz 16-bit mono PCM
      const pcm16Buffer = downsampleTo16kPCM(inputData, inputSampleRate, targetSampleRate);

      // Stream 640-byte (20ms) slices
      const chunkSize = 640;
      for (let offset = 0; offset < pcm16Buffer.byteLength; offset += chunkSize) {
        const slice = pcm16Buffer.slice(offset, offset + chunkSize);
        if (slice.byteLength === chunkSize) {
          state.ws.send(slice);
        }
      }
    };

    source.connect(state.scriptProcessor);
    // Connect through a zero-gain mute node to keep ScriptProcessor processing loop alive
    // without routing live microphone audio into the caller's computer speakers
    const muteNode = audioCtx.createGain();
    muteNode.gain.value = 0.0;
    state.scriptProcessor.connect(muteNode);
    muteNode.connect(audioCtx.destination);
  } catch (err) {
    console.warn('Microphone access failed:', err);
    updateCaption('Aria', 'Microphone not accessible. You can use the keyboard button to type.');
  }
}

function stopMicrophone() {
  if (state.scriptProcessor) {
    state.scriptProcessor.disconnect();
    state.scriptProcessor = null;
  }
  if (state.micStream) {
    state.micStream.getTracks().forEach((t) => t.stop());
    state.micStream = null;
  }
}

function toggleMute() {
  state.isMuted = !state.isMuted;
  if (state.isMuted) {
    elements.btnToggleMic.classList.remove('active');
    elements.btnToggleMic.classList.add('muted');
    elements.btnToggleMic.title = 'Unmute Microphone';
  } else {
    elements.btnToggleMic.classList.remove('muted');
    elements.btnToggleMic.classList.add('active');
    elements.btnToggleMic.title = 'Mute Microphone';
  }
}

// Convert Float32 buffer to 16kHz Int16 ArrayBuffer
function downsampleTo16kPCM(inputData, inputRate, outputRate = 16000) {
  if (inputRate === outputRate) {
    const pcm = new Int16Array(inputData.length);
    for (let i = 0; i < inputData.length; i++) {
      const s = Math.max(-1, Math.min(1, inputData[i]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return pcm.buffer;
  }

  const ratio = inputRate / outputRate;
  const newLength = Math.round(inputData.length / ratio);
  const result = new Int16Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < inputData.length; i++) {
      accum += inputData[i];
      count++;
    }
    const sample = Math.max(-1, Math.min(1, count > 0 ? accum / count : 0));
    result[offsetResult] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result.buffer;
}

function setSpeakingVisual(isSpeaking) {
  if (isSpeaking) {
    elements.avatarRing.classList.add('speaking');
    elements.waveContainer.classList.add('active-wave');
  } else {
    elements.avatarRing.classList.remove('speaking');
    elements.waveContainer.classList.remove('active-wave');
  }
}

// --- KEYBOARD TEXT FALLBACK ---
function sendTextMessage() {
  const text = elements.textPromptInput.value.trim();
  if (!text) return;
  elements.textPromptInput.value = '';

  if (!state.isConnected || !state.ws || state.ws.readyState !== WebSocket.OPEN) {
    alert('Please click the call button to start the call first.');
    return;
  }

  updateCaption('You', text);
  state.ws.send(JSON.stringify({ type: 'user_text', text: text }));
}

// --- CALL DURATION TIMER ---
function startTimer() {
  stopTimer();
  state.callDuration = 0;
  elements.callTimer.textContent = '00:00';
  state.callTimerInterval = setInterval(() => {
    state.callDuration++;
    const mins = String(Math.floor(state.callDuration / 60)).padStart(2, '0');
    const secs = String(state.callDuration % 60).padStart(2, '0');
    elements.callTimer.textContent = `${mins}:${secs}`;
  }, 1000);
}

function stopTimer() {
  if (state.callTimerInterval) {
    clearInterval(state.callTimerInterval);
    state.callTimerInterval = null;
  }
}
