/**
 * Recorder Controller - Audio/Video recording with World Engine integration
 *
 * Handles mic/screen recording, transcription, and clip management.
 * Integrates with the Studio Bridge for seamless workflow.
 */

class RecorderController {
  constructor(options = {}) {
    this.options = {
      transcription: false, // Set to true if you have a transcription service
      autoMarkRuns: true,
      chunkSize: 200,
      ...options
    };

    this.mediaRecorder = null;
    this.mediaStream = null;
    this.chunks = [];
    this.currentClipId = null;
    this.isRecording = false;
    this.transcriptionService = null;

    this.bindEvents();
    this.setupUI();
  }

  bindEvents() {
    onBus(async (msg) => {
      try {
        switch (msg.type) {
          case 'rec.start':
            await this.startRecording(msg.mode, msg.meta);
            break;
          case 'rec.stop':
            await this.stopRecording();
            break;
          case 'rec.mark':
            await this.addMarker(msg.tag, msg.runId);
            break;
          case 'eng.result':
            if (this.options.autoMarkRuns) {
              await this.addMarker('engine-result', msg.runId);
            }
            break;
        }
      } catch (error) {
        Utils.log(`Recorder error: ${error.message}`, 'error');
        sendBus({
          type: 'rec.error',
          error: error.message,
          originalMessage: msg
        });
      }
    });
  }

  setupUI() {
    // Create basic recording UI if container exists
    const container = document.getElementById('recorder-ui');
    if (container) {
      container.innerHTML = `
        <div class="recorder-controls">
          <button id="rec-mic" class="rec-btn">🎤 Start Mic</button>
          <button id="rec-screen" class="rec-btn">🖥️ Start Screen</button>
          <button id="rec-stop" class="rec-btn" disabled>⏹️ Stop</button>
          <div class="rec-status">
            <span id="rec-indicator" class="rec-dot"></span>
            <span id="rec-time">00:00</span>
          </div>
        </div>
        <div class="rec-clips" id="rec-clips"></div>
        <style>
          .recorder-controls { display: flex; gap: 8px; align-items: center; padding: 8px; }
          .rec-btn { padding: 6px 12px; border: 1px solid #333; background: #222; color: #fff; border-radius: 4px; cursor: pointer; }
          .rec-btn:disabled { opacity: 0.5; cursor: not-allowed; }
          .rec-status { margin-left: auto; display: flex; align-items: center; gap: 8px; }
          .rec-dot { width: 8px; height: 8px; border-radius: 50%; background: #666; }
          .rec-dot.recording { background: #f44; animation: pulse 1s infinite; }
          @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
          .rec-clips { padding: 8px; max-height: 200px; overflow-y: auto; }
          .rec-clip { display: flex; align-items: center; gap: 8px; padding: 4px; border: 1px solid #333; margin: 2px 0; border-radius: 4px; }
        </style>
      `;

      // Bind UI events
      document.getElementById('rec-mic')?.addEventListener('click', () => {
        sendBus({ type: 'rec.start', mode: 'mic' });
      });

      document.getElementById('rec-screen')?.addEventListener('click', () => {
        sendBus({ type: 'rec.start', mode: 'screen' });
      });

      document.getElementById('rec-stop')?.addEventListener('click', () => {
        sendBus({ type: 'rec.stop' });
      });
    }
  }

  async startRecording(mode = 'mic', meta = {}) {
    if (this.isRecording) {
      Utils.log('Already recording', 'warn');
      return;
    }

    try {
      this.currentClipId = Utils.generateId();
      this.chunks = [];

      // Get media stream based on mode
      let constraints = {};
      switch (mode) {
        case 'mic':
          constraints = { audio: true, video: false };
          break;
        case 'screen':
          this.mediaStream = await navigator.mediaDevices.getDisplayMedia({
            video: true,
            audio: true
          });
          break;
        case 'both':
          constraints = { audio: true, video: true };
          break;
        default:
          throw new Error(`Unknown recording mode: ${mode}`);
      }

      if (mode !== 'screen') {
        this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      }

      // Setup MediaRecorder
      this.mediaRecorder = new MediaRecorder(this.mediaStream, {
        mimeType: 'audio/webm'
      });

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.chunks.push(event.data);

          // Optional: streaming transcription
          if (this.options.transcription && this.transcriptionService) {
            this.handleChunkTranscription(event.data);
          }
        }
      };

      this.mediaRecorder.onstop = async () => {
        await this.finalizeRecording(meta);
      };

      this.mediaRecorder.onerror = (event) => {
        Utils.log(`MediaRecorder error: ${event.error}`, 'error');
        this.cleanup();
      };

      // Start recording
      this.mediaRecorder.start(this.options.chunkSize);
      this.isRecording = true;
      this.updateUI('recording');

      Utils.log(`Recording started: ${mode} (${this.currentClipId})`);
      sendBus({ type: 'rec.ready', clipId: this.currentClipId, mode, meta });

    } catch (error) {
      Utils.log(`Failed to start recording: ${error.message}`, 'error');
      this.cleanup();
      throw error;
    }
  }

  async stopRecording() {
    if (!this.isRecording || !this.mediaRecorder) {
      Utils.log('Not currently recording', 'warn');
      return;
    }

    this.mediaRecorder.stop();
    this.cleanup();
  }

  async finalizeRecording(meta) {
    if (this.chunks.length === 0) {
      Utils.log('No recording data to process', 'warn');
      return;
    }

    try {
      // Create blob from chunks
      const blob = new Blob(this.chunks, { type: 'audio/webm' });
      const url = URL.createObjectURL(blob);

      const clipData = {
        clipId: this.currentClipId,
        url,
        blob,
        duration: 0, // Could calculate from chunks
        size: blob.size,
        timestamp: Date.now(),
        meta: meta || {}
      };

      // Save to external store
      await Store.save(`clips.${this.currentClipId}`, clipData);

      // Update UI
      this.addClipToUI(clipData);

      // Announce clip ready
      sendBus({
        type: 'rec.clip',
        clipId: this.currentClipId,
        url,
        meta: clipData.meta,
        size: blob.size
      });

      Utils.log(`Recording saved: ${this.currentClipId} (${blob.size} bytes)`);

    } catch (error) {
      Utils.log(`Failed to finalize recording: ${error.message}`, 'error');
      throw error;
    } finally {
      this.isRecording = false;
      this.currentClipId = null;
      this.chunks = [];
      this.updateUI('stopped');
    }
  }

  async addMarker(tag, runId = null) {
    const marker = {
      id: Utils.generateId(),
      tag,
      runId,
      timestamp: Date.now(),
      clipId: this.currentClipId
    };

    await Store.save(`marks.${marker.id}`, marker);

    Utils.log(`Marker added: ${tag} ${runId ? `(run:${runId})` : ''}`);

    sendBus({
      type: 'rec.marker',
      marker
    });

    return marker;
  }

  cleanup() {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    this.mediaRecorder = null;
    this.isRecording = false;
    this.updateUI('stopped');
  }

  updateUI(state) {
    const indicator = document.getElementById('rec-indicator');
    const startBtns = document.querySelectorAll('#rec-mic, #rec-screen');
    const stopBtn = document.getElementById('rec-stop');

    if (indicator) {
      indicator.className = state === 'recording' ? 'rec-dot recording' : 'rec-dot';
    }

    if (startBtns && stopBtn) {
      startBtns.forEach(btn => btn.disabled = state === 'recording');
      stopBtn.disabled = state !== 'recording';
    }
  }

  addClipToUI(clipData) {
    const container = document.getElementById('rec-clips');
    if (!container) return;

    const clipEl = document.createElement('div');
    clipEl.className = 'rec-clip';
    clipEl.innerHTML = `
      <audio controls src="${clipData.url}"></audio>
      <span>${new Date(clipData.timestamp).toLocaleTimeString()}</span>
      <span>${(clipData.size / 1024).toFixed(1)}KB</span>
      <button onclick="navigator.clipboard.writeText('${clipData.clipId}')">📋</button>
    `;

    container.insertBefore(clipEl, container.firstChild);
  }

  handleChunkTranscription(chunk) {
    // Placeholder for transcription service integration
    // You would send chunk to your transcription API here
    if (this.transcriptionService) {
      this.transcriptionService.transcribe(chunk)
        .then(text => {
          if (text.trim()) {
            sendBus({
              type: 'rec.transcript',
              clipId: this.currentClipId,
              text: text.trim(),
              ts: Date.now()
            });
          }
        })
        .catch(error => {
          Utils.log(`Transcription error: ${error.message}`, 'warn');
        });
    }
  }

  // Public API methods
  getStatus() {
    return {
      isRecording: this.isRecording,
      currentClipId: this.currentClipId,
      hasStream: !!this.mediaStream
    };
  }

  async getClips() {
    // Retrieve all clips from storage
    const clips = [];
    // This would need to iterate through stored clips
    return clips;
  }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = RecorderController;
} else {
  window.RecorderController = RecorderController;
}
