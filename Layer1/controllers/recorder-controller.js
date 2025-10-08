/**
 * Enhanced Recorder Controller - Multi-format stream handling
 * Features:
 * - Audio/video recording with MediaRecorder
 * - MIME type auto-detection
 * - Multiple recording sessions
 * - Proper stream cleanup
 */

export class RecorderController {
    constructor(options = {}) {
        this.options = {
            video: true,
            audio: true,
            maxDuration: 120000, // 2 minutes
            ...options
        };

        this.recorders = new Map(); // sessionId -> {recorder, stream, chunks, startTime}
        this.constraints = {
            video: this.options.video,
            audio: this.options.audio
        };

        this.bindEvents();
    }

    bindEvents() {
        this.bus.onBus(async (msg) => {
            switch (msg.type) {
                case 'rec.start':
                    await this.handleStartRecording(msg);
                    break;
                case 'rec.stop':
                    await this.handleStopRecording(msg);
                    break;
                case 'rec.pause':
                    await this.handlePauseRecording(msg);
                    break;
                case 'rec.resume':
                    await this.handleResumeRecording(msg);
                    break;
                case 'rec.status':
                    await this.handleStatus(msg);
                    break;
            }
        });
    }

    async handleStartRecording(msg) {
        const sessionId = msg.sessionId || this.generateId();

        try {
            // Get user media
            const stream = await navigator.mediaDevices.getUserMedia(this.constraints);

            // Detect best MIME type
            const mimeType = this.detectMimeType();
            console.log(`Starting recording with MIME: ${mimeType}`);

            // Create recorder
            const recorder = new MediaRecorder(stream, {
                mimeType,
                bitsPerSecond: msg.bitrate || 1000000
            });

            const session = {
                id: sessionId,
                recorder,
                stream,
                chunks: [],
                startTime: Date.now(),
                mimeType,
                state: 'recording'
            };

            // Setup event handlers
            recorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    session.chunks.push(event.data);
                }
            };

            recorder.onstop = () => {
                this.handleRecordingComplete(sessionId);
            };

            recorder.onerror = (error) => {
                console.error('Recording error:', error);
                this.bus.sendBus({
                    type: 'rec.error',
                    sessionId,
                    error: error.message
                });
            };

            // Auto-stop timer
            const maxTimer = setTimeout(() => {
                if (session.state === 'recording') {
                    this.stopRecording(sessionId);
                }
            }, this.options.maxDuration);

            session.maxTimer = maxTimer;

            // Start recording
            recorder.start(1000); // Collect data every second
            this.recorders.set(sessionId, session);

            this.bus.sendBus({
                type: 'rec.started',
                sessionId,
                mimeType,
                constraints: this.constraints
            });

        } catch (error) {
            console.error('Failed to start recording:', error);
            this.bus.sendBus({
                type: 'rec.error',
                sessionId,
                error: error.message
            });
        }
    }

    async handleStopRecording(msg) {
        const sessionId = msg.sessionId;
        this.stopRecording(sessionId);
    }

    stopRecording(sessionId) {
        const session = this.recorders.get(sessionId);
        if (!session) {
            console.warn(`Recording session not found: ${sessionId}`);
            return;
        }

        try {
            if (session.recorder.state !== 'inactive') {
                session.recorder.stop();
            }
            session.state = 'stopped';

            // Clear timer
            if (session.maxTimer) {
                clearTimeout(session.maxTimer);
            }

            // Stop all tracks
            session.stream.getTracks().forEach(track => {
                track.stop();
            });

        } catch (error) {
            console.error('Error stopping recording:', error);
        }
    }

    handleRecordingComplete(sessionId) {
        const session = this.recorders.get(sessionId);
        if (!session) return;

        try {
            // Create blob from chunks
            const blob = new Blob(session.chunks, {
                type: session.mimeType
            });

            const duration = Date.now() - session.startTime;
            const url = URL.createObjectURL(blob);

            this.bus.sendBus({
                type: 'rec.complete',
                sessionId,
                blob,
                url,
                size: blob.size,
                duration,
                mimeType: session.mimeType,
                chunksCount: session.chunks.length
            });

            // Cleanup
            this.recorders.delete(sessionId);

        } catch (error) {
            console.error('Error processing recording:', error);
            this.bus.sendBus({
                type: 'rec.error',
                sessionId,
                error: error.message
            });
        }
    }

    async handlePauseRecording(msg) {
        const session = this.recorders.get(msg.sessionId);
        if (session && session.recorder.state === 'recording') {
            session.recorder.pause();
            session.state = 'paused';

            this.bus.sendBus({
                type: 'rec.paused',
                sessionId: msg.sessionId
            });
        }
    }

    async handleResumeRecording(msg) {
        const session = this.recorders.get(msg.sessionId);
        if (session && session.recorder.state === 'paused') {
            session.recorder.resume();
            session.state = 'recording';

            this.bus.sendBus({
                type: 'rec.resumed',
                sessionId: msg.sessionId
            });
        }
    }

    detectMimeType() {
        // Try video formats first
        const videoTypes = [
            'video/webm;codecs=vp9',
            'video/webm;codecs=vp8',
            'video/webm',
            'video/mp4'
        ];

        const audioTypes = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4',
            'audio/ogg;codecs=opus'
        ];

        if (this.constraints.video) {
            for (const type of videoTypes) {
                if (MediaRecorder.isTypeSupported(type)) {
                    return type;
                }
            }
        }

        if (this.constraints.audio) {
            for (const type of audioTypes) {
                if (MediaRecorder.isTypeSupported(type)) {
                    return type;
                }
            }
        }

        return ''; // Let browser choose
    }

    async handleStatus(msg) {
        const sessions = {};
        for (const [id, session] of this.recorders) {
            sessions[id] = {
                state: session.state,
                duration: Date.now() - session.startTime,
                chunks: session.chunks.length,
                mimeType: session.mimeType
            };
        }

        this.bus.sendBus({
            type: 'rec.status',
            sessions,
            activeSessions: this.recorders.size
        });
    }

    generateId() {
        return `rec_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    }

    // Helper method to download recording
    downloadRecording(sessionId, filename) {
        const session = this.recorders.get(sessionId);
        if (!session || session.chunks.length === 0) return;

        const blob = new Blob(session.chunks, { type: session.mimeType });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = filename || `recording-${sessionId}.webm`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    // Clean up all sessions
    cleanup() {
        for (const sessionId of this.recorders.keys()) {
            this.stopRecording(sessionId);
        }
        this.recorders.clear();
    }
}
