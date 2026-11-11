/**
 * Voice-Based Authentication System - Audio Recorder & WAV Encoder
 * Captures microphone audio using Web Audio API, encodes pure 16-bit PCM WAV (16kHz mono),
 * and provides live visualizer rendering and API communication.
 */

class VoiceRecorder {
  constructor(options = {}) {
    this.targetSampleRate = options.sampleRate || 16000;
    this.onStateChange = options.onStateChange || null;
    this.onVisualizerTick = options.onVisualizerTick || null;
    this.audioContext = null;
    this.mediaStream = null;
    this.sourceNode = null;
    this.processorNode = null;
    this.analyserNode = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.recordedBlob = null;
    this.startTime = null;
    this.timerInterval = null;
  }

  /**
   * Request microphone permission and initialize Web Audio context
   */
  async init() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Your browser does not support microphone audio capture.");
    }

    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    this.audioContext = new AudioContextClass({ sampleRate: this.targetSampleRate });

    // Handle suspended audio context (browser autoplay policies)
    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }

    this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
    this.analyserNode = this.audioContext.createAnalyser();
    this.analyserNode.fftSize = 256;

    // Buffer size 4096, 1 input channel, 1 output channel
    const bufferSize = 4096;
    if (this.audioContext.createScriptProcessor) {
      this.processorNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
    } else {
      this.processorNode = this.audioContext.createJavaScriptNode(bufferSize, 1, 1);
    }

    this.sourceNode.connect(this.analyserNode);
    this.analyserNode.connect(this.processorNode);
    this.processorNode.connect(this.audioContext.destination);

    this.processorNode.onaudioprocess = (e) => {
      if (!this.isRecording) return;
      const inputBuffer = e.inputBuffer.getChannelData(0);
      // Clone float32 buffer
      this.audioChunks.push(new Float32Array(inputBuffer));
    };

    return true;
  }

  /**
   * Start recording audio
   */
  async start() {
    if (!this.audioContext || this.audioContext.state === "closed") {
      await this.init();
    } else if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }

    this.audioChunks = [];
    this.recordedBlob = null;
    this.isRecording = true;
    this.startTime = Date.now();

    if (this.onStateChange) this.onStateChange("recording");
    this._startVisualizerLoop();
  }

  /**
   * Stop recording audio and build WAV Blob
   */
  async stop() {
    if (!this.isRecording) return null;
    this.isRecording = false;

    if (this.onStateChange) this.onStateChange("stopped");

    // Merge audio chunks into a single Float32Array
    const totalLength = this.audioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.audioChunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }

    // Downsample or interpolate to 16000Hz if context sample rate differs
    const actualSampleRate = this.audioContext.sampleRate;
    const resampled = this._resampleAudio(merged, actualSampleRate, this.targetSampleRate);

    // Encode to 16-bit PCM WAV Blob
    this.recordedBlob = this._encodeWav(resampled, this.targetSampleRate);

    // Stop microphone tracks to release hardware
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }

    return this.recordedBlob;
  }

  /**
   * Resample float32 buffer from sourceRate to targetRate using linear interpolation
   */
  _resampleAudio(buffer, sourceRate, targetRate) {
    if (sourceRate === targetRate || targetRate <= 0) {
      return buffer;
    }
    const ratio = sourceRate / targetRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;

    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
      let accum = 0;
      let count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
        accum += buffer[i];
        count++;
      }
      result[offsetResult] = count > 0 ? accum / count : 0;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  }

  /**
   * Encode 16-bit PCM WAV audio file with RIFF header
   */
  _encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // RIFF identifier
    this._writeString(view, 0, "RIFF");
    // File length minus 8
    view.setUint32(4, 36 + samples.length * 2, true);
    // WAVE identifier
    this._writeString(view, 8, "WAVE");
    // FMT sub-chunk
    this._writeString(view, 12, "fmt ");
    // Sub-chunk length (16 for PCM)
    view.setUint32(16, 16, true);
    // Audio format (1 = PCM)
    view.setUint16(20, 1, true);
    // Number of channels (1 = Mono)
    view.setUint16(22, 1, true);
    // Sample rate
    view.setUint32(24, sampleRate, true);
    // Byte rate (sampleRate * channels * bytesPerSample)
    view.setUint32(28, sampleRate * 2, true);
    // Block align (channels * bytesPerSample)
    view.setUint16(32, 2, true);
    // Bits per sample
    view.setUint16(34, 16, true);
    // Data sub-chunk
    this._writeString(view, 36, "data");
    // Data size
    view.setUint32(40, samples.length * 2, true);

    // Write 16-bit PCM samples
    let index = 44;
    for (let i = 0; i < samples.length; i++) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      // Convert float [-1.0, 1.0] to signed 16-bit integer [-32768, 32767]
      s = s < 0 ? s * 0x8000 : s * 0x7fff;
      view.setInt16(index, s, true);
      index += 2;
    }

    return new Blob([buffer], { type: "audio/wav" });
  }

  _writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  _startVisualizerLoop() {
    const update = () => {
      if (!this.isRecording) return;
      if (this.analyserNode && this.onVisualizerTick) {
        const dataArray = new Uint8Array(this.analyserNode.frequencyBinCount);
        this.analyserNode.getByteFrequencyData(dataArray);
        this.onVisualizerTick(dataArray);
      }
      requestAnimationFrame(update);
    };
    requestAnimationFrame(update);
  }
}

/**
 * Voice API Client methods
 */
const VoiceAuthAPI = {
  /**
   * Submit audio blob for voice enrollment
   */
  async enroll(audioBlob) {
    const formData = new FormData();
    formData.append("audio", audioBlob, "enrollment.wav");

    const response = await fetch("/voice/enroll", {
      method: "POST",
      body: formData,
    });
    return await response.json();
  },

  /**
   * Submit audio blob for voice verification
   */
  async verify(audioBlob, username = null) {
    const formData = new FormData();
    formData.append("audio", audioBlob, "verification.wav");
    if (username) {
      formData.append("username", username);
    }

    const response = await fetch("/voice/verify", {
      method: "POST",
      body: formData,
    });
    return await response.json();
  },

  /**
   * Reset voice enrollment profile
   */
  async reset() {
    const response = await fetch("/voice/reset", {
      method: "POST",
    });
    return await response.json();
  },

  /**
   * Check voice status
   */
  async checkStatus(username = null) {
    const url = username
      ? `/voice/status?username=${encodeURIComponent(username)}`
      : `/voice/status`;
    const response = await fetch(url);
    return await response.json();
  },
};

// Export globals for browser scripts
window.VoiceRecorder = VoiceRecorder;
window.VoiceAuthAPI = VoiceAuthAPI;

