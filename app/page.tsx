"use client"

import { useRef, useState, useCallback } from "react"

const DEFAULT_PARAMS = {
  pitch: -4,
  formant: 1.3,
  noise: 0.04,
  window: 0.04,
  fftWarpStrength: 0.6, // 0–1, controls how aggressively frequency axis is scrambled
}

// ─── WAV encoder ────────────────────────────────────────────────────────────
function encodeWAV(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const buf = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buf)
  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++)
      view.setUint8(offset + i, s.charCodeAt(i))
  }
  writeStr(0, "RIFF")
  view.setUint32(4, 36 + samples.length * 2, true)
  writeStr(8, "WAVE")
  writeStr(12, "fmt ")
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeStr(36, "data")
  view.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return buf
}

// ─── Cooley-Tukey FFT (in-place, power-of-2) ────────────────────────────────
function fft(re: Float64Array, im: Float64Array, inverse = false) {
  const n = re.length
  // bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      ;[re[i], re[j]] = [re[j], re[i]]
      ;[im[i], im[j]] = [im[j], im[i]]
    }
  }
  // butterfly
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((2 * Math.PI) / len) * (inverse ? 1 : -1)
    const wRe = Math.cos(ang)
    const wIm = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let curRe = 1,
        curIm = 0
      for (let j = 0; j < len / 2; j++) {
        const uRe = re[i + j]
        const uIm = im[i + j]
        const vRe = re[i + j + len / 2] * curRe - im[i + j + len / 2] * curIm
        const vIm = re[i + j + len / 2] * curIm + im[i + j + len / 2] * curRe
        re[i + j] = uRe + vRe
        im[i + j] = uIm + vIm
        re[i + j + len / 2] = uRe - vRe
        im[i + j + len / 2] = uIm - vIm
        const newCurRe = curRe * wRe - curIm * wIm
        curIm = curRe * wIm + curIm * wRe
        curRe = newCurRe
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n
      im[i] /= n
    }
  }
}

// ─── FFT-based anonymization ─────────────────────────────────────────────────
// Step-by-step explanation:
// 1. Split audio into overlapping 2048-sample frames with Hann window
// 2. FFT each frame → get magnitude + phase spectrum
// 3. Apply non-linear random frequency warp to magnitude (destroys formant structure)
// 4. Randomize phase completely (destroys fine temporal structure)
// 5. Inverse FFT → back to audio samples
// 6. Overlap-add frames back together
//
// Random warp is generated fresh per call and never stored → irreversible
async function applyFFTDistortion(
  blob: Blob,
  warpStrength: number,
): Promise<Blob> {
  const arrayBuf = await blob.arrayBuffer()
  const audioCtx = new AudioContext()
  const decoded = await audioCtx.decodeAudioData(arrayBuf)
  await audioCtx.close()

  const sampleRate = decoded.sampleRate
  const origData = decoded.getChannelData(0)
  const n = origData.length

  const frameSize = 2048 // must be power of 2
  const hopSize = frameSize / 4 // 75% overlap for smooth reconstruction

  // ── Step 1: Build Hann window ─────────────────────────────────────────────
  const hann = new Float64Array(frameSize)
  for (let i = 0; i < frameSize; i++) {
    hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / frameSize))
  }

  // ── Step 2: Generate random frequency warp map (session-unique, discarded after) ─
  // This maps each output frequency bin to a source frequency bin
  // using a randomized non-linear curve. The curve is never stored.
  const numBins = frameSize / 2 + 1
  const warpMap = new Float64Array(numBins)

  // Generate random control points for the warp curve
  const numControlPoints = 8
  const controlPoints: number[] = []
  for (let i = 0; i < numControlPoints; i++) {
    // Each control point is a random deviation from linear mapping
    // strength controls how far it can deviate
    controlPoints.push(1 + (Math.random() * 2 - 1) * warpStrength * 0.4)
  }

  // Interpolate control points to build full warp map
  for (let bin = 0; bin < numBins; bin++) {
    const t = bin / (numBins - 1) // 0 to 1
    const cpIdx = t * (numControlPoints - 1)
    const cpLow = Math.floor(cpIdx)
    const cpHigh = Math.min(cpLow + 1, numControlPoints - 1)
    const cpFrac = cpIdx - cpLow
    const warpFactor =
      controlPoints[cpLow] * (1 - cpFrac) + controlPoints[cpHigh] * cpFrac
    // Map this output bin to a (possibly fractional) source bin
    warpMap[bin] = Math.max(0, Math.min(numBins - 1, bin * warpFactor))
  }

  // ── Step 3: Generate random phase offsets (session-unique, discarded after) ─
  // These are added to the phase of each bin. Since they're random and discarded,
  // no attacker can subtract them to recover the original phase.
  const phaseOffsets = new Float64Array(numBins)
  for (let i = 0; i < numBins; i++) {
    phaseOffsets[i] = Math.random() * 2 * Math.PI
  }

  // ── Step 4: Process frame by frame ───────────────────────────────────────
  const output = new Float64Array(n + frameSize)

  for (let pos = 0; pos + frameSize <= n; pos += hopSize) {
    // Extract frame and apply Hann window
    const re = new Float64Array(frameSize)
    const im = new Float64Array(frameSize)
    for (let i = 0; i < frameSize; i++) {
      re[i] = origData[pos + i] * hann[i]
    }

    // Forward FFT → frequency domain
    fft(re, im)

    // ── Step 5: Apply frequency warp to magnitude ─────────────────────────
    // Compute magnitude spectrum
    const mag = new Float64Array(numBins)
    for (let bin = 0; bin < numBins; bin++) {
      mag[bin] = Math.sqrt(re[bin] * re[bin] + im[bin] * im[bin])
    }

    // Build warped magnitude by reading from warpMap positions
    const warpedMag = new Float64Array(numBins)
    for (let bin = 0; bin < numBins; bin++) {
      const srcBin = warpMap[bin]
      const srcLow = Math.floor(srcBin)
      const srcHigh = Math.min(srcLow + 1, numBins - 1)
      const frac = srcBin - srcLow
      warpedMag[bin] = mag[srcLow] * (1 - frac) + mag[srcHigh] * frac
    }

    // ── Step 6: Apply random phase ────────────────────────────────────────
    // Reconstruct complex spectrum with warped magnitude + random phase
    const newRe = new Float64Array(frameSize)
    const newIm = new Float64Array(frameSize)

    for (let bin = 0; bin < numBins; bin++) {
      const phase = phaseOffsets[bin] // completely random, discarded after
      newRe[bin] = warpedMag[bin] * Math.cos(phase)
      newIm[bin] = warpedMag[bin] * Math.sin(phase)
    }

    // Mirror for negative frequencies (required for real signal reconstruction)
    for (let bin = 1; bin < numBins - 1; bin++) {
      newRe[frameSize - bin] = newRe[bin]
      newIm[frameSize - bin] = -newIm[bin]
    }

    // ── Step 7: Inverse FFT → back to time domain ─────────────────────────
    fft(newRe, newIm, true)

    // ── Step 8: Overlap-add into output ───────────────────────────────────
    for (let i = 0; i < frameSize; i++) {
      output[pos + i] += newRe[i] * hann[i]
    }
  }

  // ── Step 9: Normalize output ──────────────────────────────────────────────
  const outFloat = new Float32Array(n)
  let peak = 0
  for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(output[i]))
  for (let i = 0; i < n; i++)
    outFloat[i] = peak > 0 ? (output[i] / peak) * 0.85 : 0

  const wavBuf = encodeWAV(outFloat, sampleRate)
  return new Blob([wavBuf], { type: "audio/wav" })
}

// ─── Original OLA pipeline (kept for comparison) ────────────────────────────
async function applyDistortionPipeline(
  blob: Blob,
  params: typeof DEFAULT_PARAMS,
): Promise<Blob> {
  const arrayBuf = await blob.arrayBuffer()
  const audioCtx = new AudioContext()
  const decoded = await audioCtx.decodeAudioData(arrayBuf)
  await audioCtx.close()

  const sampleRate = decoded.sampleRate
  const origData = decoded.getChannelData(0)
  const n = origData.length

  const pitchRatio = Math.pow(2, params.pitch / 12)
  const formantRatio = params.formant
  const noiseLevel = params.noise
  const windowSamples = Math.floor(params.window * sampleRate)
  const hopSize = Math.floor(windowSamples / 2)

  const omega = new Float32Array(windowSamples)
  for (let i = 0; i < windowSamples; i++) {
    omega[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / windowSamples))
  }

  const output = new Float32Array(n)
  let readPos = 0
  let writePos = 0

  while (writePos + windowSamples < n) {
    const srcIdx = Math.floor(readPos)
    const frac = readPos - srcIdx
    for (let i = 0; i < windowSamples; i++) {
      const si = srcIdx + Math.floor(i * formantRatio)
      const v =
        si < n - 1 ? origData[si] * (1 - frac) + origData[si + 1] * frac : 0
      const noise = (Math.random() * 2 - 1) * noiseLevel
      output[writePos + i] += (v + noise) * omega[i]
    }
    readPos += hopSize / pitchRatio
    writePos += hopSize
  }

  const peak = output.reduce((m, v) => Math.max(m, Math.abs(v)), 0)
  if (peak > 0)
    for (let i = 0; i < n; i++) output[i] = (output[i] / peak) * 0.85

  const wavBuf = encodeWAV(output, sampleRate)
  return new Blob([wavBuf], { type: "audio/wav" })
}

type Status = "idle" | "recording" | "processing" | "ready" | "error"

export default function VoiceMaskPage() {
  const [params, setParams] = useState(DEFAULT_PARAMS)
  const [status, setStatus] = useState<Status>("idle")
  const [statusMsg, setStatusMsg] = useState(
    "waiting for microphone permission",
  )
  const [origUrl, setOrigUrl] = useState<string | null>(null)
  const [olaUrl, setOlaUrl] = useState<string | null>(null)
  const [fftUrl, setFftUrl] = useState<string | null>(null)
  const [olaBlob, setOlaBlob] = useState<Blob | null>(null)
  const [fftBlob, setFftBlob] = useState<Blob | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [whisperKey, setWhisperKey] = useState("")
  const [whisperResult, setWhisperResult] = useState("")
  const [whisperLoading, setWhisperLoading] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const audioCtxRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const animRef = useRef<number>(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const startTimeRef = useRef(0)
  const currentAudioRef = useRef<HTMLAudioElement | null>(null)

  const drawWave = useCallback(() => {
    const canvas = canvasRef.current
    const analyser = analyserRef.current
    if (!canvas || !analyser) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    canvas.width = canvas.offsetWidth * devicePixelRatio
    canvas.height = canvas.offsetHeight * devicePixelRatio
    const data = new Uint8Array(analyser.frequencyBinCount)
    const frame = () => {
      analyser.getByteTimeDomainData(data)
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.strokeStyle = "#7c6ff7"
      ctx.lineWidth = 1.5 * devicePixelRatio
      ctx.beginPath()
      const sw = canvas.width / data.length
      let x = 0
      for (let i = 0; i < data.length; i++) {
        const y = (data[i] / 128.0) * (canvas.height / 2)
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
        x += sw
      }
      ctx.stroke()
      animRef.current = requestAnimationFrame(frame)
    }
    frame()
  }, [])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state !== "inactive")
      mediaRecorderRef.current?.stop()
    setIsRecording(false)
    if (timerRef.current) clearInterval(timerRef.current)
    cancelAnimationFrame(animRef.current)
  }, [])

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const ac = new AudioContext({ sampleRate: 44100 })
      audioCtxRef.current = ac
      const analyser = ac.createAnalyser()
      analyser.fftSize = 512
      analyserRef.current = analyser
      ac.createMediaStreamSource(stream).connect(analyser)

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm"
      const mr = new MediaRecorder(stream, { mimeType })
      mediaRecorderRef.current = mr
      chunksRef.current = []

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      mr.onstop = async () => {
        const origBlob = new Blob(chunksRef.current, { type: mimeType })
        setOrigUrl(URL.createObjectURL(origBlob))
        setStatus("processing")
        setStatusMsg("running both pipelines…")
        try {
          // Run both pipelines in parallel
          const [ola, fft] = await Promise.all([
            applyDistortionPipeline(origBlob, params),
            applyFFTDistortion(origBlob, params.fftWarpStrength),
          ])
          setOlaBlob(ola)
          setOlaUrl(URL.createObjectURL(ola))
          setFftBlob(fft)
          setFftUrl(URL.createObjectURL(fft))
          setStatus("ready")
          setStatusMsg("both pipelines complete · compare results")
        } catch (e: any) {
          setStatus("error")
          setStatusMsg("processing failed: " + e.message)
        }
        stream.getTracks().forEach((t) => t.stop())
      }

      mr.start(100)
      setIsRecording(true)
      setStatus("recording")
      setStatusMsg("recording in progress")
      startTimeRef.current = Date.now()
      timerRef.current = setInterval(() => {
        setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000))
      }, 500)
      drawWave()
    } catch {
      setStatus("error")
      setStatusMsg("microphone permission denied")
    }
  }, [params, drawWave])

  const toggleRecord = useCallback(() => {
    if (!isRecording) startRecording()
    else stopRecording()
  }, [isRecording, startRecording, stopRecording])

  const playAudio = (url: string | null) => {
    if (!url) return
    if (currentAudioRef.current) {
      currentAudioRef.current.pause()
      currentAudioRef.current = null
      setIsPlaying(false)
    }
    const audio = new Audio(url)
    currentAudioRef.current = audio
    setIsPlaying(true)
    audio.onended = () => {
      setIsPlaying(false)
      currentAudioRef.current = null
    }
    audio.play()
  }

  const download = (blob: Blob | null, name: string) => {
    if (!blob) return
    const a = document.createElement("a")
    a.href = URL.createObjectURL(blob)
    a.download = name
    a.click()
  }

  const testWhisper = async (blob: Blob | null) => {
    if (!whisperKey || !blob) return
    setWhisperLoading(true)
    setWhisperResult("")
    try {
      const fd = new FormData()
      fd.append("file", blob, "audio.wav")
      fd.append("model", "whisper-1")
      fd.append("language", "de")
      const res = await fetch(
        "https://api.openai.com/v1/audio/transcriptions",
        {
          method: "POST",
          headers: { Authorization: "Bearer " + whisperKey },
          body: fd,
        },
      )
      const data = await res.json()
      setWhisperResult(data.text ?? "error: " + JSON.stringify(data.error))
    } catch (e: any) {
      setWhisperResult("fetch failed: " + e.message)
    }
    setWhisperLoading(false)
  }

  const fmt = (s: number) =>
    `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`

  return (
    <main className="min-h-screen bg-[#0a0a0f] text-[#e8e8f0] font-sans flex justify-center px-4 py-8">
      <div className="w-full max-w-2xl space-y-4">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-1">
            <div className="w-8 h-8 bg-[#7c6ff7] rounded-lg flex items-center justify-center">
              <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                <path
                  d="M3 9 Q6 4 9 9 Q12 14 15 9"
                  stroke="white"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                />
              </svg>
            </div>
            <span className="font-mono text-lg font-medium tracking-tight">
              Voice<span className="text-[#7c6ff7]">Mask</span>
            </span>
          </div>
          <p className="font-mono text-xs text-[#888899]">
            // client-side anonymization · voice identity destroyed ·
            intelligibility preserved
          </p>
        </div>

        {/* Recording card */}
        <div className="bg-[#121218] border border-white/10 rounded-xl p-6">
          <p className="font-mono text-[10px] uppercase tracking-widest text-[#888899] mb-4">
            Recording
          </p>
          <div className="flex flex-col items-center gap-4">
            <div className="w-full h-16 bg-[#1a1a24] border border-white/10 rounded-lg overflow-hidden">
              <canvas ref={canvasRef} className="w-full h-full" />
            </div>
            <button
              onClick={toggleRecord}
              className={`w-16 h-16 rounded-full border-2 flex items-center justify-center transition-all ${
                isRecording
                  ? "border-red-500 animate-pulse"
                  : "border-white/20 hover:border-[#7c6ff7]"
              }`}
            >
              <div
                className={`bg-red-500 transition-all ${
                  isRecording ? "w-5 h-5 rounded-sm" : "w-6 h-6 rounded-full"
                }`}
              />
            </button>
            <p
              className={`font-mono text-xs ${
                isRecording ? "text-red-400" : "text-[#888899]"
              }`}
            >
              {isRecording
                ? `recording… click to stop · ${fmt(elapsed)}`
                : "click to record"}
            </p>
          </div>
          <div className="flex items-center gap-2 mt-4 pt-4 border-t border-white/10 font-mono text-xs text-[#888899]">
            <div
              className={`w-2 h-2 rounded-full ${
                status === "ready"
                  ? "bg-[#5ad8a6]"
                  : status === "recording"
                  ? "bg-red-500 animate-pulse"
                  : status === "processing"
                  ? "bg-[#7c6ff7] animate-pulse"
                  : "bg-[#888899]"
              }`}
            />
            {statusMsg}
          </div>
        </div>

        {/* Results comparison card */}
        <div className="bg-[#121218] border border-white/10 rounded-xl p-6">
          <p className="font-mono text-[10px] uppercase tracking-widest text-[#888899] mb-4">
            Compare Results
          </p>

          {/* Original */}
          <div className="mb-4">
            <p className="font-mono text-[11px] text-[#888899] mb-2">
              original
            </p>
            <div className="flex gap-2">
              <button
                disabled={!origUrl || isPlaying}
                onClick={() => playAudio(origUrl)}
                className="flex-1 py-2 rounded-lg border border-white/20 font-mono text-xs hover:border-white/50 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
              >
                {isPlaying ? "▶ playing…" : "▶ play"}
              </button>
            </div>
          </div>

          {/* OLA distorted */}
          <div className="mb-4 p-3 bg-[#1a1a24] rounded-lg border border-white/10">
            <p className="font-mono text-[11px] text-[#888899] mb-1">
              pitch shift + formant + noise{" "}
              <span className="text-[#888899]">(v1)</span>
            </p>
            <p className="font-mono text-[10px] text-[#555566] mb-2">
              time-domain OLA · formant ratios preserved
            </p>
            <div className="flex gap-2">
              <button
                disabled={!olaUrl || isPlaying}
                onClick={() => playAudio(olaUrl)}
                className="flex-1 py-2 rounded-lg border border-[#7c6ff7]/40 text-[#7c6ff7] font-mono text-xs hover:bg-[#7c6ff7]/10 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
              >
                {isPlaying ? "▶ playing…" : "▶ play"}
              </button>
              <button
                disabled={!olaBlob}
                onClick={() => download(olaBlob, "voicemask_v1_ola.wav")}
                className="px-3 py-2 rounded-lg border border-white/20 font-mono text-xs hover:border-white/50 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
              >
                ↓
              </button>
            </div>
          </div>

          {/* FFT distorted */}
          <div className="p-3 bg-[#1a1a24] rounded-lg border border-[#5ad8a6]/20">
            <p className="font-mono text-[11px] text-[#5ad8a6] mb-1">
              FFT spectral anonymization{" "}
              <span className="text-[#888899]">(v2)</span>
            </p>
            <p className="font-mono text-[10px] text-[#555566] mb-2">
              random frequency warp + phase destruction · irreversible
            </p>
            <div className="flex gap-2">
              <button
                disabled={!fftUrl || isPlaying}
                onClick={() => playAudio(fftUrl)}
                className="flex-1 py-2 rounded-lg border border-[#5ad8a6]/40 text-[#5ad8a6] font-mono text-xs hover:bg-[#5ad8a6]/10 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
              >
                {isPlaying ? "▶ playing…" : "▶ play"}
              </button>
              <button
                disabled={!fftBlob}
                onClick={() => download(fftBlob, "voicemask_v2_fft.wav")}
                className="px-3 py-2 rounded-lg border border-white/20 font-mono text-xs hover:border-white/50 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
              >
                ↓
              </button>
            </div>
          </div>
        </div>

        {/* Params card */}
        <div className="bg-[#121218] border border-white/10 rounded-xl p-6">
          <p className="font-mono text-[10px] uppercase tracking-widest text-[#888899] mb-4">
            Pipeline Parameters
          </p>
          <p className="font-mono text-[10px] text-[#555566] mb-3">
            v1 — time domain
          </p>
          <div className="grid grid-cols-2 gap-4 mb-4">
            {[
              {
                key: "pitch",
                label: "pitch shift (st)",
                min: -8,
                max: -1,
                step: 1,
                display: (v: number) => (v < 0 ? "−" + Math.abs(v) : "+" + v),
              },
              {
                key: "formant",
                label: "formant ratio",
                min: 1.1,
                max: 1.8,
                step: 0.05,
                display: (v: number) => v.toFixed(2) + "×",
              },
              {
                key: "noise",
                label: "noise level",
                min: 0.01,
                max: 0.12,
                step: 0.005,
                display: (v: number) => v.toFixed(3),
              },
              {
                key: "window",
                label: "window size (s)",
                min: 0.02,
                max: 0.08,
                step: 0.005,
                display: (v: number) => v.toFixed(3),
              },
            ].map(({ key, label, min, max, step, display }) => (
              <div key={key} className="flex flex-col gap-1.5">
                <div className="flex justify-between font-mono text-[11px] text-[#888899]">
                  <span>{label}</span>
                  <span className="text-[#7c6ff7] font-medium">
                    {display(
                      params[key as keyof typeof DEFAULT_PARAMS] as number,
                    )}
                  </span>
                </div>
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={step}
                  value={params[key as keyof typeof DEFAULT_PARAMS]}
                  onChange={(e) =>
                    setParams((p) => ({
                      ...p,
                      [key]: parseFloat(e.target.value),
                    }))
                  }
                  className="w-full accent-[#7c6ff7]"
                />
              </div>
            ))}
          </div>
          <p className="font-mono text-[10px] text-[#555566] mb-3">
            v2 — frequency domain
          </p>
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between font-mono text-[11px] text-[#888899]">
              <span>warp strength</span>
              <span className="text-[#5ad8a6] font-medium">
                {params.fftWarpStrength.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0.1}
              max={1.0}
              step={0.05}
              value={params.fftWarpStrength}
              onChange={(e) =>
                setParams((p) => ({
                  ...p,
                  fftWarpStrength: parseFloat(e.target.value),
                }))
              }
              className="w-full accent-[#5ad8a6]"
            />
            <p className="font-mono text-[10px] text-[#555566] mt-1">
              actual warp map is randomized per recording — this only controls
              intensity
            </p>
          </div>
        </div>

        {/* Whisper card */}
        <div className="bg-[#121218] border border-white/10 rounded-xl p-6">
          <p className="font-mono text-[10px] uppercase tracking-widest text-[#888899] mb-4">
            Whisper Transcription Test
          </p>
          <input
            type="text"
            value={whisperKey}
            onChange={(e) => setWhisperKey(e.target.value)}
            placeholder="sk-... (OpenAI API key, not stored)"
            className="w-full bg-[#1a1a24] border border-white/10 rounded-lg px-3 py-2 font-mono text-xs text-[#e8e8f0] placeholder:text-[#888899] outline-none focus:border-[#7c6ff7] mb-3 transition-colors"
          />
          <div className="grid grid-cols-2 gap-2">
            <button
              disabled={!whisperKey || !olaBlob || whisperLoading}
              onClick={() => testWhisper(olaBlob)}
              className="py-2.5 rounded-lg border border-[#7c6ff7] bg-[#7c6ff7]/10 text-[#7c6ff7] font-mono text-xs hover:bg-[#7c6ff7]/20 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
            >
              {whisperLoading ? "sending…" : "test v1 (OLA)"}
            </button>
            <button
              disabled={!whisperKey || !fftBlob || whisperLoading}
              onClick={() => testWhisper(fftBlob)}
              className="py-2.5 rounded-lg border border-[#5ad8a6] bg-[#5ad8a6]/10 text-[#5ad8a6] font-mono text-xs hover:bg-[#5ad8a6]/20 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
            >
              {whisperLoading ? "sending…" : "test v2 (FFT)"}
            </button>
          </div>
          {whisperResult && (
            <div className="mt-3 p-3 bg-[#1a1a24] rounded-lg font-mono text-sm text-[#5ad8a6] italic border-l-2 border-[#5ad8a6]">
              &ldquo;{whisperResult}&rdquo;
            </div>
          )}
        </div>
      </div>
    </main>
  )
}
