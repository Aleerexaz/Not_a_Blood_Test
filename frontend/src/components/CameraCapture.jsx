import React, { useEffect, useRef, useState } from 'react'

export default function CameraCapture({ onCapture, disabled }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [stream, setStream] = useState(null)
  const [active, setActive] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  async function startCamera() {
    setError(null)
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } },
        audio: false,
      })
      setStream(s)
      if (videoRef.current) {
        videoRef.current.srcObject = s
        await videoRef.current.play()
      }
      setActive(true)
    } catch (e) {
      setError('Camera access denied or unavailable')
    }
  }

  function stopCamera() {
    if (stream) {
      for (const track of stream.getTracks()) track.stop()
      setStream(null)
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setActive(false)
  }

  async function capture() {
    if (!videoRef.current) return
    const video = videoRef.current
    const canvas = canvasRef.current
    const w = 512
    const h = Math.round((video.videoHeight / video.videoWidth) * w)
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, w, h)

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.95))
    if (blob) {
      const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' })
      onCapture?.(file)
    }
  }

  return (
    <div className="camera">
      {!active ? (
        <button onClick={startCamera} disabled={disabled}>Use Camera</button>
      ) : (
        <div className="camera-active">
          <video ref={videoRef} playsInline className="cam-video" />
          <div className="cam-controls">
            <button onClick={capture} disabled={disabled}>Capture</button>
            <button onClick={stopCamera} disabled={disabled}>Stop</button>
          </div>
        </div>
      )}
      <canvas ref={canvasRef} className="cam-canvas" style={{ display: 'none' }} />
      {error && <p className="error">{error}</p>}
    </div>
  )
}
