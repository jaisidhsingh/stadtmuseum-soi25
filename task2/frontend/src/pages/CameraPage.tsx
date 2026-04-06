import * as React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Camera, RefreshCcw, ArrowRight, ArrowLeft } from "lucide-react";

const CameraPage = () => {
  const navigate = useNavigate();
  const videoRef = React.useRef<HTMLVideoElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = React.useState<MediaStream | null>(null);
  const [capturedImage, setCapturedImage] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  const startCamera = React.useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 1280, height: 720 },
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setError(null);
    } catch (err) {
      setError("Unable to access camera. Please grant camera permissions.");
      console.error("Camera error:", err);
    }
  }, []);

  const stopCamera = React.useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
  }, [stream]);

  React.useEffect(() => {
    startCamera();
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL("image/png");
        setCapturedImage(imageData);
        stopCamera();
      }
    }
  };

  const retakePhoto = () => {
    setCapturedImage(null);
    startCamera();
  };

  const proceedToSelection = () => {
    if (capturedImage) {
      sessionStorage.setItem("capturedImage", capturedImage);
      navigate("/silhouette");
    }
  };

  return (
    <div className="h-screen bg-background flex flex-col overflow-hidden">
      {/* ── Top bar ──────────────────────────────────────────── */}
      <div className="flex-shrink-0 px-8 pt-5 pb-4 flex items-start justify-between">
        <div>
          <p
            className="text-xs font-semibold uppercase tracking-widest mb-1"
            style={{ color: "hsl(var(--film-blue))", opacity: 0.7 }}
          >
            Step 1 of 4
          </p>
          <h1 className="text-2xl font-bold text-foreground">
            {capturedImage ? "Looking good!" : "Take Your Photo"}
          </h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            {capturedImage
              ? "Happy with your photo? Continue — or retake if you'd like."
              : "Stand facing the camera, then tap Capture when you're ready."}
          </p>
        </div>
        <button
          onClick={() => navigate("/")}
          className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors mt-1 flex-shrink-0 ml-6"
        >
          <ArrowLeft className="w-4 h-4" />
          Start Over
        </button>
      </div>

      {/* ── Video / Captured image ────────────────────────────── */}
      <div className="flex-1 flex items-center justify-center w-full px-8 min-h-0">
        {error ? (
          <div className="flex flex-col items-center gap-6 text-center">
            <p className="text-destructive text-xl">{error}</p>
            <Button
              size="lg"
              onClick={startCamera}
              className="text-lg px-10 py-7"
            >
              Try Again
            </Button>
          </div>
        ) : capturedImage ? (
          <img
            src={capturedImage}
            alt="Captured"
            className="max-h-full w-auto rounded-2xl border border-border shadow-2xl"
          />
        ) : (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="max-h-full w-auto rounded-2xl border border-border shadow-2xl bg-muted"
          />
        )}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {/* ── Action buttons ────────────────────────────────────── */}
      <div className="flex-shrink-0 flex items-center justify-center gap-6 py-8">
        {capturedImage ? (
          <>
            <Button
              size="lg"
              variant="outline"
              onClick={retakePhoto}
              className="text-lg px-8 py-7 rounded-xl border-border gap-2"
            >
              <RefreshCcw className="w-5 h-5" />
              Retake
            </Button>
            <Button
              size="lg"
              onClick={proceedToSelection}
              className="text-lg px-12 py-7 rounded-xl font-semibold gold-glow gap-2"
            >
              Continue
              <ArrowRight className="w-5 h-5" />
            </Button>
          </>
        ) : (
          <Button
            size="lg"
            onClick={capturePhoto}
            disabled={!!error}
            className="text-xl px-16 py-8 rounded-full font-semibold gold-glow gap-3"
          >
            <Camera className="w-6 h-6" />
            Capture
          </Button>
        )}
      </div>
    </div>
  );
};

export default CameraPage;
