import * as React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { t } from "@/lib/localization";
import { Camera, RefreshCcw, ArrowRight, LogOut } from "lucide-react";

const COUNTDOWN_SECONDS = 5;

const CameraPage = () => {
  const navigate = useNavigate();
  const videoRef = React.useRef<HTMLVideoElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = React.useState<MediaStream | null>(null);
  const [capturedImage, setCapturedImage] = React.useState<string | null>(null);
  const [countdownSeconds, setCountdownSeconds] = React.useState<number | null>(
    null,
  );
  const [error, setError] = React.useState<string | null>(null);

  const startCamera = React.useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 1920, height: 1080 },
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setError(null);
    } catch (err) {
      setError(
        t(
          "Unable to access camera. Please grant camera permissions.",
          "Kein Kamerazugriff. Bitte Kamera-Berechtigung erteilen.",
        ),
      );
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

  const capturePhoto = React.useCallback(() => {
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
  }, [stopCamera]);

  const startCountdown = () => {
    if (error || capturedImage || countdownSeconds !== null) {
      return;
    }
    setCountdownSeconds(COUNTDOWN_SECONDS);
  };

  React.useEffect(() => {
    if (countdownSeconds === null) {
      return;
    }

    if (countdownSeconds === 0) {
      setCountdownSeconds(null);
      capturePhoto();
      return;
    }

    const timeoutId = window.setTimeout(() => {
      setCountdownSeconds((previousValue) =>
        previousValue === null ? null : previousValue - 1,
      );
    }, 1000);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [capturePhoto, countdownSeconds]);

  const retakePhoto = () => {
    setCountdownSeconds(null);
    setCapturedImage(null);
    startCamera();
  };

  const proceedToSelection = () => {
    if (capturedImage) {
      sessionStorage.setItem("capturedImage", capturedImage);
      navigate("/silhouette");
    }
  };

  const handleExitSession = async () => {
    try {
      await fetch("http://localhost:8000/clear", { method: "POST" });
    } catch (error) {
      console.error("Failed to clear backend session", error);
    }

    sessionStorage.removeItem("selectedBackgroundIds");
    sessionStorage.removeItem("selectedBackgrounds");
    sessionStorage.removeItem("preComposites");
    sessionStorage.removeItem("silhouetteData");
    sessionStorage.removeItem("capturedImage");
    sessionStorage.removeItem("silhouetteStyles");
    navigate("/");
  };

  return (
    <div className="h-full min-h-0 exhibit-shell flex flex-col overflow-hidden overflow-x-hidden">
      {/* ── Top bar ──────────────────────────────────────────── */}
      <div className="flex-shrink-0 px-8 pt-5 pb-4 flex items-start justify-between">
        <div>
          <p
            className="text-xs font-semibold uppercase tracking-widest mb-1"
            style={{ color: "hsl(var(--film-blue))", opacity: 0.7 }}
          >
            {t("Step 1 of 4", "Schritt 1 von 4")}
          </p>
          <h1 className="text-2xl font-bold text-foreground">
            {capturedImage
              ? t("Looking good!", "Sieht gut aus!")
              : t("Take Your Photo", "Foto aufnehmen")}
          </h1>
          <p className="mt-0.5 text-sm text-muted-foreground md:text-base">
            {capturedImage
              ? t(
                  "Happy with your photo? Continue or retake if you want.",
                  "Zufrieden mit dem Foto? Weiter oder erneut aufnehmen.",
                )
              : t(
                  "Stand in front of the camera, then tap Capture.",
                  "Stelle dich vor die Kamera und tippe dann auf Aufnehmen.",
                )}
          </p>
        </div>
        <Button
          variant="outline"
          size="xl"
          onClick={handleExitSession}
          className="ml-6 flex-shrink-0 border-film-red/40 bg-white/80 text-film-red hover:bg-film-red/10 hover:text-film-red"
        >
          <LogOut className="w-5 h-5" />
          {t("Exit Session", "Sitzung beenden")}
        </Button>
      </div>

      <div
        className={`mx-8 mb-3 min-h-[122px] rounded-2xl border-2 p-4 md:p-5 ${capturedImage ? "border-film-green/35 bg-film-green/10" : "border-film-blue/35 bg-film-blue/10"}`}
      >
        <p
          className={`text-lg font-bold md:text-2xl ${capturedImage ? "text-film-green" : "text-film-blue"}`}
        >
          {capturedImage
            ? t("Photo captured.", "Foto aufgenommen.")
            : t(
                "Important: Take a full-body photo.",
                "Wichtig: Bitte ein Ganzkoerperfoto aufnehmen.",
              )}
        </p>
        <p className="mt-1 text-sm font-medium text-foreground/90 md:text-base">
          {capturedImage
            ? t(
                "If framing looks off, tap Retake before continuing.",
                "Falls der Bildausschnitt nicht passt, tippe vor dem Fortfahren auf Erneut aufnehmen.",
              )
            : t(
                "Step back until your full body is visible from head to feet.",
                "Bitte so weit zurueckgehen, bis der ganze Koerper von Kopf bis Fuss sichtbar ist.",
              )}
        </p>
        <p className="mt-2 inline-flex rounded-full border border-film-green/30 bg-film-green/10 px-3 py-1 text-sm font-semibold text-film-green md:text-base">
          {t(
            "You can retake your photo anytime.",
            "Du kannst dein Foto jederzeit erneut aufnehmen.",
          )}
        </p>
      </div>

      {/* ── Video / Captured image ────────────────────────────── */}
      <div className="flex-1 flex items-center justify-center w-full px-8 min-h-0">
        {error ? (
          <div className="flex flex-col items-center gap-6 text-center">
            <p className="text-destructive text-xl">{error}</p>
            <Button
              size="xl"
              onClick={startCamera}
              className="px-10 py-4 text-base md:text-lg"
            >
              {t("Try Again", "Erneut versuchen")}
            </Button>
          </div>
        ) : capturedImage ? (
          <img
            src={capturedImage}
            alt={t("Captured image", "Aufgenommenes Bild")}
            className="max-h-[calc(60vh)] w-auto rounded-2xl border border-border shadow-2xl"
          />
        ) : (
          <div className="relative max-h-full w-auto">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="max-h-[calc(60vh)] w-auto rounded-2xl border border-border shadow-2xl bg-muted"
            />
            {countdownSeconds !== null ? (
              <div className="absolute inset-0 flex items-center justify-center rounded-2xl bg-black/30">
                <p className="text-8xl font-bold text-white drop-shadow-lg">
                  {countdownSeconds}
                </p>
              </div>
            ) : null}
          </div>
        )}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {/* ── Action buttons ────────────────────────────────────── */}
      <div className="flex-shrink-0 flex items-center justify-center gap-6 py-8">
        {capturedImage ? (
          <>
            <Button
              size="xl"
              variant="outline"
              onClick={retakePhoto}
              className="px-8 py-4 text-base md:text-lg rounded-xl border-border gap-2"
            >
              <RefreshCcw className="w-5 h-5" />
              {t("Retake", "Erneut aufnehmen")}
            </Button>
            <Button
              size="xl"
              onClick={proceedToSelection}
              className="px-12 py-4 text-base text-white md:text-lg rounded-xl font-semibold cta-step-2 gap-2"
            >
              {t("Continue", "Weiter")}
              <ArrowRight className="w-5 h-5" />
            </Button>
          </>
        ) : (
          <Button
            size="xl"
            onClick={startCountdown}
            disabled={!!error || countdownSeconds !== null}
            className="rounded-full px-14 py-4 text-base text-white font-semibold md:text-lg cta-step-1 gap-3"
          >
            <Camera className="w-6 h-6" />
            {countdownSeconds !== null
              ? t(
                  `Capturing in ${countdownSeconds}...`,
                  `Aufnahme in ${countdownSeconds}...`,
                )
              : t("Capture", "Aufnehmen")}
          </Button>
        )}
      </div>
    </div>
  );
};

export default CameraPage;
