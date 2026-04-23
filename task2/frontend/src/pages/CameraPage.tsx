import * as React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { EXHIBIT_STEP_ACCENTS } from "@/lib/exhibitFlow";
import { cn } from "@/lib/utils";
import { t } from "@/lib/localization";
import { Camera, RefreshCcw, ArrowRight, LogOut } from "lucide-react";

const COUNTDOWN_SECONDS = 10;

const CAMERA_LEFT_INSTRUCTIONS: { en: string; de: string }[] = [
  {
    en: "Check if the camera is working",
    de: "Pruefe, ob die Kamera funktioniert.",
  },
  {
    en: "Make sure you are standing close to the white wall",
    de: "Stelle dich nah an die weisse Wand.",
  },
  {
    en: "Ensure that your entire body is inside the camera frame",
    de: "Achte darauf, dass dein ganzer Koerper im Kamerabild ist.",
  },
  {
    en: "Click on the CAPTURE button and strike a pose",
    de: "Tippe auf AUFNEHMEN und nimm eine Pose ein.",
  },
  { en: "Wait 10 seconds", de: "Warte 10 Sekunden." },
  {
    en: "Check if you like your pose",
    de: "Pruefe, ob dir deine Pose gefaellt.",
  },
  {
    en: "Move on to the next step or retake your image.",
    de: "Weiter zum naechsten Schritt oder erneut aufnehmen.",
  },
];

/** Match Start page START: green + outlined halo + size */
const captureButtonClassName =
  "cta-step-2 cta-start-outlined w-full min-h-14 max-w-2xl rounded-2xl py-3 text-xl font-bold uppercase tracking-[0.18em] text-white md:min-h-16 md:py-4 md:text-3xl";

/** Start-page step 1 look: light blue tint + same frame as center `exhibit-panel`; min-h-0/overflow = no page scroll */
const sidePanelFrameClass =
  "exhibit-panel-edge flex h-full min-h-0 min-w-0 max-h-full flex-col justify-center overflow-hidden rounded-2xl p-2 sm:p-3 md:p-4 max-lg:max-h-[30vh] lg:max-h-none";

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
        video: { facingMode: "user", width: 1080, height: 1920 },
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setError(null);
    } catch (err) {
      setError(
        t(
          "Unable to access the camera, please grant camera permissions.",
          "Kein Kamerazugriff — bitte Kamera-Berechtigung erteilen.",
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
    } catch (err) {
      console.error("Failed to clear backend session", err);
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
    <div className="exhibit-shell flex h-full min-h-0 w-full min-w-0 flex-col overflow-hidden overflow-x-hidden">
      {/* Same horizontal padding + 3 column template as the main grid below: title in center column, EXIT flush to end of right column = right blue panel. */}
      <div className="w-full flex-shrink-0 px-4 pt-2 pb-2 sm:pt-3 sm:pb-3 md:px-6 md:pt-4 md:pb-4 lg:pt-5 lg:pb-5">
        <div className="grid w-full min-w-0 grid-cols-[minmax(0,1fr)_minmax(0,1.85fr)_minmax(0,1.65fr)] items-center gap-2 sm:gap-3 md:gap-4 lg:gap-5">
          <div className="min-w-0" aria-hidden />
          <p className="exhibit-title min-w-0 text-center text-2xl font-bold uppercase leading-tight tracking-wider text-film-black md:text-3xl">
            {t("Step 1 of 4", "Schritt 1 von 4")}
          </p>
          <div className="flex min-w-0 items-center justify-end">
            <Button
              variant="outline"
              size="xl"
              onClick={handleExitSession}
              className="flex-shrink-0 border-film-red/40 bg-white/80 text-film-red hover:bg-film-red/10 hover:text-film-red"
            >
              <LogOut className="h-5 w-5" />
              {t("EXIT", "ABBRECHEN")}
            </Button>
          </div>
        </div>
      </div>

      <div className="min-h-0 w-full min-w-0 flex-1 overflow-hidden px-4 pb-3 pt-0 md:px-6 md:pb-4">
        <div
          className="grid h-full min-h-0 w-full min-w-0 items-stretch gap-2 sm:gap-3 md:gap-4 max-lg:grid-cols-1 max-lg:grid-rows-[minmax(0,auto)_minmax(0,1fr)_minmax(0,auto)] lg:grid-cols-[minmax(0,1fr)_minmax(0,1.85fr)_minmax(0,1.65fr)] lg:grid-rows-1 lg:gap-5"
        >
          <aside className="flex min-h-0 min-w-0 flex-col overflow-hidden lg:h-full">
            <div
              className={cn(
                sidePanelFrameClass,
                "!justify-start",
                "min-h-0 pl-0.5", /* tiny extra so list markers (inside) never clip at rounded edge */
                EXHIBIT_STEP_ACCENTS[0],
              )}
            >
              <div className="flex h-full min-h-0 w-full min-w-0 max-w-full flex-1 flex-col text-left text-film-black">
                <h2 className="exhibit-title mb-1.5 w-full min-w-0 shrink-0 text-center text-sm font-semibold uppercase leading-snug tracking-wide sm:mb-2 sm:text-lg md:mb-2.5 md:text-2xl">
                  {t("INSTRUCTIONS", "ANLEITUNG:")}
                </h2>
                <ol
                  className="exhibit-title flex h-full min-h-0 w-full list-none flex-col pl-0 pr-0.5 text-sm font-medium leading-snug sm:pl-0.5 sm:text-lg sm:leading-snug md:pl-1 md:text-2xl"
                >
                  {CAMERA_LEFT_INSTRUCTIONS.map((line, i) => (
                    <li
                      key={i}
                      className="flex min-h-0 flex-1 flex-col justify-center gap-0 py-0.5 [overflow-wrap:anywhere]"
                    >
                      <div className="flex min-w-0 items-baseline gap-1.5 sm:gap-2">
                        <span className="w-4 shrink-0 text-right font-semibold tabular-nums sm:w-5">
                          {i + 1}
                          {"."}
                        </span>
                        <span className="min-w-0 flex-1 leading-snug">
                          {t(line.en, line.de)}
                        </span>
                      </div>
                    </li>
                  ))}
                </ol>
              </div>
            </div>
          </aside>

          <div className="flex h-full min-h-0 min-w-0 flex-col">
            {/* `grid` 1fr + auto so the action row never gets clipped (flex+flex-1 could squeeze it to 0 with overflow-hidden). */}
            <div
              className="exhibit-panel relative grid h-full min-h-0 w-full min-w-0 [grid-template-rows:minmax(0,1fr)_auto] overflow-hidden rounded-2xl p-2 sm:p-3 md:p-4"
            >
              <div className="relative flex min-h-0 w-full min-w-0 flex-col items-center justify-center">
                {error ? (
                  <div className="flex w-full min-w-0 max-w-full flex-col items-center gap-6 text-center">
                    <p className="exhibit-title max-w-full px-2 text-sm font-medium leading-snug text-balance text-destructive sm:text-lg sm:leading-snug md:max-w-xl md:text-2xl">
                      {error}
                    </p>
                    <Button
                      size="xl"
                      onClick={startCamera}
                      className="cta-step-3 px-10 py-4 text-base font-semibold uppercase tracking-wider text-white md:text-lg"
                    >
                      {t("Try Again", "Erneut versuchen")}
                    </Button>
                  </div>
                ) : capturedImage ? (
                  <div className="flex h-full w-full min-h-0 min-w-0 items-center justify-center">
                    <img
                      src={capturedImage}
                      alt={t("Captured image", "Aufgenommenes Bild")}
                      className="max-h-full max-w-full object-contain"
                    />
                  </div>
                ) : (
                  <div className="relative flex h-full w-full min-h-0 max-w-3xl items-center justify-center self-center">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="max-h-full max-w-full object-contain bg-muted"
                    />
                    {countdownSeconds !== null ? (
                      <div className="absolute inset-0 flex items-center justify-center rounded-2xl bg-black/30">
                        <p className="text-4xl font-bold text-white drop-shadow-lg sm:text-6xl md:text-8xl">
                          {countdownSeconds}
                        </p>
                      </div>
                    ) : null}
                  </div>
                )}
              </div>

              {!error ? (
                <div className="relative z-10 flex w-full min-w-0 flex-col items-stretch justify-center border-t border-border/40 bg-white/95 py-2 pt-2 sm:py-2.5 sm:pt-3">
                  {capturedImage ? (
                    <div className="flex w-full max-w-2xl flex-col items-stretch justify-center gap-2 self-center sm:flex-row sm:gap-3 md:gap-4">
                      <Button
                        size="xl"
                        onClick={retakePhoto}
                        className="cta-step-3 gap-2 rounded-2xl px-4 py-3 text-base font-semibold uppercase tracking-wider text-white md:px-6 md:py-4"
                      >
                        <RefreshCcw className="h-5 w-5" />
                        {t("RETAKE PHOTO", "FOTO ERNEUT AUFNEHMEN")}
                      </Button>
                      <Button
                        size="xl"
                        onClick={proceedToSelection}
                        className="cta-step-2 gap-2 rounded-2xl px-4 py-3 text-base font-semibold uppercase tracking-wider text-white md:px-8 md:py-4"
                      >
                        {t("Continue", "Weiter")}
                        <ArrowRight className="h-5 w-5" />
                      </Button>
                    </div>
                  ) : (
                    <Button
                      size="xl"
                      onClick={startCountdown}
                      disabled={!!error || countdownSeconds !== null}
                      className={`mx-auto w-full max-w-2xl items-center justify-center gap-3 ${captureButtonClassName}`}
                    >
                      <Camera className="h-6 w-6 md:h-7 md:w-7" />
                      {countdownSeconds !== null
                        ? t(
                            `Capturing in ${countdownSeconds}...`,
                            `Aufnahme in ${countdownSeconds}...`,
                          )
                        : t("Capture", "Aufnehmen")}
                    </Button>
                  )}
                </div>
              ) : null}
            </div>
          </div>

          <aside className="flex min-h-0 min-w-0 flex-col overflow-hidden lg:h-full">
            <div
              className={cn(
                sidePanelFrameClass,
                "items-center text-center",
                EXHIBIT_STEP_ACCENTS[0],
              )}
            >
              <p className="exhibit-title text-sm font-semibold text-film-black sm:text-base md:text-lg">
                {t("Examples", "Beispiele")}
              </p>
              <p className="mt-1 text-xs text-film-black/80 sm:mt-2 sm:text-sm md:text-base">
                {t("Coming later.", "Folgen spaeter.")}
              </p>
            </div>
          </aside>
        </div>
        <canvas ref={canvasRef} className="hidden" />
      </div>
    </div>
  );
};

export default CameraPage;
