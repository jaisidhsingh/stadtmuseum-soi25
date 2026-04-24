import * as React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import {
  CameraStepCard,
  cameraStepDescriptionClass,
  cameraStepHeadingClass,
} from "@/components/CameraStepCard";
import { FlowStepIndicator } from "@/components/FlowStepIndicator";
import { type ExhibitStepIndex } from "@/lib/exhibitFlow";
import { cn } from "@/lib/utils";
import { t } from "@/lib/localization";
import {
  captureButtonClassName,
  continueButtonClassName,
  flowExitButtonClassName,
  retakeButtonClassName,
} from "@/lib/flowCtaClassNames";
import { Camera, RefreshCcw, ArrowRight, LogOut } from "lucide-react";
import pose1 from "../assets/pose_1.png";
import pose2 from "../assets/pose_2.png";
import pose3 from "../assets/pose_3.png";
import pose4 from "../assets/pose_4.png";
import pose5 from "../assets/pose_5.png";
import pose6 from "../assets/pose_6.png";

const POSE_EXAMPLE_IMAGES: readonly string[] = [
  pose1,
  pose2,
  pose3,
  pose4,
  pose5,
  pose6,
];

const COUNTDOWN_SECONDS = 10;

const CAMERA_STEP1_DESC_INSTRUCTION_CLASS =
  "text-[1.0625rem] leading-snug md:text-[1.4rem]";

/** Layout on the card root; typography lives in `CameraStepCard` (no drop shadow on camera). */
const cameraStepCardClassName =
  "h-full max-w-full min-h-0 min-w-0 w-full";

const CAMERA_STEP1_INSTRUCTION_LINES: { en: string; de: string }[] = [
  {
    en: "Make sure you are standing close to the white wall.",
    de: "Stelle dich nah an die weisse Wand.",
  },
  {
    en: "Ensure that your entire body is inside the camera frame.",
    de: "Achte darauf, dass dein ganzer Koerper im Kamerabild ist.",
  },
  {
    en: "Click on the CAPTURE button and strike a pose.",
    de: "Tippe auf AUFNEHMEN und nimm eine Pose ein.",
  },
  { en: "Wait 10 seconds.", de: "Warte 10 Sekunden." },
  {
    en: "Check if you like your pose.",
    de: "Pruefe, ob dir deine Pose gefaellt.",
  },
  {
    en: "Move on to the next step or retake your image.",
    de: "Weiter zum naechsten Schritt oder erneut aufnehmen.",
  },
];

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
      {/* Same 3 column template as the main grid: title left (column 1) with the step cards, EXIT in column 3. */}
      <div className="w-full flex-shrink-0 px-4 pt-1 pb-1 sm:pt-2 sm:pb-2 md:px-6 md:pt-2 md:pb-2 lg:pt-3 lg:pb-3">
        <div className="grid w-full min-w-0 grid-cols-[minmax(0,1fr)_minmax(0,1.85fr)_minmax(0,1.65fr)] items-center gap-2 sm:gap-3 md:gap-4 lg:gap-5">
          <FlowStepIndicator activeStepIndex={0} className="min-w-0 pl-2 sm:pl-3" />
          <div className="min-w-0" aria-hidden />
          <div className="flex min-w-0 items-center justify-end">
            <Button
              size="xl"
              onClick={handleExitSession}
              className={flowExitButtonClassName}
            >
              <LogOut className="shrink-0" />
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
                "exhibit-panel flex h-full min-h-0 w-full min-w-0 max-h-full flex-col overflow-hidden rounded-2xl p-2 sm:p-3 md:p-4",
                "max-lg:max-h-[30vh] lg:max-h-none",
              )}
            >
              <div className="exhibit-title flex min-h-0 w-full min-w-0 flex-1 flex-col gap-2 overflow-y-auto overscroll-contain [scrollbar-gutter:stable] sm:gap-2 md:gap-3">
                <div
                  className="w-full min-w-0 max-w-full shrink-0"
                  aria-current="step"
                >
                  <CameraStepCard
                    stepIndex={0}
                    className={cameraStepCardClassName}
                  >
                    <ol
                      className={cn(
                        "exhibit-title mt-2 list-inside list-decimal space-y-1.5 border-t border-border/50 pl-0 pt-2 text-left text-film-black [overflow-wrap:anywhere] marker:font-medium",
                        CAMERA_STEP1_DESC_INSTRUCTION_CLASS,
                      )}
                    >
                      {CAMERA_STEP1_INSTRUCTION_LINES.map((line, i) => (
                        <li key={i}>{t(line.en, line.de)}</li>
                      ))}
                    </ol>
                  </CameraStepCard>
                </div>
                {([1, 2, 3] as const).map((stepIndex) => (
                  <div
                    key={stepIndex}
                    className="w-full min-w-0 max-w-full shrink-0 opacity-50 grayscale transition-[filter,opacity] duration-200"
                  >
                    <CameraStepCard
                      stepIndex={stepIndex as ExhibitStepIndex}
                      className={cameraStepCardClassName}
                    />
                  </div>
                ))}
              </div>
            </div>
          </aside>

          <div className="flex h-full min-h-0 min-w-0 flex-col">
            {/* `grid` 1fr + auto so the action row never gets clipped (flex+flex-1 could squeeze it to 0 with overflow-hidden). */}
            <div
              className="exhibit-panel relative grid h-full min-h-0 w-full min-w-0 [grid-template-rows:minmax(0,1fr)_auto] overflow-hidden rounded-2xl p-0"
            >
              <div className="relative flex min-h-0 w-full min-w-0 flex-col items-center justify-center p-2 sm:p-3 md:p-4">
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
                <div className="relative z-10 flex w-full min-w-0 flex-col items-stretch justify-center border-t border-border/40 bg-white/95 px-2 py-2 pt-2 sm:px-3 sm:py-2.5 sm:pt-3 md:px-4 md:py-2.5 md:pt-3">
                  <div
                    className={cn(
                      "grid w-full min-w-0 items-stretch gap-2 sm:gap-3 md:gap-4",
                      "grid-cols-1 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]",
                      "[&>button]:max-w-none [&>button]:min-w-0 [&>button]:w-full",
                    )}
                  >
                    {capturedImage ? (
                      <>
                        <Button
                          size="xl"
                          onClick={retakePhoto}
                          className={retakeButtonClassName}
                        >
                          <RefreshCcw className="shrink-0" />
                          {t("RETAKE PHOTO", "FOTO ERNEUT AUFNEHMEN")}
                        </Button>
                        <Button
                          size="xl"
                          onClick={proceedToSelection}
                          className={continueButtonClassName}
                        >
                          {t("Continue", "Weiter")}
                          <ArrowRight className="shrink-0" />
                        </Button>
                      </>
                    ) : (
                      <Button
                        size="xl"
                        onClick={startCountdown}
                        disabled={!!error || countdownSeconds !== null}
                        className={cn("col-span-full", captureButtonClassName)}
                      >
                        <Camera className="shrink-0" />
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
              ) : null}
            </div>
          </div>

          <aside className="flex min-h-0 min-w-0 flex-col overflow-hidden lg:h-full">
            <div
              className={cn(
                "exhibit-panel flex h-full min-h-0 w-full min-w-0 max-h-full flex-col overflow-hidden rounded-2xl p-2 sm:p-3 md:p-4",
                "max-lg:max-h-[30vh] lg:max-h-none",
              )}
            >
              <h3
                className={cn(
                  "exhibit-title min-w-0 w-full shrink-0 text-center text-film-black",
                  cameraStepHeadingClass,
                )}
              >
                {t("Examples", "Beispiele")}
              </h3>
              <p
                className={cn(
                  "mt-1 w-full shrink-0 text-balance text-center text-film-black",
                  cameraStepDescriptionClass,
                )}
              >
                {t(
                  "Use one of the following poses as a guide to strike your pose!",
                  "Nutze eine der folgenden Pose als Anleitung, um deine Pose einzunehmen!",
                )}
              </p>
              <div className="mt-2 min-h-0 w-full flex-1 overflow-y-auto overscroll-contain [scrollbar-gutter:stable] sm:mt-3">
                <div className="grid w-full min-w-0 grid-cols-2 gap-1.5 sm:gap-2">
                  {POSE_EXAMPLE_IMAGES.map((src, i) => (
                    <div
                      key={src}
                      className="relative aspect-[3/5] w-full min-w-0 overflow-hidden rounded-lg border border-border/50 bg-muted"
                    >
                      <img
                        src={src}
                        alt={t(
                          `Example pose ${i + 1} of 6`,
                          `Beispielpose ${i + 1} von 6`,
                        )}
                        className="h-full w-full object-cover object-center"
                        loading="lazy"
                        decoding="async"
                      />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </aside>
        </div>
        <canvas ref={canvasRef} className="hidden" />
      </div>
    </div>
  );
};

export default CameraPage;
