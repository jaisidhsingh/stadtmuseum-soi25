import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { CameraStepCard, cameraStepHeadingClass } from "@/components/CameraStepCard";
import { FlowStepIndicator } from "@/components/FlowStepIndicator";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { flowExitButtonClassName } from "@/lib/flowCtaClassNames";
import { t } from "@/lib/localization";
import { EXHIBIT_STEPS, type ExhibitStepIndex } from "@/lib/exhibitFlow";
import { cn } from "@/lib/utils";
import { toast } from "@/hooks/use-toast";
import {
  ArrowLeft,
  CheckCircle2,
  LogOut,
  QrCode,
  ScanLine,
  X,
  ZoomIn,
} from "lucide-react";

type CompositeItem = {
  id: string;
  originalBgId: string;
  title: string;
  url: string;
};

type ShareInfo = {
  shareUrl: string;
  expiresAt: string;
};

/** Same as `SelectionPage` / `SilhouetteLeftColumn` step cards. */
const cameraStepCardClassName = "h-full max-w-full min-h-0 min-w-0 w-full";

const greyedStepClass =
  "opacity-50 grayscale transition-[filter,opacity] duration-200";

/** Same list body as `CameraPage` / `SilhouetteLeftColumn` step instruction lists. */
const stepInstructionListTextClass =
  "text-[1.0625rem] leading-snug md:text-[1.4rem]";

/** Step 4 of 4 (QR / confirmation) — `CameraStepCard` body list. */
const CONFIRMATION_STEP4_INSTRUCTION_LINES: { en: string; de: string }[] = [
  {
    en: "Review your selected backgrounds with your silhouette inserted in them.",
    de: "Pruefe deine ausgewaehlten Hintergruende mit deiner eingesetzten Silhouette.",
  },
  {
    en: "Go back to select more scenes if you wish.",
    de: "Geh zurueck, um bei Bedarf weitere Szenen auszuwaehlen.",
  },
  {
    en: 'Tap "Create QR Share" to generate a unique code to download your gallery.',
    de: 'Tippe auf „QR-Freigabe erstellen", um einen eindeutigen Code zum Herunterladen deiner Galerie zu erhalten.',
  },
  {
    en: "Connect to Wi-Fi if required.",
    de: "Stelle bei Bedarf eine WLAN-Verbindung her.",
  },
  {
    en: "Scan the QR code with your phone's camera.",
    de: "Scanne den QR-Code mit der Kamera deines Smartphones.",
  },
  {
    en: "Download your scenes!",
    de: "Lade deine Szenen herunter!",
  },
];

/** Step 4 of 4 (QR / confirmation). */
const CONFIRMATION_ACTIVE_STEP_INDEX: ExhibitStepIndex = 3;

const ConfirmationPage = () => {
  const navigate = useNavigate();
  const [compositedItems, setCompositedItems] = useState<CompositeItem[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [enlargedImage, setEnlargedImage] = useState<CompositeItem | null>(null);
  const [shareInfo, setShareInfo] = useState<ShareInfo | null>(null);
  const [isCreatingShare, setIsCreatingShare] = useState(false);

  const isShareLocked = shareInfo !== null;

  useEffect(() => {
    const silhouetteDataStr = sessionStorage.getItem("silhouetteData");
    const bgIdsStr = sessionStorage.getItem("selectedBackgroundIds");
    const bgObjectsStr = sessionStorage.getItem("selectedBackgrounds");
    const preCompositesStr = sessionStorage.getItem("preComposites");

    if (!silhouetteDataStr || !bgIdsStr) {
      navigate("/selection");
      return;
    }

    const silhouette = JSON.parse(silhouetteDataStr);
    const bgIds: string[] = JSON.parse(bgIdsStr);
    const bgObjects = bgObjectsStr ? JSON.parse(bgObjectsStr) : [];
    const preComposites: Record<string, string> = preCompositesStr
      ? JSON.parse(preCompositesStr)
      : {};

    const generateComposites = async () => {
      setIsProcessing(true);
      try {
        const promises = bgIds.map(async (bgId) => {
          const bgObj = bgObjects.find((b: { id: string }) => b.id === bgId);
          const title = bgObj
            ? bgObj.title
            : t(`Background ${bgId}`, `Hintergrund ${bgId}`);

          if (preComposites[bgId]) {
            return {
              id: `pre_${bgId}`,
              originalBgId: bgId,
              title,
              url: preComposites[bgId],
            } as CompositeItem;
          }

          const res = await fetch("http://localhost:8000/composite", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              silhouette_id: silhouette.id,
              background_id: bgId,
            }),
          });
          if (!res.ok) throw new Error("Composition failed");
          const data = await res.json();

          return {
            id: data.result.id,
            originalBgId: bgId,
            title,
            url: `http://localhost:8000${data.result.url}`,
          } as CompositeItem;
        });

        const results = await Promise.all(promises);
        setCompositedItems(results);
      } catch (error) {
        console.error(error);
        toast({
          title: t("Error", "Fehler"),
          description: t(
            "Failed to generate some composite images.",
            "Einige Kompositionsbilder konnten nicht erzeugt werden.",
          ),
          variant: "destructive",
        });
      } finally {
        setIsProcessing(false);
      }
    };

    generateComposites();
  }, [navigate]);

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

  const removeItem = (id: string) => {
    if (isShareLocked) {
      return;
    }

    setCompositedItems((prev) => prev.filter((item) => item.id !== id));
  };

  const handleCreateShare = async () => {
    if (isShareLocked) {
      return;
    }

    if (compositedItems.length === 0) {
      toast({
        title: t("No images", "Keine Bilder"),
        description: t(
          "Generate at least one composition first.",
          "Bitte zuerst mindestens eine Komposition erzeugen.",
        ),
        variant: "destructive",
      });
      return;
    }

    setIsCreatingShare(true);
    try {
      const res = await fetch("http://localhost:8000/share/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ids: compositedItems.map((item) => item.id),
          ttl_minutes: 15,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(
          err.detail ||
            t(
              "Failed to create share link",
              "Freigabe-Link konnte nicht erstellt werden",
            ),
        );
      }

      const data = await res.json();
      setShareInfo({ shareUrl: data.share_url, expiresAt: data.expires_at });
    } catch (error: unknown) {
      toast({
        title: t("Share link failed", "Freigabe-Link fehlgeschlagen"),
        description:
          error instanceof Error
            ? error.message
            : t(
                "Could not create QR share link.",
                "QR-Freigabe-Link konnte nicht erstellt werden.",
              ),
        variant: "destructive",
      });
    } finally {
      setIsCreatingShare(false);
    }
  };

  const handleStartOver = async () => {
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
    <div className="exhibit-shell flex h-full min-h-0 w-full min-w-0 flex-col overflow-hidden overflow-x-hidden">
      <div className="w-full flex-shrink-0 px-4 pt-1 pb-1 sm:pt-2 sm:pb-2 md:px-6 md:pt-2 md:pb-2 lg:pt-3 lg:pb-3">
        <div className="grid w-full min-w-0 grid-cols-[minmax(0,1fr)_minmax(0,2.25fr)_minmax(0,1.1fr)] items-center gap-2 sm:gap-3 md:gap-4 lg:gap-5">
          <FlowStepIndicator
            activeStepIndex={3}
            className="min-w-0 pl-2 sm:pl-3"
          />
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

      <div className="min-h-0 w-full min-w-0 flex-1 overflow-hidden px-4 pb-1 pt-0 md:px-6 md:pb-1.5">
        <div className="grid h-full min-h-0 w-full min-w-0 items-stretch gap-2 sm:gap-3 md:gap-4 max-lg:grid-cols-1 max-lg:grid-rows-[minmax(0,auto)_minmax(0,1fr)_minmax(0,auto)] lg:grid-cols-[minmax(0,1fr)_minmax(0,2.25fr)_minmax(0,1.1fr)] lg:grid-rows-1 lg:gap-5">
          <aside className="flex min-h-0 min-w-0 flex-col overflow-hidden lg:h-full">
            <div
              className={cn(
                "exhibit-panel flex h-full min-h-0 w-full min-w-0 max-h-full flex-col overflow-hidden rounded-2xl p-2 sm:p-3 md:p-4",
                "max-lg:max-h-[30vh] lg:max-h-none",
              )}
            >
              <div className="exhibit-title flex min-h-0 w-full min-w-0 flex-1 flex-col gap-2 overflow-y-auto overscroll-contain [scrollbar-gutter:stable] sm:gap-2 md:gap-3">
                {EXHIBIT_STEPS.map((step, index) => (
                  <div
                    key={step.titleEn}
                    className={cn(
                      "w-full min-w-0 max-w-full shrink-0",
                      index !== CONFIRMATION_ACTIVE_STEP_INDEX && greyedStepClass,
                    )}
                    aria-current={
                      index === CONFIRMATION_ACTIVE_STEP_INDEX
                        ? "step"
                        : undefined
                    }
                  >
                    <CameraStepCard
                      stepIndex={index as ExhibitStepIndex}
                      className={cameraStepCardClassName}
                    >
                      {index === CONFIRMATION_ACTIVE_STEP_INDEX ? (
                        <ol
                          className={cn(
                            "exhibit-title mt-2 list-inside list-decimal space-y-1.5 border-t border-border/50 pl-0 pt-2 text-left text-film-black [overflow-wrap:anywhere] marker:font-medium",
                            stepInstructionListTextClass,
                          )}
                        >
                          {CONFIRMATION_STEP4_INSTRUCTION_LINES.map((line, i) => (
                            <li key={i}>{t(line.en, line.de)}</li>
                          ))}
                        </ol>
                      ) : null}
                    </CameraStepCard>
                  </div>
                ))}
              </div>
            </div>
          </aside>

          <div className="flex h-full min-h-0 min-w-0 flex-col">
            <div className="exhibit-panel flex h-full min-h-0 w-full min-w-0 flex-col overflow-hidden rounded-2xl p-4 md:p-5">
              <div className="mb-3 flex min-w-0 flex-wrap items-center justify-between gap-3">
                <h2
                  className={cn(
                    "exhibit-title min-w-0 text-film-black",
                    cameraStepHeadingClass,
                  )}
                >
                  {t("Your Gallery", "Deine Galerie")}
                </h2>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full bg-film-blue/10 px-3 py-1 text-sm font-semibold text-film-blue">
                    {compositedItems.length} {t("scene(s)", "Szene(n)")}
                  </span>
                  {isShareLocked ? (
                    <span className="film-tag">
                      {t("Share Locked", "Freigabe gesperrt")}
                    </span>
                  ) : null}
                </div>
              </div>
              {!isShareLocked ? (
                <div className="mb-3">
                  <Button
                    variant="outline"
                    size="xl"
                    className="exhibit-panel w-full min-w-0 border-film-blue/20 bg-white/70 sm:w-auto"
                    onClick={() => navigate("/selection")}
                  >
                    <ArrowLeft className="h-5 w-5 shrink-0" />
                    {t("Back to Selection", "Zurueck zur Auswahl")}
                  </Button>
                </div>
              ) : null}

              {isProcessing ? (
                <div className="flex min-h-0 flex-1 items-center justify-center rounded-2xl bg-muted/50 p-6 text-center">
                  <p className="animate-pulse text-lg font-medium text-muted-foreground">
                    {t("Generating artwork...", "Bilder werden erstellt...")}
                  </p>
                </div>
              ) : compositedItems.length === 0 ? (
                <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-4 rounded-2xl border border-dashed border-border bg-muted/40 p-6 text-center">
                  <p className="text-base text-muted-foreground md:text-lg">
                    {t(
                      "No selected scenes found.",
                      "Keine ausgewaehlten Szenen gefunden.",
                    )}
                  </p>
                  {!isShareLocked ? (
                    <Button onClick={() => navigate("/selection")} size="xl">
                      {t("Return to Selection", "Zurueck zur Auswahl")}
                    </Button>
                  ) : null}
                </div>
              ) : (
                <div className="soft-scroll min-h-0 flex-1 overflow-y-auto overscroll-contain [scrollbar-gutter:stable] pr-1">
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 2xl:grid-cols-3">
                    {compositedItems.map((item) => (
                      <Card
                        key={item.id}
                        className="relative h-[320px] overflow-hidden rounded-xl border bg-white/85 shadow-md sm:h-[340px] md:h-[352px]"
                      >
                        <div className="h-[244px] w-full overflow-hidden bg-muted sm:h-[256px] md:h-[276px]">
                          <img
                            src={item.url}
                            alt={item.title}
                            className="h-full w-full object-cover"
                          />
                        </div>

                        <div className="flex h-[64px] items-center justify-between gap-2 p-2.5 sm:h-[68px] md:h-[76px]">
                          <span className="min-w-0 truncate text-sm font-medium md:text-base">
                            {item.title}
                          </span>
                          <div className="flex shrink-0 gap-1">
                            <button
                              type="button"
                              className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                              onClick={() => setEnlargedImage(item)}
                              title={t("Zoom", "Vergruessern")}
                            >
                              <ZoomIn className="h-5 w-5" />
                            </button>
                            <button
                              type="button"
                              className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive disabled:opacity-40"
                              onClick={() => removeItem(item.id)}
                              title={t("Remove", "Entfernen")}
                              disabled={isShareLocked}
                            >
                              <X className="h-5 w-5" />
                            </button>
                          </div>
                        </div>

                        {isShareLocked ? (
                          <div className="absolute left-2 top-2 rounded-full bg-film-green px-2 py-0.5 text-xs font-semibold text-white">
                            {t("Locked", "Gesperrt")}
                          </div>
                        ) : null}
                      </Card>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          <aside className="flex min-h-0 min-w-0 flex-col overflow-hidden lg:h-full">
            <div
              className={cn(
                "exhibit-panel flex h-full min-h-0 w-full min-w-0 max-h-full flex-col overflow-hidden rounded-2xl p-4 md:p-5",
                "max-lg:max-h-[50vh] lg:max-h-none",
              )}
            >
              {!shareInfo ? (
                <>
                  <div className="rounded-xl border border-film-red/20 bg-film-red/10 p-4 text-sm text-film-black md:text-base">
                    {t(
                      "QR link expires in 15 minutes and all generated data is deleted after 15 minutes.",
                      "Der QR-Link laeuft nach 15 Minuten ab und alle generierten Daten werden nach 15 Minuten geloescht.",
                    )}
                  </div>

                  <div className="mt-auto space-y-3 pt-4">
                    <Button
                      size="xl"
                      onClick={handleCreateShare}
                      disabled={isCreatingShare || compositedItems.length === 0}
                      className="cta-step-4 w-full font-semibold"
                    >
                      <QrCode className="h-5 w-5" />
                      {isCreatingShare
                        ? t("Creating QR...", "QR wird erstellt...")
                        : t("Create QR Share", "QR-Freigabe erstellen")}
                    </Button>
                    <Button
                      variant="outline"
                      size="xl"
                      className="w-full"
                      onClick={handleStartOver}
                    >
                      {t(
                        "Start over and clear data",
                        "Neu starten und Daten loeschen",
                      )}
                    </Button>
                  </div>
                </>
              ) : (
                <div className="flex min-h-0 min-w-0 flex-1 flex-col">
                  <div className="mb-3 inline-flex w-fit max-w-full items-center gap-2 rounded-full bg-film-green/10 px-3 py-1 text-sm font-semibold text-film-green">
                    <CheckCircle2 className="h-4 w-4 shrink-0" />
                    {t("QR ready and locked", "QR bereit und gesperrt")}
                  </div>
                  <h2 className="text-xl font-semibold md:text-2xl">
                    {t("Scan and save", "Scannen und speichern")}
                  </h2>
                  <p className="mt-1 text-sm text-muted-foreground md:text-base">
                    {t(
                      "Use your phone camera to open and save your artwork.",
                      "Nutze deine Smartphone-Kamera, um dein Kunstwerk zu speichern.",
                    )}
                  </p>

                  <div className="mt-4 min-h-0 flex-1 overflow-y-auto">
                    <div className="rounded-2xl border bg-white/86 p-4 shadow-lg">
                      <div className="flex items-center justify-center">
                        <img
                          src={`https://api.qrserver.com/v1/create-qr-code/?size=320x320&data=${encodeURIComponent(shareInfo.shareUrl)}`}
                          alt={t("Share QR code", "QR-Code teilen")}
                          className="h-48 w-48 max-w-full sm:h-56 sm:w-56 md:h-64 md:w-64"
                        />
                      </div>
                      <div className="mt-3 space-y-2 text-sm md:text-base">
                        <p className="flex items-start gap-2">
                          <ScanLine className="mt-0.5 h-4 w-4 shrink-0 text-film-blue" />
                          {t(
                            "1. Open phone camera and scan the QR code.",
                            "1. Smartphone-Kamera oeffnen und QR-Code scannen.",
                          )}
                        </p>
                        <p className="flex items-start gap-2">
                          <ScanLine className="mt-0.5 h-4 w-4 shrink-0 text-film-blue" />
                          {t(
                            "2. Tap the link to open your gallery.",
                            "2. Link antippen, um die Galerie zu oeffnen.",
                          )}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 rounded-xl border border-film-red/20 bg-film-red/10 p-4 text-sm text-film-black md:text-base">
                    {t(
                      `QR expires at ${new Date(shareInfo.expiresAt).toLocaleTimeString()}. All generated data is deleted after 15 minutes.`,
                      `QR gueltig bis ${new Date(shareInfo.expiresAt).toLocaleTimeString()}. Alle generierten Daten werden nach 15 Minuten geloescht.`,
                    )}
                  </div>

                  <div className="mt-auto shrink-0 pt-4">
                    <Button
                      size="xl"
                      onClick={handleStartOver}
                      className="w-full bg-film-red text-white hover:bg-film-red/90"
                    >
                      {t("Close and clear data", "Schliessen und Daten loeschen")}
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </aside>
        </div>
      </div>

      <Dialog open={!!enlargedImage} onOpenChange={() => setEnlargedImage(null)}>
        <DialogContent className="max-w-2xl border-none bg-transparent px-0 shadow-none">
          {enlargedImage ? (
            <img
              src={enlargedImage.url}
              alt={enlargedImage.title}
              className="h-auto w-full rounded"
            />
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default ConfirmationPage;
