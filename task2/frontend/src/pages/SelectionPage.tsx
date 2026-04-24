import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  CameraStepCard,
  cameraStepHeadingClass,
} from "@/components/CameraStepCard";
import { FlowStepIndicator } from "@/components/FlowStepIndicator";
import { Button } from "@/components/ui/button";
import {
  continueButtonClassName,
  continueToQrButtonClassName,
  flowExitButtonClassName,
  retakeButtonClassName,
} from "@/lib/flowCtaClassNames";
import { t } from "@/lib/localization";
import { EXHIBIT_STEPS, type ExhibitStepIndex } from "@/lib/exhibitFlow";
import { cn } from "@/lib/utils";
import { ArrowLeft, ArrowRight, Check, Circle, LogOut, X } from "lucide-react";

type Background = {
  id: string;
  title: string;
  url: string;
  positions?: [number, number][];
  max_w?: number;
  max_h?: number;
  bg_w?: number;
  bg_h?: number;
  silhouette_color?: [number, number, number];
};

type SilhouetteData = {
  id: string;
  url: string;
};

/** Same as `SilhouetteLeftColumn` / `CameraPage` step cards. */
const cameraStepCardClassName = "h-full max-w-full min-h-0 min-w-0 w-full";

const greyedStepClass =
  "opacity-50 grayscale transition-[filter,opacity] duration-200";

/** Scene names in strip + export queue (a touch under `cameraStepDescriptionClass`). */
const sceneTitleLabelClassName =
  "text-[0.9375rem] leading-snug md:text-[1.1875rem]";

/** 0-based index for "Step 3 of 4" (pick backgrounds). */
const SELECTION_ACTIVE_STEP_INDEX: ExhibitStepIndex = 2;

/** Matches exhibit-shell gutter so footer rules don’t read as a grey cut. */
const panelGutterRuleClass = "border-t border-[hsl(var(--background))]";

/** Shared padding with the center preview action strip for aligned bottom row. */
const selectionActionStripPaddingClass =
  "px-2 py-2 pt-2 sm:px-3 sm:py-2.5 sm:pt-3 md:px-4 md:py-2.5 md:pt-3";

const SelectionPage = () => {
  const navigate = useNavigate();
  const [selectedCards, setSelectedCards] = useState<string[]>([]);
  const [backgrounds, setBackgrounds] = useState<Background[]>([]);
  const [previewBgId, setPreviewBgId] = useState<string | null>(null);
  const [silhouette, setSilhouette] = useState<SilhouetteData | null>(null);

  useEffect(() => {
    const storedSil = sessionStorage.getItem("silhouetteData");
    if (storedSil) {
      setSilhouette(JSON.parse(storedSil));
    } else {
      navigate("/camera");
    }

    const fetchBackgrounds = async () => {
      try {
        const res = await fetch("http://localhost:8000/backgrounds");
        if (res.ok) {
          const data = await res.json();
          setBackgrounds(data);
          if (data.length > 0) {
            setPreviewBgId(data[0].id);
          }
        }
      } catch (error) {
        console.error("Failed to fetch backgrounds", error);
      }
    };

    fetchBackgrounds();
  }, [navigate]);

  const toggleSelection = (id: string) => {
    setSelectedCards((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id],
    );
  };

  const handleProceed = () => {
    sessionStorage.setItem(
      "selectedBackgroundIds",
      JSON.stringify(selectedCards),
    );
    const selectedBackgrounds = backgrounds.filter((bg) =>
      selectedCards.includes(bg.id),
    );
    sessionStorage.setItem(
      "selectedBackgrounds",
      JSON.stringify(selectedBackgrounds),
    );
    sessionStorage.removeItem("preComposites");
    navigate("/confirmation");
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

  const currentPreviewBg = useMemo(
    () => backgrounds.find((background) => background.id === previewBgId),
    [backgrounds, previewBgId],
  );

  const selectedBackgrounds = useMemo(
    () =>
      backgrounds.filter((background) => selectedCards.includes(background.id)),
    [backgrounds, selectedCards],
  );

  const isCurrentPreviewSelected = currentPreviewBg
    ? selectedCards.includes(currentPreviewBg.id)
    : false;

  return (
    <div className="exhibit-shell flex h-full min-h-0 w-full min-w-0 flex-col overflow-hidden overflow-x-hidden">
      <div className="w-full flex-shrink-0 px-4 pt-1 pb-1 sm:pt-2 sm:pb-2 md:px-6 md:pt-2 md:pb-2 lg:pt-3 lg:pb-3">
        <div className="grid w-full min-w-0 grid-cols-[minmax(0,1fr)_minmax(0,2.25fr)_minmax(0,1.1fr)] items-center gap-2 sm:gap-3 md:gap-4 lg:gap-5">
          <FlowStepIndicator
            activeStepIndex={2}
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
                      index !== SELECTION_ACTIVE_STEP_INDEX && greyedStepClass,
                    )}
                    aria-current={
                      index === SELECTION_ACTIVE_STEP_INDEX ? "step" : undefined
                    }
                  >
                    <CameraStepCard
                      stepIndex={index as ExhibitStepIndex}
                      className={cameraStepCardClassName}
                    />
                  </div>
                ))}
              </div>
            </div>
          </aside>

          <div className="flex h-full min-h-0 min-w-0 flex-col">
            <div className="exhibit-panel relative grid h-full min-h-0 w-full min-w-0 [grid-template-rows:minmax(0,1fr)_auto] overflow-hidden rounded-2xl p-0">
              <div className="relative flex min-h-0 w-full min-w-0 flex-col p-2 sm:p-3 md:p-4">
                <div className="flex min-h-0 flex-1 items-center justify-center">
                  {currentPreviewBg && silhouette ? (
                    <div className="relative flex w-full max-w-6xl justify-center">
                      {(() => {
                        const bw = currentPreviewBg.bg_w || 1920;
                        const bh = currentPreviewBg.bg_h || 1080;
                        const maxWidthFromHeightVh = (53 * bw) / bh;

                        return (
                          <div
                            className={`relative h-full w-full overflow-hidden rounded-2xl border-4 shadow-2xl transition-all ${isCurrentPreviewSelected ? "border-film-green ring-4 ring-film-green/30" : "border-border"}`}
                            style={{
                              width: `min(100%, ${maxWidthFromHeightVh}vh)`,
                              aspectRatio: `${bw} / ${bh}`,
                            }}
                          >
                            <img
                              src={`http://localhost:8000${currentPreviewBg.url}`}
                              alt={currentPreviewBg.title}
                              className="absolute inset-0 h-full w-full object-contain"
                            />

                            {(() => {
                              const pos = currentPreviewBg.positions?.[0] || [
                                bw / 2,
                                bh,
                              ];
                              const mw = currentPreviewBg.max_w;
                              const mh = currentPreviewBg.max_h;
                              const tint = currentPreviewBg.silhouette_color || [
                                0, 0, 0,
                              ];
                              const tintCss = `rgb(${tint[0]}, ${tint[1]}, ${tint[2]})`;
                              const silhouetteUrl = `http://localhost:8000${silhouette.url}`;
                              const widthPct = mw ? `${(mw / bw) * 100}%` : "30%";
                              const heightPct = mh ? `${(mh / bh) * 100}%` : "60%";

                              return (
                                <div
                                  aria-label={t("My silhouette", "Meine Silhouette")}
                                  className="absolute pointer-events-none transition-all duration-500"
                                  style={{
                                    left: `${(pos[0] / bw) * 100}%`,
                                    top: `${(pos[1] / bh) * 100}%`,
                                    transform: "translate(-50%, -100%)",
                                    width: widthPct,
                                    height: heightPct,
                                    backgroundColor: tintCss,
                                    WebkitMaskImage: `url(${silhouetteUrl})`,
                                    maskImage: `url(${silhouetteUrl})`,
                                    WebkitMaskRepeat: "no-repeat",
                                    maskRepeat: "no-repeat",
                                    WebkitMaskPosition: "center bottom",
                                    maskPosition: "center bottom",
                                    WebkitMaskSize: "contain",
                                    maskSize: "contain",
                                  }}
                                />
                              );
                            })()}

                            {/* <div className="absolute left-3 top-3 rounded-full bg-black/55 px-4 py-1.5 text-sm font-semibold text-white md:text-base">
                              {t("Preview", "Vorschau")}: {currentPreviewBg.title}
                            </div> */}
                            {/* <div className="absolute bottom-3 left-3 rounded-full bg-black/55 px-4 py-1.5 text-xs font-medium text-white md:text-sm">
                              {t(
                                "Only selected scenes go to QR export.",
                                "Nur ausgewaehlte Szenen gehen in den QR-Export.",
                              )}
                            </div> */}
                            {currentPreviewBg ? (
                              <button
                                type="button"
                                onClick={() => toggleSelection(currentPreviewBg.id)}
                                className="absolute right-3 top-3 rounded-full bg-white/85 p-2 text-foreground shadow-lg transition-colors hover:bg-white"
                                title={
                                  isCurrentPreviewSelected
                                    ? t("Unselect scene", "Szene abwaehlen")
                                    : t("Select scene", "Szene auswaehlen")
                                }
                              >
                                <Check
                                  className={cn(
                                    "h-6 w-6",
                                    isCurrentPreviewSelected
                                      ? "text-film-green"
                                      : "text-muted-foreground",
                                  )}
                                />
                              </button>
                            ) : null}
                          </div>
                        );
                      })()}
                    </div>
                  ) : (
                    <div className="flex aspect-[16/9] w-full max-w-5xl items-center justify-center rounded-2xl bg-muted/60">
                      <span className="text-lg text-muted-foreground">
                        {t("Loading scene preview...", "Szenen-Vorschau laedt...")}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              <div className="relative z-10 flex w-full min-w-0 flex-col items-stretch justify-center border-t border-border/40 bg-white/95 px-2 py-2 pt-2 sm:px-3 sm:py-2.5 sm:pt-3 md:px-4 md:py-2.5 md:pt-3">
                <div
                  className={cn(
                    "grid w-full min-w-0 items-stretch gap-2 sm:gap-3 md:gap-4",
                    "grid-cols-1 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]",
                    "[&>button]:max-w-none [&>button]:min-w-0 [&>button]:w-full",
                  )}
                >
                  <Button
                    size="xl"
                    onClick={() => navigate("/silhouette")}
                    className={cn(retakeButtonClassName, "!text-white")}
                  >
                    <ArrowLeft className="shrink-0" />
                    {t("Go Back", "Zurueck gehen")}
                  </Button>
                  <Button
                    size="xl"
                    disabled={!currentPreviewBg}
                    onClick={() =>
                      currentPreviewBg ? toggleSelection(currentPreviewBg.id) : null
                    }
                    className={cn(continueButtonClassName, "!text-white")}
                  >
                    {isCurrentPreviewSelected ? (
                      <>
                        <Check className="shrink-0" />
                        {t("Selected for Download", "Fuer Download ausgewaehlt")}
                      </>
                    ) : (
                      <>
                        <Circle className="shrink-0" />
                        {t("Select for Download", "Zum Download auswaehlen")}
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </div>
          </div>

          <aside className="flex min-h-0 min-w-0 flex-col overflow-hidden lg:h-full">
            <div
              className={cn(
                "exhibit-panel flex h-full min-h-0 w-full min-w-0 max-h-full flex-col overflow-hidden rounded-2xl p-2 sm:p-3 md:p-4",
                "max-lg:max-h-[30vh] lg:max-h-none",
              )}
            >
              <h2 className="text-2xl md:text-3xl font-bold flex items-center">
                {t("Selected Scenes", "Ausgewaehlte Szenen")}
                <span className="ml-3 rounded-full bg-film-blue/10 px-3 py-0.5 text-lg md:text-xl text-film-blue">
                  {selectedCards.length}
                </span>
              </h2>

              <div className="soft-scroll mt-3 min-h-0 flex-1 space-y-2 overflow-y-auto overscroll-contain [scrollbar-gutter:stable] pr-1">
                {selectedBackgrounds.length === 0 ? (
                  <div className="flex h-full flex-col items-center justify-center rounded-xl border border-dashed border-border bg-muted/40 p-6 text-center text-lg md:text-xl font-medium text-muted-foreground">
                    <p>{t("No scenes selected. Select backgrounds to download.", "Noch keine Szenen ausgewaehlt. Waehle Hintergruende zum Herunterladen.")}</p>
                  </div>
                ) : (
                  selectedBackgrounds.map((background) => (
                    <div
                      key={background.id}
                      className={`flex items-center gap-3 rounded-xl border bg-white/80 p-2 shadow-sm transition-all hover:bg-white ${previewBgId === background.id ? "border-film-blue ring-1 ring-film-blue/20" : ""}`}
                    >
                      <button
                        type="button"
                        onClick={() => setPreviewBgId(background.id)}
                        className="flex min-w-0 flex-1 items-center gap-3 text-left"
                      >
                        <div className="h-12 w-20 shrink-0 overflow-hidden rounded-md border border-border/50 bg-muted">
                          <img
                            src={`http://localhost:8000${background.url}`}
                            alt={background.title}
                            className="h-full w-full object-cover"
                          />
                        </div>
                        <span
                          className={cn(
                            "block min-w-0 truncate font-medium text-film-black",
                            sceneTitleLabelClassName,
                          )}
                        >
                          {background.title}
                        </span>
                      </button>
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          toggleSelection(background.id);
                        }}
                        className="rounded-full p-2 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
                        title={t("Remove", "Entfernen")}
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                  ))
                )}
              </div>

              <div className="mt-3 shrink-0 border-t pt-3">
                <Button
                  size="xl"
                  className="cta-step-4 w-full shadow-md transition-transform hover:scale-[1.02] active:scale-[0.98] !h-14 md:!h-16 !text-xl md:!text-3xl !font-bold uppercase tracking-wide !text-white"
                  onClick={handleProceed}
                  disabled={selectedCards.length === 0}
                >
                  {t("Continue to Download", "Weiter zum Download")}
                </Button>
              </div>
            </div>
          </aside>
        </div>
      </div>

      <div className="w-full flex-shrink-0 px-4 pb-2 md:px-6 md:pb-3">
        <section className="exhibit-panel mt-1 rounded-2xl p-3 md:p-4">
          <div className="mb-3 w-full text-center">
            <h2
              className={cn(
                "exhibit-title min-w-0 text-film-black",
                cameraStepHeadingClass,
              )}
            >
              {t("Scene Strip", "Szenen-Leiste")}
            </h2>
          </div>
          <div className="custom-scrollbar flex w-full snap-x snap-mandatory items-start justify-center gap-4 overflow-x-auto pb-2">
            {backgrounds.map((background) => {
              const selected = selectedCards.includes(background.id);
              const isPreviewing = previewBgId === background.id;

              return (
                <button
                  type="button"
                  key={background.id}
                  className="group snap-start text-center"
                  onClick={() => setPreviewBgId(background.id)}
                >
                  <div
                    className={`relative h-[124px] w-[220px] overflow-hidden rounded-xl border-4 transition-all duration-200 ${isPreviewing ? "border-film-blue shadow-xl" : "border-transparent opacity-80 group-hover:opacity-100"} ${selected ? "ring-2 ring-film-green/40" : ""}`}
                  >
                    <img
                      src={`http://localhost:8000${background.url}`}
                      alt={background.title}
                      className="h-full w-full object-cover"
                    />
                    {isPreviewing ? (
                      <span className="absolute left-2 top-2 rounded-full bg-film-blue px-2 py-0.5 text-xs font-semibold text-white">
                        {t("Preview", "Vorschau")}
                      </span>
                    ) : null}
                    {selected ? (
                      <span className="absolute right-2 top-2 rounded-full bg-film-green px-2 py-0.5 text-xs font-semibold text-white">
                        {t("Selected", "Ausgewaehlt")}
                      </span>
                    ) : null}
                  </div>
                  <p
                    className={cn(
                      "exhibit-title mt-1.5 max-w-[220px] min-w-0 truncate text-center",
                      sceneTitleLabelClassName,
                      isPreviewing ? "text-film-blue" : "text-foreground/80",
                    )}
                  >
                    {background.title}
                  </p>
                </button>
              );
            })}
          </div>
        </section>
      </div>
    </div>
  );
};

export default SelectionPage;
