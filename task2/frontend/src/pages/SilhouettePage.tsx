import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { toast } from "@/hooks/use-toast";
import { t } from "@/lib/localization";
import { EXHIBIT_STEP_ACCENTS } from "@/lib/exhibitFlow";
import { cn } from "@/lib/utils";
import { ArrowLeft, ArrowRight, LogOut } from "lucide-react";

type ImageResource = {
  id: string;
  url: string;
};

type CharacterSelections = {
  hat: string | null;
  arms: string | null;
  legs: string | null;
  feet: string | null;
};

type PartGroup = {
  id: keyof CharacterSelections;
  labelEn: string;
  labelDe: string;
  fallbacks: string[];
};

const PART_GROUPS: PartGroup[] = [
  {
    id: "hat",
    labelEn: "Head Accessories",
    labelDe: "Kopf-Accessoires",
    fallbacks: ["hat_thumbnail.png", "hat.png", "head.png"],
  },
  {
    id: "arms",
    labelEn: "Arms",
    labelDe: "Arme",
    fallbacks: [
      "arms_thumbnail.png",
      "right_forearm.png",
      "left_forearm.png",
      "right_upper_arm.png",
    ],
  },
  {
    id: "legs",
    labelEn: "Legs",
    labelDe: "Beine",
    fallbacks: [
      "legs_thumbnail.png",
      "right_calf.png",
      "left_calf.png",
      "right_thigh.png",
    ],
  },
  {
    id: "feet",
    labelEn: "Feet",
    labelDe: "Fuesse",
    fallbacks: ["feet_thumbnail.png", "right_foot.png", "left_foot.png"],
  },
];

const SILHOUETTE_LEFT_STEPS: { en: string; de: string }[] = [
  {
    en: "Choose a body part row.",
    de: "Eine Koerperteil-Reihe waehlen.",
  },
  {
    en: "Tap a character thumbnail.",
    de: "Eine Figuren-Miniatur antippen.",
  },
  {
    en: "Watch your silhouette update live.",
    de: "Silhouette live in der Vorschau ansehen.",
  },
];

const sidePanelFrameClass =
  "exhibit-panel-edge flex h-full min-h-0 min-w-0 max-h-full flex-col justify-center overflow-hidden rounded-2xl p-2 sm:p-3 md:p-4 max-lg:max-h-[30vh] lg:max-h-none";

const SilhouettePage = () => {
  const navigate = useNavigate();
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [contextId, setContextId] = useState<string | null>(null);
  const [originalResource, setOriginalResource] = useState<ImageResource | null>(
    null,
  );
  const [warpedResource, setWarpedResource] = useState<ImageResource | null>(
    null,
  );

  const [isLoadingOriginal, setIsLoadingOriginal] = useState(false);
  const [isLoadingWarped, setIsLoadingWarped] = useState(false);

  const [characters, setCharacters] = useState<string[]>([]);

  const [selections, setSelections] = useState<CharacterSelections>({
    hat: null,
    arms: null,
    legs: null,
    feet: null,
  });

  const isProcessing = isLoadingOriginal || isLoadingWarped;
  const selectedCount = Object.values(selections).filter(Boolean).length;

  const selectedParts = Object.entries(selections)
    .filter(([, characterName]) => characterName !== null)
    .map(([group]) => group);

  const characterMapping = Object.fromEntries(
    Object.entries(selections).filter(([, char]) => char !== null),
  );

  const displayResource =
    selectedParts.length > 0 && warpedResource
      ? warpedResource
      : originalResource;
  const previewImage = displayResource
    ? `http://localhost:8000${displayResource.url}`
    : null;

  useEffect(() => {
    const image = sessionStorage.getItem("capturedImage");
    if (image) {
      setCapturedImage(image);
    } else {
      navigate("/camera");
    }

    fetch("http://localhost:8000/characters")
      .then((res) => res.json())
      .then((data) => setCharacters(data))
      .catch((err) => console.error("Failed to fetch characters", err));
  }, [navigate]);

  const dataURLtoFile = (dataurl: string, filename: string) => {
    const arr = dataurl.split(",");
    const mimeMatch = arr[0].match(/:(.*?);/);
    const mime = mimeMatch ? mimeMatch[1] : "";
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
  };

  useEffect(() => {
    if (!capturedImage) return;
    let cancelled = false;

    const fetchOriginal = async () => {
      setIsLoadingOriginal(true);
      try {
        const formData = new FormData();
        formData.append("image", dataURLtoFile(capturedImage, "capture.png"));
        const res = await fetch("http://localhost:8000/segment", {
          method: "POST",
          body: formData,
        });
        if (!res.ok) throw new Error("Segment failed");
        const data = await res.json();
        if (!cancelled) {
          setOriginalResource(data.original);
          setContextId(data.context_id || null);
        }
      } catch (error) {
        console.error("Error fetching original:", error);
        if (!cancelled) {
          toast({
            title: t("Processing failed", "Verarbeitung fehlgeschlagen"),
            description: t(
              "Make sure the backend is running.",
              "Bitte pruefe, ob das Backend laeuft.",
            ),
            variant: "destructive",
          });
        }
      } finally {
        if (!cancelled) setIsLoadingOriginal(false);
      }
    };

    fetchOriginal();
    return () => {
      cancelled = true;
    };
  }, [capturedImage]);

  useEffect(() => {
    if (!originalResource || !contextId) return;

    if (selectedParts.length === 0) {
      setWarpedResource(null);
      return;
    }

    let cancelled = false;
    const timeoutId = setTimeout(async () => {
      setIsLoadingWarped(true);
      try {
        const res = await fetch("http://localhost:8000/stylize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            context_id: contextId,
            parts_to_warp: selectedParts.join(","),
            character_mapping: characterMapping,
            base_name: characters[0] || "Prince_Achmed",
          }),
        });
        if (!res.ok) throw new Error("Stylize failed");
        const data = await res.json();
        if (!cancelled) {
          setWarpedResource(data.stylized?.[0] || null);
        }
      } catch (error) {
        console.error("Error fetching warped:", error);
        if (!cancelled) setWarpedResource(null);
      } finally {
        if (!cancelled) setIsLoadingWarped(false);
      }
    }, 500);

    return () => {
      cancelled = true;
      clearTimeout(timeoutId);
    };
  }, [selections, originalResource, contextId, characters]);

  const toggleSelection = (
    group: keyof CharacterSelections,
    charName: string,
  ) => {
    setSelections((prev) => ({
      ...prev,
      [group]: prev[group] === charName ? null : charName,
    }));
  };

  const handleNext = () => {
    const resource = displayResource;
    if (resource) {
      sessionStorage.setItem("silhouetteData", JSON.stringify(resource));
      sessionStorage.setItem("silhouetteStyles", JSON.stringify(selections));
      navigate("/selection");
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
    <div className="exhibit-shell flex h-full min-h-0 w-full min-w-0 flex-col overflow-hidden overflow-x-hidden">
      <div className="w-full flex-shrink-0 px-4 pt-2 pb-2 sm:pt-3 sm:pb-3 md:px-6 md:pt-4 md:pb-4 lg:pt-5 lg:pb-5">
        <div className="grid w-full min-w-0 grid-cols-[minmax(0,1fr)_minmax(0,1.85fr)_minmax(0,1.65fr)] items-center gap-2 sm:gap-3 md:gap-4 lg:gap-5">
          <div className="min-w-0" aria-hidden />
          <p className="exhibit-title min-w-0 text-center text-2xl font-bold uppercase leading-tight tracking-wider text-film-black md:text-3xl">
            {t("Step 2 of 4", "Schritt 2 von 4")}
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
        <div className="grid h-full min-h-0 w-full min-w-0 items-stretch gap-2 sm:gap-3 md:gap-4 max-lg:grid-cols-1 max-lg:grid-rows-[minmax(0,auto)_minmax(0,1fr)_minmax(0,auto)] lg:grid-cols-[minmax(0,1fr)_minmax(0,1.85fr)_minmax(0,1.65fr)] lg:grid-rows-1 lg:gap-5">
          <aside className="flex min-h-0 min-w-0 flex-col overflow-hidden lg:h-full">
            <div
              className={cn(
                sidePanelFrameClass,
                "!justify-start",
                "min-h-0 pl-0.5",
                EXHIBIT_STEP_ACCENTS[1],
              )}
            >
              <div className="flex h-full min-h-0 w-full min-w-0 max-w-full flex-1 flex-col text-left text-film-black">
                <h2 className="exhibit-title mb-1.5 w-full min-w-0 shrink-0 text-center text-sm font-semibold uppercase leading-snug tracking-wide sm:mb-2 sm:text-lg md:mb-2.5 md:text-2xl">
                  {t("What to do here", "Was du hier machst")}
                </h2>
                <ol className="exhibit-title flex min-h-0 w-full list-none flex-col pl-0 pr-0.5 text-sm font-medium leading-snug sm:pl-0.5 sm:text-lg sm:leading-snug md:pl-1 md:text-2xl">
                  {SILHOUETTE_LEFT_STEPS.map((line, i) => (
                    <li
                      key={i}
                      className="flex min-h-0 flex-col justify-center gap-0 py-0.5 [overflow-wrap:anywhere]"
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
            <div className="exhibit-panel relative grid h-full min-h-0 w-full min-w-0 [grid-template-rows:minmax(0,1fr)_auto] overflow-hidden rounded-2xl p-2 sm:p-3 md:p-4">
              <div className="relative flex min-h-0 w-full min-w-0 flex-col items-center justify-center">
                {isProcessing && (
                  <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-background/50 backdrop-blur-sm">
                    <div className="mb-4 h-12 w-12 animate-spin rounded-full border-b-2 border-primary" />
                    <p className="exhibit-title text-sm font-medium sm:text-lg md:text-xl">
                      {t("Applying styles...", "Stile werden angewendet...")}
                    </p>
                  </div>
                )}

                {previewImage ? (
                  <div className="flex h-full w-full min-h-0 min-w-0 items-center justify-center">
                    <img
                      src={previewImage}
                      alt={t("Segmented silhouette", "Segmentierte Silhouette")}
                      className="max-h-full max-w-full object-contain"
                    />
                  </div>
                ) : capturedImage ? (
                  <div className="flex h-full w-full min-h-0 min-w-0 items-center justify-center">
                    <img
                      src={capturedImage}
                      alt={t("Captured silhouette", "Aufgenommene Silhouette")}
                      className="max-h-full max-w-full object-contain opacity-50 grayscale"
                    />
                  </div>
                ) : (
                  <p className="exhibit-title text-sm text-muted-foreground sm:text-base">
                    {t("Loading image...", "Bild wird geladen...")}
                  </p>
                )}
              </div>

              <div className="relative z-10 flex w-full min-w-0 flex-col items-stretch justify-center border-t border-border/40 bg-white/95 py-2 pt-2 sm:py-2.5 sm:pt-3">
                <div className="mx-auto flex w-full max-w-2xl flex-col items-stretch justify-center gap-2 sm:flex-row sm:gap-3 md:gap-4">
                  <Button
                    size="xl"
                    onClick={() => navigate("/camera")}
                    className="cta-step-3 gap-2 rounded-2xl px-4 py-3 text-base font-semibold uppercase tracking-wider text-white md:px-6 md:py-4"
                  >
                    <ArrowLeft className="h-5 w-5" />
                    {t("RETAKE PHOTO", "FOTO ERNEUT AUFNEHMEN")}
                  </Button>
                  <Button
                    size="xl"
                    onClick={handleNext}
                    disabled={!displayResource || isProcessing}
                    className="cta-step-2 gap-2 rounded-2xl px-4 py-3 text-base font-semibold uppercase tracking-wider text-white md:px-8 md:py-4"
                  >
                    {t("BACKGROUNDS", "HINTERGRUENDE")}
                    <ArrowRight className="h-5 w-5" />
                  </Button>
                </div>
              </div>
            </div>
          </div>

          {/* Taller than default side panels: Head → Feet must scroll inside; override shared `max-h-[30vh]`. */}
          <aside className="flex min-h-0 min-w-0 flex-col overflow-hidden max-lg:max-h-[min(60vh,36rem)] lg:h-full">
            <div
              className={cn(
                sidePanelFrameClass,
                "!items-stretch !justify-start",
                "h-full min-h-0 max-lg:!max-h-full",
                "bg-background",
              )}
            >
              <div className="flex h-full min-h-0 w-full min-w-0 flex-col text-film-black">
                <p
                  className={cn(
                    "exhibit-title shrink-0 text-center text-sm font-semibold leading-snug tracking-wide sm:text-lg md:text-2xl",
                    selectedCount > 0 ? "text-film-green" : "text-film-red",
                  )}
                  aria-live="polite"
                >
                  {selectedCount > 0
                    ? t(
                        `${selectedCount} style group(s) selected.`,
                        `${selectedCount} Stil-Gruppe(n) ausgewaehlt.`,
                      )
                    : t(
                        "No style selected yet. Start with any row.",
                        "Noch kein Stil ausgewaehlt. Beginne mit einer beliebigen Reihe.",
                      )}
                </p>
                <ScrollArea className="mt-2 min-h-0 w-full flex-1 pr-1">
                  <div className="space-y-5 pb-1 sm:space-y-6 md:space-y-7">
                    {PART_GROUPS.map((group) => (
                      <div key={group.id} className="min-w-0 space-y-2">
                        <h3 className="exhibit-title text-center text-sm font-semibold leading-snug tracking-wide sm:text-lg md:text-2xl">
                          {t(group.labelEn, group.labelDe)}
                        </h3>
                        <ScrollArea className="w-full max-w-full whitespace-nowrap">
                          <div className="flex w-max min-w-0 max-w-full space-x-3 p-0.5 sm:space-x-4">
                            <div
                              role="button"
                              tabIndex={0}
                              onClick={() =>
                                setSelections((prev) => ({
                                  ...prev,
                                  [group.id]: null,
                                }))
                              }
                              onKeyDown={(e) => {
                                if (e.key === "Enter" || e.key === " ") {
                                  e.preventDefault();
                                  setSelections((prev) => ({
                                    ...prev,
                                    [group.id]: null,
                                  }));
                                }
                              }}
                              className={`
                                relative flex h-24 w-24 flex-shrink-0 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl border-2 transition-all duration-200
                                hover:scale-105 active:scale-95 sm:h-28 sm:w-28
                                ${selections[group.id] === null ? "border-primary bg-primary/5 ring-2 ring-primary/30" : "border-transparent bg-muted/50 hover:border-primary/50"}
                              `}
                            >
                              <div className="mb-1.5 flex h-9 w-9 items-center justify-center rounded-full border-2 border-dashed border-muted-foreground/50 sm:mb-2 sm:h-10 sm:w-10">
                                <span className="block rotate-45 text-xl leading-none text-muted-foreground/50">
                                  +
                                </span>
                              </div>
                              <span className="exhibit-title text-[0.7rem] font-medium text-muted-foreground sm:text-sm">
                                {t("None", "Keine")}
                              </span>
                              {selections[group.id] === null && (
                                <div className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-sm">
                                  <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    strokeWidth="3"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    className="h-3 w-3"
                                  >
                                    <polyline points="20 6 9 17 4 12" />
                                  </svg>
                                </div>
                              )}
                            </div>

                            {characters.map((char) => {
                              const isSelected =
                                selections[
                                  group.id as keyof CharacterSelections
                                ] === char;
                              const initialThumbUrl = `http://localhost:8000/assets/${char}/${group.fallbacks[0]}`;

                              return (
                                <div
                                  key={char}
                                  role="button"
                                  tabIndex={0}
                                  onClick={() =>
                                    toggleSelection(group.id, char)
                                  }
                                  onKeyDown={(e) => {
                                    if (e.key === "Enter" || e.key === " ") {
                                      e.preventDefault();
                                      toggleSelection(group.id, char);
                                    }
                                  }}
                                  className={`
                                    relative h-24 w-24 flex-shrink-0 cursor-pointer overflow-hidden rounded-xl border-2 bg-muted/30 transition-all duration-200
                                    hover:scale-105 active:scale-95 sm:h-28 sm:w-28
                                    ${isSelected ? "border-primary ring-2 ring-primary/30" : "border-transparent hover:border-primary/50"}
                                  `}
                                >
                                  <div className="absolute inset-0 flex items-center justify-center p-1.5 sm:p-2">
                                    <img
                                      src={initialThumbUrl}
                                      alt={`${char} ${t(group.labelEn, group.labelDe)}`}
                                      className="max-h-full max-w-full object-contain drop-shadow-md"
                                      data-fallback-index="0"
                                      onError={(e) => {
                                        const target =
                                          e.target as HTMLImageElement;
                                        let currentIndex = parseInt(
                                          target.dataset.fallbackIndex || "0",
                                          10,
                                        );
                                        currentIndex++;
                                        if (
                                          currentIndex < group.fallbacks.length
                                        ) {
                                          target.dataset.fallbackIndex =
                                            currentIndex.toString();
                                          target.src = `http://localhost:8000/assets/${char}/${group.fallbacks[currentIndex]}`;
                                        } else {
                                          target.style.display = "none";
                                        }
                                      }}
                                    />
                                  </div>
                                  {isSelected && (
                                    <div className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-sm">
                                      <svg
                                        xmlns="http://www.w3.org/2000/svg"
                                        viewBox="0 0 24 24"
                                        fill="none"
                                        stroke="currentColor"
                                        strokeWidth="3"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        className="h-3 w-3"
                                      >
                                        <polyline points="20 6 9 17 4 12" />
                                      </svg>
                                    </div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                          <ScrollBar orientation="horizontal" />
                        </ScrollArea>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
};

export default SilhouettePage;
