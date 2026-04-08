import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { toast } from "@/hooks/use-toast";
import { t } from "@/lib/localization";
import { ArrowLeft, LogOut } from "lucide-react";

type ImageResource = {
  id: string;
  url: string;
};

// Represents user's character selections per body part group
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

  // Derive API payload parts
  const selectedParts = Object.entries(selections)
    .filter(([_, characterName]) => characterName !== null)
    .map(([group]) => group);

  const characterMapping = Object.fromEntries(
    Object.entries(selections).filter(([_, char]) => char !== null),
  );

  const displayResource =
    selectedParts.length > 0 && warpedResource
      ? warpedResource
      : originalResource;
  const previewImage = displayResource
    ? `http://localhost:8000${displayResource.url}`
    : null;

  // Initialize
  useEffect(() => {
    const image = sessionStorage.getItem("capturedImage");
    if (image) {
      setCapturedImage(image);
    } else {
      navigate("/camera");
    }

    // Fetch available characters
    fetch("http://localhost:8000/characters")
      .then((res) => res.json())
      .then((data) => setCharacters(data))
      .catch((err) => console.error("Failed to fetch characters", err));
  }, [navigate]);

  // Helper: dataURL → File
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

  // Effect 1: fetch the original silhouette once when capturedImage is ready
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

  // Effect 2: fetch warped silhouette whenever selections change (debounced)
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
    }, 500); // 500ms debounce since warping takes time

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
      // Optionally store the mapping if needed by next screen
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
    <div className="h-full min-h-0 exhibit-shell relative flex w-full flex-col gap-8 overflow-hidden overflow-x-hidden p-4 pt-20 md:flex-row md:p-8 md:pt-8">
      {/* Top Bar with Back Button */}
      <div className="absolute top-4 left-4 md:top-8 md:left-8 z-20">
        <Button
          variant="outline"
          size="xl"
          className="h-14 px-6 text-base md:text-lg rounded-xl shadow-sm hover:shadow-md transition-all flex items-center gap-2 bg-background/80 backdrop-blur-sm"
          onClick={() => navigate("/camera")}
        >
          <ArrowLeft className="w-6 h-6" />{" "}
          {t("Re-take Photo", "Foto erneut aufnehmen")}
        </Button>
      </div>
      <div className="absolute top-4 right-4 md:top-8 md:right-8 z-20">
        <Button
          variant="outline"
          size="xl"
          className="h-14 px-6 text-base md:text-lg rounded-xl border-film-red/40 bg-white/80 text-film-red hover:bg-film-red/10 hover:text-film-red"
          onClick={handleExitSession}
        >
          <LogOut className="w-5 h-5" /> {t("Exit Session", "Sitzung beenden")}
        </Button>
      </div>

      {/* Left Half: Large Captured Image / Preview */}
      <div className="flex-1 flex items-center justify-center bg-muted/20 rounded-lg p-4 relative overflow-hidden min-h-[50vh] md:min-h-full">
        {isProcessing && (
          <div className="absolute inset-0 bg-background/50 flex flex-col items-center justify-center z-10 backdrop-blur-sm">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
            <p className="text-lg font-medium">
              {t("Applying styles...", "Stile werden angewendet...")}
            </p>
          </div>
        )}

        {previewImage ? (
          <img
            src={previewImage}
            alt={t("Segmented silhouette", "Segmentierte Silhouette")}
            className="max-h-[80vh] w-auto object-contain rounded-lg shadow-lg"
          />
        ) : capturedImage ? (
          <img
            src={capturedImage}
            alt={t("Captured silhouette", "Aufgenommene Silhouette")}
            className="max-h-[80vh] w-auto object-contain rounded-lg shadow-lg opacity-50 grayscale"
          />
        ) : (
          <p className="text-muted-foreground">
            {t("Loading image...", "Bild wird geladen...")}
          </p>
        )}
      </div>

      {/* Right Half: Style Options */}
      <div className="w-full md:w-[500px] flex flex-col h-full shrink-0 z-10 pt-4 md:pt-0">
        <h2 className="text-4xl font-bold mb-2">
          {t("Step 2: Stylize", "Schritt 2: Silhouette gestalten")}
        </h2>
        <p className="text-base md:text-lg text-primary font-medium mb-3 animate-pulse">
          {t(
            "Tap the thumbnails below to apply character styles.",
            "Tippe auf die Miniaturen, um Figuren-Stile anzuwenden.",
          )}
        </p>

        <div className="mb-4 rounded-xl border border-film-green/30 bg-film-green/10 p-4">
          <p className="text-base font-semibold text-film-green md:text-lg">
            {t("What to do here", "Was du hier machst")}
          </p>
          <ul className="mt-2 space-y-1 text-sm text-foreground/90 md:text-base">
            <li>
              {t(
                "1. Choose a body part row.",
                "1. Eine Koerperteil-Reihe waehlen.",
              )}
            </li>
            <li>
              {t(
                "2. Tap a character thumbnail.",
                "2. Eine Figuren-Miniatur antippen.",
              )}
            </li>
            <li>
              {t(
                "3. Watch your silhouette update live.",
                "3. Silhouette live in der Vorschau ansehen.",
              )}
            </li>
          </ul>
          <p className="mt-3 inline-flex rounded-full border border-film-blue/30 bg-film-blue/10 px-3 py-1 text-sm font-semibold text-film-blue md:text-base">
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
        </div>

        <ScrollArea className="flex-1 pr-4 -mr-4 pb-8">
          <div className="space-y-8">
            {PART_GROUPS.map((group) => (
              <div key={group.id} className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-2xl font-semibold">
                    {t(group.labelEn, group.labelDe)}
                  </h3>
                </div>

                <ScrollArea className="w-full whitespace-nowrap pb-4">
                  <div className="flex w-max space-x-4 p-1">
                    {/* The "None" Selection Card */}
                    <div
                      onClick={() =>
                        setSelections((prev) => ({ ...prev, [group.id]: null }))
                      }
                      className={`
                        relative cursor-pointer overflow-hidden rounded-xl border-2 transition-all duration-200
                        hover:scale-105 active:scale-95
                        w-28 h-28 flex-shrink-0 bg-muted/50 flex flex-col items-center justify-center
                        ${selections[group.id] === null ? "border-primary ring-2 ring-primary/30 bg-primary/5" : "border-transparent hover:border-primary/50"}
                      `}
                    >
                      <div className="w-10 h-10 rounded-full border-2 border-dashed border-muted-foreground/50 flex items-center justify-center mb-2">
                        <span className="text-muted-foreground/50 text-xl block leading-none rotate-45">
                          +
                        </span>
                      </div>
                      <span className="text-sm font-medium text-muted-foreground">
                        {t("None", "Keine")}
                      </span>

                      {/* Selected indicator */}
                      {selections[group.id] === null && (
                        <div className="absolute top-1 right-1 bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center shadow-sm">
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="3"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            className="w-3 h-3"
                          >
                            <polyline points="20 6 9 17 4 12"></polyline>
                          </svg>
                        </div>
                      )}
                    </div>

                    {/* Character Cards */}
                    {characters.map((char) => {
                      const isSelected =
                        selections[group.id as keyof CharacterSelections] ===
                        char;
                      // We start with the first item in the fallbacks array
                      const initialThumbUrl = `http://localhost:8000/assets/${char}/${group.fallbacks[0]}`;

                      return (
                        <div
                          key={char}
                          onClick={() => toggleSelection(group.id, char)}
                          className={`
                            relative cursor-pointer overflow-hidden rounded-xl border-2 transition-all duration-200
                            hover:scale-105 active:scale-95
                            w-28 h-28 flex-shrink-0 bg-muted/30
                            ${isSelected ? "border-primary ring-2 ring-primary/30" : "border-transparent hover:border-primary/50"}
                          `}
                        >
                          <div className="absolute inset-0 p-2 flex items-center justify-center">
                            <img
                              src={initialThumbUrl}
                              alt={`${char} ${t(group.labelEn, group.labelDe)}`}
                              className="max-w-full max-h-full object-contain drop-shadow-md"
                              data-fallback-index="0"
                              onError={(e) => {
                                const target = e.target as HTMLImageElement;
                                let currentIndex = parseInt(
                                  target.dataset.fallbackIndex || "0",
                                  10,
                                );
                                currentIndex++;

                                if (currentIndex < group.fallbacks.length) {
                                  target.dataset.fallbackIndex =
                                    currentIndex.toString();
                                  target.src = `http://localhost:8000/assets/${char}/${group.fallbacks[currentIndex]}`;
                                } else {
                                  target.style.display = "none";
                                }
                              }}
                            />
                          </div>

                          {/* Label overlay (commented out per user request)
                          <div className="absolute bottom-0 inset-x-0 bg-background/80 backdrop-blur-sm p-1.5 text-center">
                            <p className="text-[10px] font-medium leading-tight truncate">
                              {char.replace(/_/g, " ")}
                            </p>
                          </div>
                          */}

                          {/* Selected indicator */}
                          {isSelected && (
                            <div className="absolute top-1 right-1 bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center shadow-sm">
                              <svg
                                xmlns="http://www.w3.org/2000/svg"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="3"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                className="w-3 h-3"
                              >
                                <polyline points="20 6 9 17 4 12"></polyline>
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

        <div className="mt-4 pt-4">
          <Button
            size="xl"
            className="h-14 cta-step-3 w-full text-base text-white md:text-lg"
            onClick={handleNext}
            disabled={!displayResource || isProcessing}
          >
            {t("Continue to Backgrounds", "Weiter zu Hintergruenden")}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default SilhouettePage;
