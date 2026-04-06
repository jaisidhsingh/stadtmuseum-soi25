import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Check, ArrowLeft, Circle } from "lucide-react";

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

const SelectionPage = () => {
  const navigate = useNavigate();
  const [selectedCards, setSelectedCards] = useState<string[]>([]);
  const [backgrounds, setBackgrounds] = useState<Background[]>([]);
  const [previewBgId, setPreviewBgId] = useState<string | null>(null);

  const [silhouette, setSilhouette] = useState<SilhouetteData | null>(null);

  // Track the currently previewed background in the large main view

  useEffect(() => {
    // Load silhouette from session storage
    const storedSil = sessionStorage.getItem("silhouetteData");
    if (storedSil) {
      setSilhouette(JSON.parse(storedSil));
    } else {
      navigate("/camera");
    }

    // Fetch backgrounds
    const fetchBackgrounds = async () => {
      try {
        const res = await fetch("http://localhost:8000/backgrounds");
        if (res.ok) {
          const data = await res.json();
          setBackgrounds(data);
          if (data.length > 0) {
            setPreviewBgId(data[0].id); // Preview first background by default
          }
        }
      } catch (e) {
        console.error("Failed to fetch backgrounds", e);
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

    // We intentionally ignore `preComposites` now,
    // relying on ConfirmationPage to do the actual backend heavy-lifting rendering.
    sessionStorage.removeItem("preComposites");

    navigate("/confirmation");
  };

  const currentPreviewBg = backgrounds.find((b) => b.id === previewBgId);
  const isCurrentPreviewSelected = previewBgId
    ? selectedCards.includes(previewBgId)
    : false;

  return (
    <div className="h-screen bg-background relative flex flex-col p-4 md:p-8">
      {/* Top Bar with Back Button */}
      <div className="absolute top-4 left-4 md:top-8 md:left-8 z-20">
        <Button
          variant="outline"
          size="lg"
          className="h-14 px-6 text-lg rounded-xl shadow-sm hover:shadow-md transition-all flex items-center gap-2 bg-background/80 backdrop-blur-sm"
          onClick={() => navigate("/silhouette")}
        >
          <ArrowLeft className="w-6 h-6" /> Back to Stylize
        </Button>
      </div>

      {/* Header text */}
      <div className="w-full text-center pt-16 md:pt-0 mb-4 z-10 shrink-0">
        <h1 className="text-4xl font-bold text-center mb-4">
          Step 3: Choose Your Scenes
        </h1>
        <p className="text-center text-xl text-primary font-medium mb-8 animate-pulse shadow-sm">
          Tap the circle on a preview to select it for export. You can choose
          multiple!
        </p>
      </div>

      {/* Large Main Preview (Top Center) */}
      <div className="flex-1 min-h-0 flex items-center justify-center mb-6 relative z-10 w-full px-4">
        {currentPreviewBg && silhouette ? (
          <div
            className={`
              relative w-full max-w-5xl max-h-full flex items-center justify-center rounded-2xl p-0 transition-all
              ${isCurrentPreviewSelected ? "scale-[1.01]" : ""}
            `}
            onClick={() => toggleSelection(currentPreviewBg.id)}
          >
            {/* The proportionate container ensuring math is accurate */}
            <div
              className={`relative h-full w-full max-h-[60vh] max-w-full rounded-xl overflow-hidden shadow-2xl bg-muted border-4 transition-all
                ${isCurrentPreviewSelected ? "border-primary ring-4 ring-primary/20" : "border-transparent group-hover:border-primary/50"}
              `}
              style={{
                aspectRatio: `${currentPreviewBg.bg_w || 1920} / ${currentPreviewBg.bg_h || 1080}`,
              }}
            >
              {/* Background Layer */}
              <img
                src={`http://localhost:8000${currentPreviewBg.url}`}
                alt={currentPreviewBg.title}
                className="absolute inset-0 w-full h-full object-contain pointer-events-none"
              />

              {/* Silhouette Layer */}
              {(() => {
                const bw = currentPreviewBg.bg_w || 1920;
                const bh = currentPreviewBg.bg_h || 1080;
                const pos = currentPreviewBg.positions?.[0] || [bw / 2, bh];
                const mw = currentPreviewBg.max_w;
                const mh = currentPreviewBg.max_h;
                const tint = currentPreviewBg.silhouette_color || [0, 0, 0];
                const tintCss = `rgb(${tint[0]}, ${tint[1]}, ${tint[2]})`;
                const silhouetteUrl = `http://localhost:8000${silhouette.url}`;
                const widthPct = mw ? `${(mw / bw) * 100}%` : "30%";
                const heightPct = mh ? `${(mh / bh) * 100}%` : "60%";

                return (
                  <div
                    aria-label="My Silhouette"
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

              {/* Selection Text overlay instructing to tap */}
              {!isCurrentPreviewSelected && (
                <div className="absolute inset-x-0 top-6 text-center opacity-0 hover:opacity-100 transition-opacity pointer-events-none">
                  <span className="bg-background/80 backdrop-blur px-6 py-2 rounded-full font-medium shadow-lg">
                    Tap image to select for export
                  </span>
                </div>
              )}

              {/* Radio / Checkmark overlay for selection state */}
              {isCurrentPreviewSelected ? (
                <div className="absolute top-4 right-4 bg-primary text-primary-foreground rounded-full p-2 shadow-xl animate-in zoom-in pointer-events-none">
                  <Check className="w-8 h-8" />
                </div>
              ) : (
                <div className="absolute top-4 right-4 bg-black/40 text-white rounded-full p-2 shadow-xl hover:bg-black/60 transition-colors">
                  <Circle className="w-8 h-8" />
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="w-full max-w-4xl aspect-[4/3] bg-muted/50 rounded-2xl animate-pulse flex items-center justify-center">
            <span className="text-muted-foreground text-xl">
              Loading preview...
            </span>
          </div>
        )}
      </div>

      {/* Bottom Horizontal Thumbnails Strip (Native scroll so touch devices can swipe inside it easily) */}
      <div className="w-full shrink-0 bg-muted/10 rounded-xl p-4 border relative z-10">
        <div className="w-full overflow-x-auto touch-pan-x snap-x snap-mandatory flex space-x-4 pb-2 items-center custom-scrollbar">
          {backgrounds.map((bg) => {
            const selected = selectedCards.includes(bg.id);
            const isPreviewing = previewBgId === bg.id;

            return (
              <div
                key={bg.id}
                className="snap-start flex flex-col items-center gap-2 group cursor-pointer"
                onClick={() => {
                  // If it's already the preview, clicking it toggles selection.
                  // If it's not the preview, the first click makes it the preview.
                  if (isPreviewing) {
                    toggleSelection(bg.id);
                  } else {
                    setPreviewBgId(bg.id);
                  }
                }}
              >
                <div
                  className={`
                      relative w-40 h-30 flex-shrink-0 rounded-lg overflow-hidden border-4 transition-all duration-300
                      ${selected ? "border-primary ring-2 ring-primary/30" : "border-transparent group-hover:border-primary/50"}
                      ${isPreviewing ? "shadow-lg scale-105" : "opacity-70 group-hover:opacity-100"}
                    `}
                >
                  <img
                    src={`http://localhost:8000${bg.url}`}
                    alt={bg.title}
                    className="w-full h-full object-cover"
                  />
                  {/* Radio / Checkmark overlay */}
                  {selected ? (
                    <div className="absolute top-1 right-1 bg-primary text-primary-foreground rounded-full p-1 shadow-sm animate-in zoom-in">
                      <Check className="w-4 h-4" />
                    </div>
                  ) : (
                    <div className="absolute top-1 right-1 bg-black/40 text-white rounded-full p-1 shadow-sm transition-colors group-hover:bg-black/60">
                      <Circle className="w-4 h-4" />
                    </div>
                  )}
                </div>
                <span
                  className={`text-sm font-medium ${isPreviewing ? "text-primary scale-110 mt-1 transition-transform font-bold" : "text-muted-foreground group-hover:text-foreground"}`}
                >
                  {bg.title}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Action Bar (Now static below to prevent overlapping) */}
      <div className="w-full shrink-0 py-4 flex justify-end items-center z-30 mt-auto border-t">
        <div className="flex items-center gap-6 bg-background rounded-full shadow-sm p-2 pl-6">
          <div className="flex flex-col items-end mr-2">
            <span className="text-xl font-bold text-primary">
              {selectedCards.length}
            </span>
            <span className="text-xs text-muted-foreground uppercase tracking-wider font-semibold">
              Selected
            </span>
          </div>
          <Button
            size="lg"
            className="h-16 px-10 text-xl font-bold rounded-xl shadow-lg hover:shadow-xl transition-all"
            onClick={handleProceed}
            disabled={selectedCards.length === 0}
          >
            Review & Export
          </Button>
        </div>
      </div>
    </div>
  );
};

export default SelectionPage;
