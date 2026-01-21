import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Check } from "lucide-react";

// Mock data for silhouettes and backgrounds
const SILHOUETTES = [
  { id: "s1", name: "Classic", color: "#1a1a2e" },
  { id: "s2", name: "Ornate", color: "#16213e" },
  { id: "s3", name: "Minimal", color: "#0f3460" },
  { id: "s4", name: "Dramatic", color: "#1a1a1a" },
];

const BACKGROUNDS = [
  { id: "b1", name: "Desert Palace", colors: ["#e94560", "#ff6b6b", "#feca57", "#48dbfb"] },
  { id: "b2", name: "Moonlit Garden", colors: ["#5f27cd", "#341f97", "#2e86de", "#54a0ff"] },
  { id: "b3", name: "Flying Carpet", colors: ["#ff9f43", "#ee5a24", "#f8b739", "#ffeaa7"] },
  { id: "b4", name: "Magic Cave", colors: ["#00d2d3", "#01a3a4", "#10ac84", "#1dd1a1"] },
  { id: "b5", name: "Sunset Oasis", colors: ["#ff6b6b", "#feca57", "#ff9ff3", "#f368e0"] },
  { id: "b6", name: "Starry Night", colors: ["#222f3e", "#576574", "#8395a7", "#c8d6e5"] },
  { id: "b7", name: "Golden Temple", colors: ["#ffd32a", "#ff9f1a", "#ff793f", "#cd6133"] },
  { id: "b8", name: "Enchanted Forest", colors: ["#26de81", "#20bf6b", "#0fb9b1", "#2bcbba"] },
];

// Quadrant colors
const QUADRANT_COLORS = ["#e94560", "#feca57", "#48dbfb", "#26de81"];

type Selection = {
  silhouetteId: string;
  backgroundId: string;
  quadrantIndex: number;
};

const SelectionPage = () => {
  const navigate = useNavigate();
  const [selectedSilhouette, setSelectedSilhouette] = useState(SILHOUETTES[0].id);
  const [selectedBackground, setSelectedBackground] = useState(BACKGROUNDS[0].id);
  const [selections, setSelections] = useState<Selection[]>([]);

  const currentBackground = BACKGROUNDS.find((b) => b.id === selectedBackground);

  const isSelected = (silhouetteId: string, backgroundId: string, quadrantIndex: number) => {
    return selections.some(
      (s) =>
        s.silhouetteId === silhouetteId &&
        s.backgroundId === backgroundId &&
        s.quadrantIndex === quadrantIndex
    );
  };

  const toggleSelection = (quadrantIndex: number) => {
    const selection: Selection = {
      silhouetteId: selectedSilhouette,
      backgroundId: selectedBackground,
      quadrantIndex,
    };

    const existingIndex = selections.findIndex(
      (s) =>
        s.silhouetteId === selection.silhouetteId &&
        s.backgroundId === selection.backgroundId &&
        s.quadrantIndex === selection.quadrantIndex
    );

    if (existingIndex >= 0) {
      setSelections(selections.filter((_, i) => i !== existingIndex));
    } else {
      setSelections([...selections, selection]);
    }
  };

  const proceedToConfirmation = () => {
    sessionStorage.setItem("selections", JSON.stringify(selections));
    sessionStorage.setItem("silhouettes", JSON.stringify(SILHOUETTES));
    sessionStorage.setItem("backgrounds", JSON.stringify(BACKGROUNDS));
    navigate("/confirmation");
  };

  return (
    <div className="h-screen bg-background p-4 pb-16 flex overflow-hidden">
      {/* Left sidebar - Silhouettes */}
      <div className="w-24 flex-shrink-0 mr-4">
        <h3 className="text-sm font-semibold mb-2">Silhouettes</h3>
        <div className="space-y-2">
          {SILHOUETTES.map((silhouette) => (
            <Card
              key={silhouette.id}
              className={`p-2 cursor-pointer transition-all ${
                selectedSilhouette === silhouette.id
                  ? "ring-2 ring-primary"
                  : "hover:ring-1 hover:ring-muted-foreground"
              }`}
              onClick={() => setSelectedSilhouette(silhouette.id)}
            >
              <div
                className="w-full aspect-square rounded"
                style={{ backgroundColor: silhouette.color }}
              />
              <p className="text-xs text-center mt-1">{silhouette.name}</p>
            </Card>
          ))}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Quadrants */}
        <div className="flex-1 grid grid-cols-2 gap-2 mb-4">
          {QUADRANT_COLORS.map((color, index) => (
            <Card
              key={index}
              className={`relative cursor-pointer transition-all hover:scale-[1.02] ${
                isSelected(selectedSilhouette, selectedBackground, index)
                  ? "ring-2 ring-primary"
                  : ""
              }`}
              style={{ backgroundColor: currentBackground?.colors[index] || color }}
              onClick={() => toggleSelection(index)}
            >
              <div className="absolute inset-0 flex items-center justify-center">
                <div
                  className="w-1/2 h-3/4 rounded opacity-80"
                  style={{
                    backgroundColor: SILHOUETTES.find((s) => s.id === selectedSilhouette)?.color,
                  }}
                />
              </div>
              {isSelected(selectedSilhouette, selectedBackground, index) && (
                <div className="absolute top-2 right-2 bg-primary text-primary-foreground rounded-full p-1">
                  <Check className="w-4 h-4" />
                </div>
              )}
              <div className="absolute bottom-2 left-2 text-xs font-semibold text-white drop-shadow">
                Quadrant {index + 1}
              </div>
            </Card>
          ))}
        </div>

        {/* Bottom carousel - Backgrounds */}
        <div>
          <h3 className="text-sm font-semibold mb-2">Backgrounds</h3>
          <ScrollArea className="w-full whitespace-nowrap">
            <div className="flex space-x-2 pb-2">
              {BACKGROUNDS.map((bg) => (
                <Card
                  key={bg.id}
                  className={`flex-shrink-0 w-24 p-2 cursor-pointer transition-all ${
                    selectedBackground === bg.id
                      ? "ring-2 ring-primary"
                      : "hover:ring-1 hover:ring-muted-foreground"
                  }`}
                  onClick={() => setSelectedBackground(bg.id)}
                >
                  <div className="grid grid-cols-2 gap-1 aspect-square">
                    {bg.colors.map((color, i) => (
                      <div
                        key={i}
                        className="rounded-sm"
                        style={{ backgroundColor: color }}
                      />
                    ))}
                  </div>
                  <p className="text-xs text-center mt-1 truncate">{bg.name}</p>
                </Card>
              ))}
            </div>
            <ScrollBar orientation="horizontal" />
          </ScrollArea>
        </div>

        {/* Selection count and proceed button */}
        <div className="flex items-center justify-between mt-4 pt-4 border-t">
          <p className="text-sm">
            {selections.length} selected
          </p>
          <Button
            disabled={selections.length === 0}
            onClick={proceedToConfirmation}
          >
            Send Selection to Email
          </Button>
        </div>
      </div>
    </div>
  );
};

export default SelectionPage;
