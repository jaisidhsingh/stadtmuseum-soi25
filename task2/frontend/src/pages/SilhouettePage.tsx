import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { toast } from "@/hooks/use-toast";
import { ArrowLeft } from "lucide-react";

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
  label: string;
  fallbacks: string[];
};

const PART_GROUPS: PartGroup[] = [
  { id: "hat", label: "Head Accessories", fallbacks: ["hat_thumbnail.png", "hat.png", "head.png"] },
  { id: "arms", label: "Arms", fallbacks: ["arms_thumbnail.png", "right_forearm.png", "left_forearm.png", "right_upper_arm.png"] },
  { id: "legs", label: "Legs", fallbacks: ["legs_thumbnail.png", "right_calf.png", "left_calf.png", "right_thigh.png"] },
  { id: "feet", label: "Feet", fallbacks: ["feet_thumbnail.png", "right_foot.png", "left_foot.png"] },
];

const SilhouettePage = () => {
  const navigate = useNavigate();
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [originalResource, setOriginalResource] = useState<ImageResource | null>(null);
  const [warpedResource, setWarpedResource] = useState<ImageResource | null>(null);

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

  // Derive API payload parts
  const selectedParts = Object.entries(selections)
    .filter(([_, characterName]) => characterName !== null)
    .map(([group]) => group);

  const characterMapping = Object.fromEntries(
    Object.entries(selections).filter(([_, char]) => char !== null)
  );

  const displayResource = selectedParts.length > 0 && warpedResource ? warpedResource : originalResource;
  const previewImage = displayResource ? `http://localhost:8000${displayResource.url}` : null;

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
      .then(res => res.json())
      .then(data => setCharacters(data))
      .catch(err => console.error("Failed to fetch characters", err));
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
        formData.append("use_classical_warping", "true");

        const res = await fetch("http://localhost:8000/segment", {
          method: "POST",
          body: formData,
        });
        if (!res.ok) throw new Error("Segment failed");
        const data = await res.json();
        if (!cancelled) setOriginalResource(data.original);
      } catch (error) {
        console.error("Error fetching original:", error);
        if (!cancelled) {
          toast({
            title: "Processing Failed",
            description: "Make sure the backend is running.",
            variant: "destructive",
          });
        }
      } finally {
        if (!cancelled) setIsLoadingOriginal(false);
      }
    };

    fetchOriginal();
    return () => { cancelled = true; };
  }, [capturedImage]);

  // Effect 2: fetch warped silhouette whenever selections change (debounced)
  useEffect(() => {
    if (!capturedImage || !originalResource) return;

    if (selectedParts.length === 0) {
      setWarpedResource(null);
      return;
    }

    let cancelled = false;
    const timeoutId = setTimeout(async () => {
      setIsLoadingWarped(true);
      try {
        const formData = new FormData();
        formData.append("image", dataURLtoFile(capturedImage, "capture.png"));
        formData.append("use_classical_warping", "true");
        formData.append("parts_to_warp", selectedParts.join(","));
        formData.append("character_mapping", JSON.stringify(characterMapping));

        // Use a default base name to avoid errors, though mapping overrides it
        formData.append("base_name", characters[0] || "Prince_Achmed");

        const res = await fetch("http://localhost:8000/segment", {
          method: "POST",
          body: formData,
        });
        if (!res.ok) throw new Error("Segment failed");
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
  }, [selections, capturedImage, originalResource, characters]);

  const toggleSelection = (group: keyof CharacterSelections, charName: string) => {
    setSelections(prev => ({
      ...prev,
      [group]: prev[group] === charName ? null : charName
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

  return (
    <div className="min-h-screen bg-background relative flex flex-col md:flex-row gap-8 p-4 pt-20 md:p-8 md:pt-8 w-full">
      {/* Top Bar with Back Button */}
      <div className="absolute top-4 left-4 md:top-8 md:left-8 z-20">
        <Button
          variant="outline"
          size="lg"
          className="h-14 px-6 text-lg rounded-xl shadow-sm hover:shadow-md transition-all flex items-center gap-2 bg-background/80 backdrop-blur-sm"
          onClick={() => navigate("/camera")}
        >
          <ArrowLeft className="w-6 h-6" /> Re-take Photo
        </Button>
      </div>

      {/* Left Half: Large Captured Image / Preview */}
      <div className="flex-1 flex items-center justify-center bg-muted/20 rounded-lg p-4 relative overflow-hidden min-h-[50vh] md:min-h-full">
        {isProcessing && (
          <div className="absolute inset-0 bg-background/50 flex flex-col items-center justify-center z-10 backdrop-blur-sm">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
            <p className="text-lg font-medium">Applying styles...</p>
          </div>
        )}

        {previewImage ? (
          <img
            src={previewImage}
            alt="Segmented Silhouette"
            className="max-h-[80vh] w-auto object-contain rounded-lg shadow-lg"
          />
        ) : capturedImage ? (
          <img
            src={capturedImage}
            alt="Captured Silhouette"
            className="max-h-[80vh] w-auto object-contain rounded-lg shadow-lg opacity-50 grayscale"
          />
        ) : (
          <p className="text-muted-foreground">Loading image...</p>
        )}
      </div>

      {/* Right Half: Style Options */}
      <div className="w-full md:w-[500px] flex flex-col h-full shrink-0 z-10 pt-4 md:pt-0">
        <h2 className="text-4xl font-bold mb-2">Step 2: Stylize</h2>
        <p className="text-lg text-primary font-medium mb-6 animate-pulse">
          Tap the thumbnails below to apply character styles.
        </p>

        <ScrollArea className="flex-1 pr-4 -mr-4 pb-20">
          <div className="space-y-8">
            {PART_GROUPS.map((group) => (
              <div key={group.id} className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-2xl font-semibold">{group.label}</h3>
                </div>

                <ScrollArea className="w-full whitespace-nowrap pb-4">
                  <div className="flex w-max space-x-4 p-1">
                    {/* The "None" Selection Card */}
                    <div
                      onClick={() => setSelections(prev => ({ ...prev, [group.id]: null }))}
                      className={`
                        relative cursor-pointer overflow-hidden rounded-xl border-2 transition-all duration-200
                        hover:scale-105 active:scale-95
                        w-28 h-28 flex-shrink-0 bg-muted/50 flex flex-col items-center justify-center
                        ${selections[group.id] === null ? 'border-primary ring-2 ring-primary/30 bg-primary/5' : 'border-transparent hover:border-primary/50'}
                      `}
                    >
                      <div className="w-10 h-10 rounded-full border-2 border-dashed border-muted-foreground/50 flex items-center justify-center mb-2">
                        <span className="text-muted-foreground/50 text-xl block leading-none rotate-45">+</span>
                      </div>
                      <span className="text-sm font-medium text-muted-foreground">None</span>

                      {/* Selected indicator */}
                      {selections[group.id] === null && (
                        <div className="absolute top-1 right-1 bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center shadow-sm">
                          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="w-3 h-3"><polyline points="20 6 9 17 4 12"></polyline></svg>
                        </div>
                      )}
                    </div>

                    {/* Character Cards */}
                    {characters.map((char) => {
                      const isSelected = selections[group.id as keyof CharacterSelections] === char;
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
                            ${isSelected ? 'border-primary ring-2 ring-primary/30' : 'border-transparent hover:border-primary/50'}
                          `}
                        >
                          <div className="absolute inset-0 p-2 flex items-center justify-center">
                            <img
                              src={initialThumbUrl}
                              alt={`${char} ${group.label}`}
                              className="max-w-full max-h-full object-contain drop-shadow-md"
                              data-fallback-index="0"
                              onError={(e) => {
                                const target = e.target as HTMLImageElement;
                                let currentIndex = parseInt(target.dataset.fallbackIndex || "0", 10);
                                currentIndex++;

                                if (currentIndex < group.fallbacks.length) {
                                  target.dataset.fallbackIndex = currentIndex.toString();
                                  target.src = `http://localhost:8000/assets/${char}/${group.fallbacks[currentIndex]}`;
                                } else {
                                  target.style.display = 'none';
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
                              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="w-3 h-3"><polyline points="20 6 9 17 4 12"></polyline></svg>
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

        <div className="pt-6 mt-auto border-t bg-background">
          <Button
            size="lg"
            className="w-full text-lg h-14"
            onClick={handleNext}
            disabled={!displayResource || isProcessing}
          >
            Continue to Backgrounds
          </Button>
        </div>
      </div>
    </div>
  );
};

export default SilhouettePage;
