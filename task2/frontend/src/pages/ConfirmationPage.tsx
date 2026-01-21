import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Check, X, ZoomIn } from "lucide-react";
import { toast } from "@/hooks/use-toast";

type Selection = {
  silhouetteId: string;
  backgroundId: string;
  quadrantIndex: number;
};

type Silhouette = {
  id: string;
  name: string;
  color: string;
};

type Background = {
  id: string;
  name: string;
  colors: string[];
};

const ConfirmationPage = () => {
  const navigate = useNavigate();
  const [selections, setSelections] = useState<Selection[]>([]);
  const [silhouettes, setSilhouettes] = useState<Silhouette[]>([]);
  const [backgrounds, setBackgrounds] = useState<Background[]>([]);
  const [email, setEmail] = useState("");
  const [showSuccess, setShowSuccess] = useState(false);
  const [enlargedImage, setEnlargedImage] = useState<{
    silhouette: Silhouette;
    background: Background;
    quadrantIndex: number;
  } | null>(null);

  useEffect(() => {
    const storedSelections = sessionStorage.getItem("selections");
    const storedSilhouettes = sessionStorage.getItem("silhouettes");
    const storedBackgrounds = sessionStorage.getItem("backgrounds");

    if (storedSelections) setSelections(JSON.parse(storedSelections));
    if (storedSilhouettes) setSilhouettes(JSON.parse(storedSilhouettes));
    if (storedBackgrounds) setBackgrounds(JSON.parse(storedBackgrounds));
  }, []);

  const toggleSelection = (selection: Selection) => {
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

  const isSelected = (silhouetteId: string, backgroundId: string, quadrantIndex: number) => {
    return selections.some(
      (s) =>
        s.silhouetteId === silhouetteId &&
        s.backgroundId === backgroundId &&
        s.quadrantIndex === quadrantIndex
    );
  };

  const getSelectionsForSilhouette = (silhouetteId: string) => {
    return selections.filter((s) => s.silhouetteId === silhouetteId);
  };

  const handleSendEmail = async () => {
    if (!email) {
      toast({
        title: "Email required",
        description: "Please enter your email address",
        variant: "destructive",
      });
      return;
    }

    if (selections.length === 0) {
      toast({
        title: "No selections",
        description: "Please select at least one image",
        variant: "destructive",
      });
      return;
    }

    // Mock API call
    console.log("Sending to email:", email, "Selections:", selections);
    
    // Simulate API delay
    await new Promise((resolve) => setTimeout(resolve, 1000));
    
    setShowSuccess(true);
  };

  const handleSuccessClose = () => {
    setShowSuccess(false);
    sessionStorage.clear();
    navigate("/");
  };

  // Get unique silhouettes that have selections or were part of original selection
  const usedSilhouetteIds = [...new Set(selections.map((s) => s.silhouetteId))];
  const usedSilhouettes = silhouettes.filter((s) => usedSilhouetteIds.includes(s.id));

  return (
    <div className="min-h-screen bg-background p-8">
      <h1 className="text-2xl font-bold text-center mb-2">
        Confirm Your Selections
      </h1>
      <p className="text-center text-muted-foreground mb-8">
        Review and modify your selections before sending
      </p>

      <div className="max-w-6xl mx-auto space-y-8">
        {usedSilhouettes.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-muted-foreground">No selections made</p>
            <Button onClick={() => navigate("/selection")} className="mt-4">
              Go Back to Selection
            </Button>
          </div>
        ) : (
          usedSilhouettes.map((silhouette) => (
            <div key={silhouette.id} className="space-y-4">
              <div className="flex items-center gap-4">
                <div
                  className="w-12 h-12 rounded"
                  style={{ backgroundColor: silhouette.color }}
                />
                <h2 className="text-lg font-semibold">{silhouette.name} Silhouette</h2>
                <span className="text-sm text-muted-foreground">
                  ({getSelectionsForSilhouette(silhouette.id).length} selected)
                </span>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {backgrounds.map((background) =>
                  [0, 1, 2, 3].map((quadrantIndex) => {
                    const selected = isSelected(silhouette.id, background.id, quadrantIndex);
                    return (
                      <Card
                        key={`${background.id}-${quadrantIndex}`}
                        className={`relative cursor-pointer transition-all hover:scale-105 ${
                          selected ? "ring-2 ring-primary" : "opacity-60 hover:opacity-100"
                        }`}
                        onClick={() =>
                          toggleSelection({
                            silhouetteId: silhouette.id,
                            backgroundId: background.id,
                            quadrantIndex,
                          })
                        }
                      >
                        <div
                          className="aspect-square rounded-t flex items-center justify-center"
                          style={{ backgroundColor: background.colors[quadrantIndex] }}
                        >
                          <div
                            className="w-1/2 h-3/4 rounded opacity-80"
                            style={{ backgroundColor: silhouette.color }}
                          />
                        </div>
                        <div className="p-2 flex items-center justify-between">
                          <span className="text-xs truncate">{background.name} #{quadrantIndex + 1}</span>
                          <button
                            className="p-1 hover:bg-muted rounded"
                            onClick={(e) => {
                              e.stopPropagation();
                              setEnlargedImage({ silhouette, background, quadrantIndex });
                            }}
                          >
                            <ZoomIn className="w-3 h-3" />
                          </button>
                        </div>
                        {selected && (
                          <div className="absolute top-1 right-1 bg-primary text-primary-foreground rounded-full p-1">
                            <Check className="w-3 h-3" />
                          </div>
                        )}
                      </Card>
                    );
                  })
                )}
              </div>
            </div>
          ))
        )}

        {/* Email input and send button */}
        <div className="flex items-center gap-4 pt-8 border-t">
          <Input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="max-w-md"
          />
          <Button onClick={handleSendEmail} disabled={selections.length === 0}>
            Send Selected Images to Email
          </Button>
          <span className="text-sm text-muted-foreground">
            {selections.length} image(s) selected
          </span>
        </div>
      </div>

      {/* Success Dialog */}
      <Dialog open={showSuccess} onOpenChange={handleSuccessClose}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Success!</DialogTitle>
          </DialogHeader>
          <p>Your selected images have been sent to {email}</p>
          <Button onClick={handleSuccessClose} className="mt-4">
            Start Over
          </Button>
        </DialogContent>
      </Dialog>

      {/* Enlarge Dialog */}
      <Dialog open={!!enlargedImage} onOpenChange={() => setEnlargedImage(null)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>
              {enlargedImage?.background.name} - Quadrant {(enlargedImage?.quadrantIndex ?? 0) + 1}
            </DialogTitle>
          </DialogHeader>
          {enlargedImage && (
            <div
              className="aspect-square rounded flex items-center justify-center"
              style={{ backgroundColor: enlargedImage.background.colors[enlargedImage.quadrantIndex] }}
            >
              <div
                className="w-1/2 h-3/4 rounded opacity-80"
                style={{ backgroundColor: enlargedImage.silhouette.color }}
              />
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default ConfirmationPage;
