import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Card } from "@/components/ui/card";
import { toast } from "@/hooks/use-toast";

type ImageResource = {
  id: string;
  url: string;
};

const SilhouettePage = () => {
  const navigate = useNavigate();
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [originalResource, setOriginalResource] =
    useState<ImageResource | null>(null);
  const [warpedResource, setWarpedResource] = useState<ImageResource | null>(
    null,
  );
  const [isLoadingOriginal, setIsLoadingOriginal] = useState(false);
  const [isLoadingWarped, setIsLoadingWarped] = useState(false);

  const [styles, setStyles] = useState({
    arms: false,
    legs: false,
    // head: false,
    feet: false,
  });

  const isProcessing = isLoadingOriginal || isLoadingWarped;

  // Maps each UI checkbox to the backend part names it represents
  const PART_EXPANSION: Record<string, string[]> = {
    arms: ["arms", "hands"],
    legs: ["legs", "torso"],
    // head: ["head", "neck"],
    feet: ["feet"],
  };

  // Derived: which UI checkboxes are checked
  const selectedParts = Object.entries(styles)
    .filter(([_, v]) => v)
    .map(([k]) => k);

  // Expanded backend part names from all checked checkboxes
  const expandedParts = selectedParts.flatMap((k) => PART_EXPANSION[k] ?? [k]);

  // What to show: warped when parts are selected and warping succeeded, otherwise original
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
      setOriginalResource(null);
      setWarpedResource(null);
      try {
        const formData = new FormData();
        formData.append("image", dataURLtoFile(capturedImage, "capture.png"));
        formData.append("use_classical_warping", "false");

        const res = await fetch("http://localhost:8000/segment", {
          method: "POST",
          body: formData,
        });
        if (!res.ok) throw new Error("Segment failed");
        const data = await res.json();
        if (!cancelled) setOriginalResource(data.original);
      } catch (error) {
        console.error("Error fetching original:", error);
        toast({
          title: "Processing Failed",
          description: "Make sure the backend is running.",
          variant: "destructive",
        });
      } finally {
        if (!cancelled) setIsLoadingOriginal(false);
      }
    };

    fetchOriginal();
    return () => {
      cancelled = true;
    };
  }, [capturedImage]);

  // Effect 2: fetch warped silhouette whenever checked parts change (debounced)
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
        formData.append("parts_to_warp", expandedParts.join(","));

        const res = await fetch("http://localhost:8000/segment", {
          method: "POST",
          body: formData,
        });
        if (!res.ok) throw new Error("Segment failed");
        const data = await res.json();
        if (!cancelled) {
          setWarpedResource(
            data.stylized && data.stylized.length > 0 ? data.stylized[0] : null,
          );
        }
      } catch (error) {
        console.error("Error fetching warped:", error);
        if (!cancelled) setWarpedResource(null);
      } finally {
        if (!cancelled) setIsLoadingWarped(false);
      }
    }, 300);

    return () => {
      cancelled = true;
      clearTimeout(timeoutId);
    };
  }, [styles, capturedImage, originalResource]);

  const handleStyleChange = (key: keyof typeof styles) => {
    setStyles((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleNext = () => {
    const resource = displayResource;
    if (resource) {
      sessionStorage.setItem("silhouetteData", JSON.stringify(resource));
      sessionStorage.setItem("silhouetteStyles", JSON.stringify(styles));
      navigate("/selection");
    }
  };

  return (
    <div className="min-h-screen bg-background p-8 flex flex-col md:flex-row gap-8">
      {/* Left Half: Large Captured Image / Preview */}
      <div className="flex-1 flex items-center justify-center bg-muted/20 rounded-lg p-4 relative overflow-hidden">
        {isProcessing && (
          <div className="absolute inset-0 bg-background/50 flex flex-col items-center justify-center z-10 backdrop-blur-sm">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
            <p className="text-lg font-medium">Processing...</p>
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
      <div className="flex-1 flex flex-col justify-center max-w-md mx-auto w-full">
        <Card className="p-8">
          <h2 className="text-2xl font-bold mb-6 text-center">Add Style</h2>

          <div className="space-y-6 mb-8">
            {(["arms", "legs", "feet"] as const).map((part) => (
              <div
                key={part}
                className="flex items-center space-x-3 p-4 border rounded-lg hover:bg-muted/50 transition-colors"
              >
                <Checkbox
                  id={part}
                  checked={styles[part as keyof typeof styles]}
                  onCheckedChange={() =>
                    handleStyleChange(part as keyof typeof styles)
                  }
                  disabled={isProcessing}
                />
                <label
                  htmlFor={part}
                  className="text-lg font-medium capitalize leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer flex-1"
                >
                  {
                    {
                      arms: "Arms",
                      legs: "Legs",
                      // head: "Head & Neck",
                      feet: "Feet",
                    }[part]
                  }
                </label>
              </div>
            ))}
          </div>

          <div className="flex justify-end">
            <Button
              size="lg"
              className="w-full text-lg py-6"
              onClick={handleNext}
              disabled={!displayResource || isProcessing}
            >
              Next
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default SilhouettePage;
