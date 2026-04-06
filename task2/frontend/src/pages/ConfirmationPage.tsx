import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { X, ZoomIn, ArrowLeft } from "lucide-react";
import { toast } from "@/hooks/use-toast";

type CompositeItem = {
  id: string; // The composite ID returned from backend
  originalBgId: string;
  title: string;
  url: string; // The fully qualified URL returned from backend
};

type ShareInfo = {
  shareUrl: string;
  expiresAt: string;
};

const ConfirmationPage = () => {
  const navigate = useNavigate();
  const [compositedItems, setCompositedItems] = useState<CompositeItem[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [enlargedImage, setEnlargedImage] = useState<CompositeItem | null>(null);
  const [shareInfo, setShareInfo] = useState<ShareInfo | null>(null);
  const [isCreatingShare, setIsCreatingShare] = useState(false);

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
          const title = bgObj ? bgObj.title : `Background ${bgId}`;

          // Use pre-composited result from SelectionPage if available
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
      } catch (e) {
        console.error(e);
        toast({
          title: "Error",
          description: "Failed to generate some composite images.",
          variant: "destructive",
        });
      } finally {
        setIsProcessing(false);
      }
    };

    generateComposites();
  }, [navigate]);

  const removeItem = (id: string) => {
    const updatedItems = compositedItems.filter((item) => item.id !== id);
    setCompositedItems(updatedItems);
  };

  const handleCreateShare = async () => {
    if (compositedItems.length === 0) {
      toast({
        title: "No images",
        description: "Generate at least one composition first.",
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
          ttl_minutes: 30,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to create share link");
      }

      const data = await res.json();
      setShareInfo({ shareUrl: data.share_url, expiresAt: data.expires_at });
    } catch (e: unknown) {
      toast({
        title: "Share link failed",
        description:
          e instanceof Error ? e.message : "Could not create QR share link.",
        variant: "destructive",
      });
    } finally {
      setIsCreatingShare(false);
    }
  };

  const handleStartOver = async () => {
    try {
      await fetch("http://localhost:8000/clear", { method: "POST" });
    } catch (e) {
      console.error("Failed to clear backend session", e);
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
    <div className="min-h-screen bg-background relative flex flex-col p-4 md:p-8">
      {/* Top Bar with Back Button */}
      <div className="absolute top-4 left-4 md:top-8 md:left-8 z-20">
        <Button
          variant="outline"
          size="lg"
          className="h-14 px-6 text-lg rounded-xl shadow-sm hover:shadow-md transition-all flex items-center gap-2 bg-background/80 backdrop-blur-sm"
          onClick={() => navigate("/selection")}
        >
          <ArrowLeft className="w-6 h-6" /> Back to Selection
        </Button>
      </div>

      <div className="text-center w-full pt-16 md:pt-4 mb-8">
        <h1 className="text-4xl font-bold mb-2">Step 4: Confirm & Export</h1>
        <p className="text-xl text-primary font-medium animate-pulse">
          Review your composited images and export them to a QR share link.
        </p>
      </div>

      <div className="max-w-7xl mx-auto w-full space-y-8 flex-1">
        {isProcessing ? (
          <div className="text-center py-12">
            <p className="text-muted-foreground text-lg animate-pulse">
              Generating your composited images...
            </p>
          </div>
        ) : compositedItems.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-muted-foreground">No selections made or found.</p>
            <Button onClick={() => navigate("/selection")} className="mt-4">
              Go Back to Selection
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {compositedItems.map((item) => (
              <Card
                key={item.id}
                className="relative group transition-all hover:scale-105"
              >
                <div className="aspect-[4/3] rounded-t w-full overflow-hidden bg-muted">
                  <img
                    src={item.url}
                    alt={item.title}
                    className="w-full h-full object-cover"
                  />
                </div>

                <div className="p-2 flex items-center justify-between">
                  <span className="text-xs truncate font-medium">
                    {item.title}
                  </span>
                  <div className="flex gap-1">
                    <button
                      className="p-1 hover:bg-muted rounded text-muted-foreground hover:text-foreground"
                      onClick={() => setEnlargedImage(item)}
                      title="Zoom"
                    >
                      <ZoomIn className="w-3 h-3" />
                    </button>
                    <button
                      className="p-1 hover:bg-destructive/10 rounded text-muted-foreground hover:text-destructive"
                      onClick={() => removeItem(item.id)}
                      title="Remove"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}

        {/* QR export and session actions */}
        {!isProcessing && compositedItems.length > 0 && (
          <div className="grid grid-cols-1 gap-8 pt-12 border-t w-full max-w-6xl mx-auto">
            <div className="flex flex-col gap-4">
              <span className="text-lg text-muted-foreground font-semibold">
                {compositedItems.length} image(s) selected
              </span>
              <div className="flex flex-wrap items-center gap-3">
                <Button
                  size="lg"
                  variant="secondary"
                  onClick={handleCreateShare}
                  disabled={isCreatingShare || compositedItems.length === 0}
                  className="h-16 px-10 text-xl font-bold rounded-xl shadow-lg hover:shadow-xl transition-all"
                >
                  {isCreatingShare
                    ? "Creating QR Export..."
                    : "Export to QR Code"}
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  className="h-16 px-8 text-xl rounded-xl"
                  onClick={handleStartOver}
                >
                  Start Over
                </Button>
              </div>

              {shareInfo && (
                <Card className="p-4 w-full max-w-sm">
                  <div className="flex flex-col items-center gap-3">
                    <img
                      src={`https://api.qrserver.com/v1/create-qr-code/?size=280x280&data=${encodeURIComponent(shareInfo.shareUrl)}`}
                      alt="Share QR Code"
                      className="w-56 h-56"
                    />
                    <p className="text-xs text-muted-foreground text-center break-all">
                      {shareInfo.shareUrl}
                    </p>
                    <p className="text-xs text-muted-foreground text-center">
                      Expires: {new Date(shareInfo.expiresAt).toLocaleString()}
                    </p>
                  </div>
                </Card>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Enlarge Dialog */}
      <Dialog open={!!enlargedImage} onOpenChange={() => setEnlargedImage(null)}>
        <DialogContent className="max-w-2xl px-0 border-none bg-transparent shadow-none">
          {enlargedImage && (
            <img
              src={enlargedImage.url}
              alt={enlargedImage.title}
              className="w-full h-auto rounded"
            />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default ConfirmationPage;
