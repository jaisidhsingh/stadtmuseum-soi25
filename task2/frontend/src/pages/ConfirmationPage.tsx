import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { X, ZoomIn } from "lucide-react";
import { toast } from "@/hooks/use-toast";

type CompositeItem = {
  id: string; // The composite ID returned from backend
  originalBgId: string;
  title: string;
  url: string; // The fully qualified URL returned from backend
};

const ConfirmationPage = () => {
  const navigate = useNavigate();
  const [compositedItems, setCompositedItems] = useState<CompositeItem[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [email, setEmail] = useState("");
  const [showSuccess, setShowSuccess] = useState(false);
  const [enlargedImage, setEnlargedImage] = useState<CompositeItem | null>(null);

  useEffect(() => {
    const silhouetteDataStr = sessionStorage.getItem("silhouetteData");
    const bgIdsStr = sessionStorage.getItem("selectedBackgroundIds");
    const bgObjectsStr = sessionStorage.getItem("selectedBackgrounds");

    if (!silhouetteDataStr || !bgIdsStr) {
      navigate("/selection");
      return;
    }

    const silhouette = JSON.parse(silhouetteDataStr);
    const bgIds: string[] = JSON.parse(bgIdsStr);
    const bgObjects = bgObjectsStr ? JSON.parse(bgObjectsStr) : [];

    const generateComposites = async () => {
      setIsProcessing(true);
      try {
        const promises = bgIds.map(async (bgId) => {
          const res = await fetch("http://localhost:8000/composite", {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ silhouette_id: silhouette.id, background_id: bgId })
          });
          if (!res.ok) throw new Error("Composition failed");
          const data = await res.json();

          const bgObj = bgObjects.find((b: { id: string }) => b.id === bgId);
          const title = bgObj ? bgObj.title : `Background ${bgId}`;

          return {
            id: data.result.id,
            originalBgId: bgId,
            title: title,
            url: `http://localhost:8000${data.result.url}`
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

  const handleSendEmail = async () => {
    if (!email) {
      toast({
        title: "Email required",
        description: "Please enter your email address",
        variant: "destructive",
      });
      return;
    }

    if (compositedItems.length === 0) {
      toast({
        title: "No selections",
        description: "Please keep at least one image to send",
        variant: "destructive",
      });
      return;
    }

    try {
      const res = await fetch("http://localhost:8000/send-email", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email, ids: compositedItems.map(item => item.id) })
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Email failed");
      }

      setShowSuccess(true);
    } catch (e: unknown) {
      toast({
        title: "Error",
        description: e instanceof Error ? e.message : "Failed to send email.",
        variant: "destructive",
      });
    }
  };

  const handleSuccessClose = () => {
    setShowSuccess(false);
    sessionStorage.removeItem("selectedBackgroundIds");
    sessionStorage.removeItem("selectedBackgrounds");
    sessionStorage.removeItem("silhouetteData");
    sessionStorage.removeItem("capturedImage");
    sessionStorage.removeItem("silhouetteStyles");
    navigate("/");
  };

  return (
    <div className="min-h-screen bg-background p-8">
      <h1 className="text-2xl font-bold text-center mb-2">
        Confirm Your Images
      </h1>
      <p className="text-center text-muted-foreground mb-8">
        Review your composited images before sending
      </p>

      <div className="max-w-6xl mx-auto space-y-8">
        {isProcessing ? (
          <div className="text-center py-12">
            <p className="text-muted-foreground text-lg animate-pulse">Generating your composited images...</p>
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
                <div
                  className="aspect-[4/3] rounded-t w-full bg-muted bg-cover bg-center"
                  style={{ backgroundImage: `url(${item.url})` }}
                />

                <div className="p-2 flex items-center justify-between">
                  <span className="text-xs truncate font-medium">{item.title}</span>
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

        {/* Email input and send button */}
        {!isProcessing && compositedItems.length > 0 && (
          <div className="flex items-center gap-4 pt-8 border-t">
            <Input
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="max-w-md"
            />
            <Button onClick={handleSendEmail} disabled={compositedItems.length === 0}>
              Send Selected Images to Email
            </Button>
            <span className="text-sm text-muted-foreground">
              {compositedItems.length} image(s) selected
            </span>
          </div>
        )}
      </div>

      {/* Success Dialog */}
      <Dialog open={showSuccess} onOpenChange={handleSuccessClose}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Success!</DialogTitle>
          </DialogHeader>
          <p>Your beautiful images have been sent to {email}</p>
          <Button onClick={handleSuccessClose} className="mt-4">
            Start Over
          </Button>
        </DialogContent>
      </Dialog>

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
