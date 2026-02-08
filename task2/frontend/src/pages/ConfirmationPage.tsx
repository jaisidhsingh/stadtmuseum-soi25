import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { X, ZoomIn } from "lucide-react";
import { toast } from "@/hooks/use-toast";

type GalleryItem = {
  id: string;
  title: string;
  color: string;
};

const ConfirmationPage = () => {
  const navigate = useNavigate();
  const [selectedItems, setSelectedItems] = useState<GalleryItem[]>([]);
  const [email, setEmail] = useState("");
  const [showSuccess, setShowSuccess] = useState(false);
  const [enlargedImage, setEnlargedImage] = useState<GalleryItem | null>(null);

  useEffect(() => {
    const storedItems = sessionStorage.getItem("selectedGalleryItems");

    if (storedItems) {
      setSelectedItems(JSON.parse(storedItems));
    }
  }, []);

  const removeItem = (id: string) => {
    const updatedItems = selectedItems.filter((item) => item.id !== id);
    setSelectedItems(updatedItems);
    // Update storage to keep state consistent if they refresh
    sessionStorage.setItem("selectedGalleryItems", JSON.stringify(updatedItems));
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

    if (selectedItems.length === 0) {
      toast({
        title: "No selections",
        description: "Please select at least one image",
        variant: "destructive",
      });
      return;
    }

    // Mock API call
    console.log("Sending to email:", email, "Selected Items:", selectedItems);

    // Simulate API delay
    await new Promise((resolve) => setTimeout(resolve, 1000));

    setShowSuccess(true);
  };

  const handleSuccessClose = () => {
    setShowSuccess(false);
    sessionStorage.removeItem("selectedGalleryItems");
    navigate("/");
  };

  return (
    <div className="min-h-screen bg-background p-8">
      <h1 className="text-2xl font-bold text-center mb-2">
        Confirm Your Selections
      </h1>
      <p className="text-center text-muted-foreground mb-8">
        Review and modify your selections before sending
      </p>

      <div className="max-w-6xl mx-auto space-y-8">
        {selectedItems.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-muted-foreground">No selections made</p>
            <Button onClick={() => navigate("/selection")} className="mt-4">
              Go Back to Selection
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {selectedItems.map((item) => (
              <Card
                key={item.id}
                className="relative group transition-all hover:scale-105"
              >
                <div
                  className="aspect-[4/3] rounded-t w-full"
                  style={{ backgroundColor: item.color }}
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
        <div className="flex items-center gap-4 pt-8 border-t">
          <Input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="max-w-md"
          />
          <Button onClick={handleSendEmail} disabled={selectedItems.length === 0}>
            Send Selected Images to Email
          </Button>
          <span className="text-sm text-muted-foreground">
            {selectedItems.length} image(s) selected
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
              {enlargedImage?.title}
            </DialogTitle>
          </DialogHeader>
          {enlargedImage && (
            <div
              className="aspect-[4/3] rounded w-full"
              style={{ backgroundColor: enlargedImage.color }}
            />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default ConfirmationPage;
