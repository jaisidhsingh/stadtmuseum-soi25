import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Check } from "lucide-react";

type Background = {
  id: string;
  title: string;
  url: string;
};

type SilhouetteData = {
  id: string;
  url: string;
};

const SelectionPage = () => {
  const navigate = useNavigate();
  const [selectedCards, setSelectedCards] = useState<string[]>([]);
  const [backgrounds, setBackgrounds] = useState<Background[]>([]);
  const [silhouette, setSilhouette] = useState<SilhouetteData | null>(null);

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
        }
      } catch (e) {
        console.error("Failed to fetch backgrounds", e);
      }
    };
    fetchBackgrounds();
  }, [navigate]);

  const toggleSelection = (id: string) => {
    setSelectedCards((prev) =>
      prev.includes(id)
        ? prev.filter((item) => item !== id)
        : [...prev, id]
    );
  };

  const handleProceed = () => {
    sessionStorage.setItem("selectedBackgroundIds", JSON.stringify(selectedCards));
    // Also save the background objects for easy reading on the next page
    const selectedBackgrounds = backgrounds.filter(bg => selectedCards.includes(bg.id));
    sessionStorage.setItem("selectedBackgrounds", JSON.stringify(selectedBackgrounds));
    navigate("/confirmation");
  };

  return (
    <div className="h-screen bg-background p-4 flex">
      {/* Silhouette (single) preview - Left Panel */}
      <div className="w-64 flex-shrink-0 border-r pr-6 mr-6 flex flex-col">
        <h3 className="text-xl font-semibold mb-4">Silhouette</h3>
        <Card className="p-4 flex flex-col items-center">
          {silhouette ? (
            <img
              src={`http://localhost:8000${silhouette.url}`}
              alt="Silhouette"
              className="w-full h-auto rounded"
            />
          ) : (
            <div className="w-full aspect-square rounded bg-muted animate-pulse" />
          )}
          <p className="text-lg text-center mt-2 font-medium">Your Outline</p>
        </Card>
      </div>

      {/* Main content - Right Panel */}
      <div className="flex-1 flex flex-col min-h-0 relative">
        {/* Gallery Section - Scrollable Area */}
        <div className="flex-1 overflow-y-auto pb-28 p-3">
          <h3 className="text-xl font-semibold mb-4">Gallery</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {backgrounds.map((item) => (
              <Card
                key={item.id}
                className={`cursor-pointer transition-all hover:shadow-md relative overflow-hidden group ${selectedCards.includes(item.id) ? "ring-2 ring-primary" : ""
                  }`}
                onClick={() => toggleSelection(item.id)}
              >
                <div
                  className="aspect-[4/3] w-full bg-muted bg-cover bg-center"
                  style={{ backgroundImage: `url(http://localhost:8000${item.url})` }}
                />
                <div className="p-3">
                  <p className="font-medium truncate">{item.title}</p>
                </div>
                {selectedCards.includes(item.id) && (
                  <div className="absolute top-2 right-2 bg-primary text-primary-foreground rounded-full p-1 shadow-sm">
                    <Check className="w-4 h-4" />
                  </div>
                )}
              </Card>
            ))}
          </div>
        </div>

        {/* Footer Area with Count and Next Button */}
        <div className="absolute bottom-0 right-0 p-2 bg-background/80 backdrop-blur-sm w-full flex justify-end items-center gap-6 border-t pb-12">
          <span className="text-lg font-medium text-muted-foreground">
            {selectedCards.length} selected
          </span>
          <Button
            size="lg"
            className="text-lg px-8 py-6"
            onClick={handleProceed}
            disabled={selectedCards.length === 0}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  );
};

export default SelectionPage;

