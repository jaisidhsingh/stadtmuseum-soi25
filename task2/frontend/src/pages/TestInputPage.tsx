import * as React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { t } from "@/lib/localization";
import dummyImage from "../assets/dummy.jpeg";

const TestInputPage = () => {
  const navigate = useNavigate();
  const [imageBase64, setImageBase64] = React.useState<string | null>(null);

  React.useEffect(() => {
    // Convert the imported image URL to base64 so it mimics the camera capture format
    const fetchImageAsBase64 = async () => {
      try {
        const response = await fetch(dummyImage);
        const blob = await response.blob();
        const reader = new FileReader();
        reader.onloadend = () => {
          if (typeof reader.result === "string") {
            setImageBase64(reader.result);
          }
        };
        reader.readAsDataURL(blob);
      } catch (error) {
        console.error("Failed to load test image as base64:", error);
      }
    };

    fetchImageAsBase64();
  }, []);

  const proceedToSelection = () => {
    if (imageBase64) {
      sessionStorage.setItem("capturedImage", imageBase64);
      navigate("/silhouette");
    }
  };

  return (
    <div className="exhibit-shell flex h-full min-h-0 w-full min-w-0 flex-col overflow-hidden overflow-x-hidden bg-background px-4 py-3 md:px-6 md:py-4">
      <h1 className="shrink-0 text-center text-lg font-bold md:text-xl">
        {t("Test Input", "Testeingabe")}
      </h1>
      <p className="mt-1 shrink-0 text-center text-xs text-muted-foreground md:text-sm">
        {t(
          "Using predefined test image instead of camera.",
          "Vordefiniertes Testbild statt Kamera wird verwendet.",
        )}
      </p>

      <div className="mx-auto flex min-h-0 w-full max-w-5xl flex-1 flex-col pt-2">
        <Card className="flex min-h-0 flex-1 flex-col overflow-hidden border border-border/80 shadow-sm">
          <CardContent className="flex min-h-0 flex-1 flex-col gap-2 p-3 md:p-4">
            <div className="flex min-h-0 flex-1 items-center justify-center">
              <img
                src={dummyImage}
                alt={t("Test input", "Testeingabe")}
                className="max-h-full max-w-full rounded-lg object-contain"
              />
            </div>
            <div className="flex shrink-0 justify-center gap-3 pb-0.5 pt-1">
              <Button onClick={proceedToSelection} disabled={!imageBase64}>
                {t("Proceed with Test Image", "Mit Testbild fortfahren")}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default TestInputPage;
