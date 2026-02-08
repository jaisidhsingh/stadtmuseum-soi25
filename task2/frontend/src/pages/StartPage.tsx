import * as React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import topImage from "../assets/bg_start_page.jpg";

const StartPage = () => {
  const navigate = useNavigate();
  const [acceptedPrivacy, setAcceptedPrivacy] = React.useState(false);
  const [acceptedTerms, setAcceptedTerms] = React.useState(false);

  const canStart = acceptedPrivacy && acceptedTerms;

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Top Section: Background Image */}
      <div className="flex-1 relative w-full overflow-hidden">
        <img
          src={topImage}
          alt="Start Page Background"
          className="w-full h-full object-cover absolute inset-0"
        />
        {/* Optional overlay gradient if improved text readability is needed later, 
            though text is now in the bottom section */}
      </div>

      {/* Bottom Section: Content */}
      <div className="p-8 pb-32 bg-background relative z-10 shadow-[0_-10px_40px_rgba(0,0,0,0.1)]">
        <div className="max-w-5xl mx-auto">
          <h1 className="text-3xl font-bold text-center mb-2">
            The Adventures of Prince Achmed
          </h1>
          <p className="text-center text-muted-foreground mb-8">
            Celebrating the Anniversary of Lotte Reiniger's Masterpiece
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {/* Column 1: Instructions */}
            <Card>
              <CardHeader>
                <CardTitle>How It Works</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <p>1. Take a photo of yourself</p>
                <p>2. We'll transform you into a silhouette in Lotte Reiniger's style</p>
                <p>3. Choose your favorite backgrounds</p>
                <p>4. Get your artwork sent to your email!</p>
              </CardContent>
            </Card>

            {/* Column 2: Privacy Policy */}
            <Card>
              <CardHeader>
                <CardTitle>Privacy Policy</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-xs text-muted-foreground">
                  Your photo will only be used to create your personalized artwork.
                  We do not store your original photo after processing.
                </p>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="privacy"
                    checked={acceptedPrivacy}
                    onCheckedChange={(checked) => setAcceptedPrivacy(checked === true)}
                  />
                  <label htmlFor="privacy" className="text-xs cursor-pointer font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                    I accept the Privacy Policy
                  </label>
                </div>
              </CardContent>
            </Card>

            {/* Column 3: Terms & Conditions */}
            <Card>
              <CardHeader>
                <CardTitle>Terms & Conditions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-xs text-muted-foreground">
                  By using this exhibit, you agree to let us process your image
                  for artistic purposes. The generated artwork is for personal use only.
                </p>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="terms"
                    checked={acceptedTerms}
                    onCheckedChange={(checked) => setAcceptedTerms(checked === true)}
                  />
                  <label htmlFor="terms" className="text-xs cursor-pointer font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                    I accept the Terms & Conditions
                  </label>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="flex justify-center">
            <Button
              size="lg"
              disabled={!canStart}
              onClick={() => navigate("/camera")}
              className="text-lg px-8 py-6 shadow-lg hover:shadow-xl transition-all"
            >
              Start Experience
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StartPage;
