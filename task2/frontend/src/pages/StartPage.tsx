import * as React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const StartPage = () => {
  const navigate = useNavigate();
  const [acceptedPrivacy, setAcceptedPrivacy] = React.useState(false);
  const [acceptedTerms, setAcceptedTerms] = React.useState(false);

  const canStart = acceptedPrivacy && acceptedTerms;

  return (
    <div className="min-h-screen bg-background p-8 flex flex-col">
      <h1 className="text-3xl font-bold text-center mb-4">
        The Adventures of Prince Achmed
      </h1>
      <p className="text-center text-muted-foreground mb-6">
        Celebrating the Anniversary of Lotte Reiniger's Masterpiece
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 flex-1">
        {/* Column 1: Instructions */}
        <Card>
          <CardHeader>
            <CardTitle>How It Works</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
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
            <p className="text-sm text-muted-foreground">
              Your photo will only be used to create your personalized artwork.
              We do not store your original photo after processing.
              Your email will only be used to send you your selected images.
            </p>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="privacy"
                checked={acceptedPrivacy}
                onCheckedChange={(checked) => setAcceptedPrivacy(checked === true)}
              />
              <label htmlFor="privacy" className="text-sm cursor-pointer">
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
            <p className="text-sm text-muted-foreground">
              By using this exhibit, you agree to let us process your image
              for artistic purposes. The generated artwork is for personal use only.
              You may share your artwork on social media with proper attribution.
            </p>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="terms"
                checked={acceptedTerms}
                onCheckedChange={(checked) => setAcceptedTerms(checked === true)}
              />
              <label htmlFor="terms" className="text-sm cursor-pointer">
                I accept the Terms & Conditions
              </label>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="flex justify-center mt-8">
        <Button
          size="lg"
          disabled={!canStart}
          onClick={() => navigate("/camera")}
        >
          Start Experience
        </Button>
      </div>
    </div>
  );
};

export default StartPage;
