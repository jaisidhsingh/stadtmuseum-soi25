import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

const PrivacyFooter = () => {
  const [showPrivacy, setShowPrivacy] = useState(false);
  const [showTerms, setShowTerms] = useState(false);

  return (
    <footer className="fixed bottom-0 left-0 right-0 bg-background border-t py-2 px-4">
      <div className="flex justify-center gap-4 text-sm">
        <Dialog open={showPrivacy} onOpenChange={setShowPrivacy}>
          <DialogTrigger asChild>
            <Button variant="link" size="sm">
              Privacy Policy
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Privacy Policy</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 text-sm text-muted-foreground">
              <p>
                Your photo will only be used to create your personalized artwork.
                We do not store your original photo after processing.
              </p>
              <p>
                Your email will only be used to send you your selected images.
                We do not share your information with third parties.
              </p>
              <p>
                All processing is done securely and your data is handled in
                accordance with applicable data protection regulations.
              </p>
            </div>
          </DialogContent>
        </Dialog>

        <span className="text-muted-foreground">|</span>

        <Dialog open={showTerms} onOpenChange={setShowTerms}>
          <DialogTrigger asChild>
            <Button variant="link" size="sm">
              Terms & Conditions
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Terms & Conditions</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 text-sm text-muted-foreground">
              <p>
                By using this exhibit, you agree to let us process your image
                for artistic purposes.
              </p>
              <p>
                The generated artwork is for personal use only. You may share
                your artwork on social media with proper attribution to the
                museum and Lotte Reiniger.
              </p>
              <p>
                The museum reserves the right to use anonymized, aggregated data
                for improving the exhibit experience.
              </p>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </footer>
  );
};

export default PrivacyFooter;
