import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { t } from "@/lib/localization";

const PrivacyFooter = () => {
  const [showPrivacy, setShowPrivacy] = useState(false);
  const [showTerms, setShowTerms] = useState(false);

  return (
    <footer className="w-full border-t bg-background/95 py-2 px-4">
      <div className="flex flex-wrap items-center justify-center gap-2 text-sm md:gap-4">
        <Dialog open={showPrivacy} onOpenChange={setShowPrivacy}>
          <DialogTrigger asChild>
            <Button variant="link" size="sm">
              {t("Privacy Policy", "Datenschutzerklaerung")}
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>
                {t("Privacy Policy", "Datenschutzerklaerung")}
              </DialogTitle>
            </DialogHeader>
            <div className="space-y-4 text-sm text-muted-foreground">
              <p>
                {t(
                  "Your photo is used only to create your personalized artwork. We do not store your original photo after processing.",
                  "Dein Foto wird nur zur Erstellung deines personalisierten Kunstwerks verwendet. Nach der Verarbeitung speichern wir dein Originalfoto nicht.",
                )}
              </p>
              <p>
                {t(
                  "QR share links are temporary and expire automatically. Expired files are deleted from the server.",
                  "QR-Freigabelinks sind temporaer und verfallen automatisch. Abgelaufene Dateien werden vom Server geloescht.",
                )}
              </p>
              <p>
                {t(
                  "All processing is done securely and your data is handled in accordance with applicable data protection regulations.",
                  "Alle Verarbeitungen erfolgen sicher und deine Daten werden gemaess den geltenden Datenschutzbestimmungen behandelt.",
                )}
              </p>
            </div>
          </DialogContent>
        </Dialog>

        <span className="text-muted-foreground">|</span>

        <Dialog open={showTerms} onOpenChange={setShowTerms}>
          <DialogTrigger asChild>
            <Button variant="link" size="sm">
              {t("Terms and Conditions", "Nutzungsbedingungen")}
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>
                {t("Terms and Conditions", "Nutzungsbedingungen")}
              </DialogTitle>
            </DialogHeader>
            <div className="space-y-4 text-sm text-muted-foreground">
              <p>
                {t(
                  "By using this exhibit, you agree that your image may be processed for artistic purposes.",
                  "Durch die Nutzung dieser Ausstellung stimmst du zu, dass dein Bild fuer kuenstlerische Zwecke verarbeitet werden darf.",
                )}
              </p>
              <p>
                {t(
                  "The generated artwork is for personal use only. You may share it on social media with proper attribution to the museum and Lotte Reiniger.",
                  "Das generierte Kunstwerk ist nur fuer den persoenlichen Gebrauch bestimmt. Du darfst es in sozialen Medien teilen, wenn das Museum und Lotte Reiniger korrekt genannt werden.",
                )}
              </p>
              <p>
                {t(
                  "The museum may use anonymized, aggregated data to improve the exhibit experience.",
                  "Das Museum darf anonymisierte, aggregierte Daten zur Verbesserung des Ausstellungserlebnisses verwenden.",
                )}
              </p>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </footer>
  );
};

export default PrivacyFooter;
