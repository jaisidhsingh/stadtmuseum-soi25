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
import logoTueai from "@/assets/logo-tueai.svg";
import stadtmuseumLogo from "@/assets/stadtmuseum-logo.svg";
import tuebingenUniLogo from "@/assets/tuebingen-universitaetsstadt-logo-white.svg";

/** Match StartPage consent dialog: no scroll-lock jump, no enter/exit motion, edge sealing. */
const FOOTER_DIALOG_OVERLAY_CLASS =
  "data-[state=open]:!animate-none data-[state=closed]:!animate-none";

const FOOTER_DIALOG_CONTENT_CLASS =
  "max-w-2xl rounded-2xl border border-film-blue/25 p-6 ring-1 ring-inset ring-background md:p-8 data-[state=open]:!animate-none data-[state=closed]:!animate-none [transform:translateZ(0)]";

const PrivacyFooter = () => {
  const [showPrivacy, setShowPrivacy] = useState(false);
  const [showTerms, setShowTerms] = useState(false);

  return (
    <footer className="w-full border-t bg-background/95 py-2 px-4 md:px-6">
      <div className="flex flex-wrap items-center justify-center gap-x-5 gap-y-2 md:gap-x-8">
        <div className="flex flex-wrap items-center gap-2 text-sm md:gap-4">
        <Dialog modal={false} open={showPrivacy} onOpenChange={setShowPrivacy}>
          <DialogTrigger asChild>
            <Button variant="link" size="sm">
              {t("Privacy Policy", "Datenschutzerklaerung")}
            </Button>
          </DialogTrigger>
          <DialogContent
            overlayClassName={FOOTER_DIALOG_OVERLAY_CLASS}
            className={FOOTER_DIALOG_CONTENT_CLASS}
            onOpenAutoFocus={(event) => event.preventDefault()}
          >
            <DialogHeader className="text-left">
              <DialogTitle className="exhibit-title text-2xl md:text-3xl">
                {t("Privacy Policy", "Datenschutzerklaerung")}
              </DialogTitle>
            </DialogHeader>
            <div className="space-y-4 text-sm text-film-black md:text-base">
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

        <Dialog modal={false} open={showTerms} onOpenChange={setShowTerms}>
          <DialogTrigger asChild>
            <Button variant="link" size="sm">
              {t("Terms and Conditions", "Nutzungsbedingungen")}
            </Button>
          </DialogTrigger>
          <DialogContent
            overlayClassName={FOOTER_DIALOG_OVERLAY_CLASS}
            className={FOOTER_DIALOG_CONTENT_CLASS}
            onOpenAutoFocus={(event) => event.preventDefault()}
          >
            <DialogHeader className="text-left">
              <DialogTitle className="exhibit-title text-2xl md:text-3xl">
                {t("Terms and Conditions", "Nutzungsbedingungen")}
              </DialogTitle>
            </DialogHeader>
            <div className="space-y-4 text-sm text-film-black md:text-base">
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

        <div className="flex flex-wrap items-center gap-10 md:gap-16 [&_img]:object-contain">
          <img
            src={logoTueai}
            alt={t("Tübingen AI", "Tuebingen AI")}
            className="h-6 w-auto md:h-11"
          />
          <img
            src={stadtmuseumLogo}
            alt={t("Stadtmuseum", "Stadtmuseum")}
            className="h-4 w-auto md:h-3.5"
          />
          <img
            src={tuebingenUniLogo}
            alt={t(
              "University city of Tübingen",
              "Universitaetsstadt Tuebingen",
            )}
            className="h-8 w-auto brightness-0 md:h-8"
          />
        </div>
      </div>
    </footer>
  );
};

export default PrivacyFooter;
