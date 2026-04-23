import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { t } from "@/lib/localization";

/** Match consent dialog: no scroll-lock jump, no enter/exit motion, edge sealing. */
export const LEGAL_DIALOG_OVERLAY_CLASS =
  "data-[state=open]:!animate-none data-[state=closed]:!animate-none";

export const LEGAL_DIALOG_CONTENT_CLASS =
  "max-w-2xl rounded-2xl border border-film-blue/25 p-6 ring-1 ring-inset ring-background md:p-8 data-[state=open]:!animate-none data-[state=closed]:!animate-none [transform:translateZ(0)]";

type LegalDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

export function PrivacyPolicyDocumentDialog({
  open,
  onOpenChange,
}: LegalDialogProps) {
  return (
    <Dialog modal={false} open={open} onOpenChange={onOpenChange}>
      <DialogContent
        overlayClassName={LEGAL_DIALOG_OVERLAY_CLASS}
        className={LEGAL_DIALOG_CONTENT_CLASS}
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
  );
}

export function TermsDocumentDialog({ open, onOpenChange }: LegalDialogProps) {
  return (
    <Dialog modal={false} open={open} onOpenChange={onOpenChange}>
      <DialogContent
        overlayClassName={LEGAL_DIALOG_OVERLAY_CLASS}
        className={LEGAL_DIALOG_CONTENT_CLASS}
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
  );
}
