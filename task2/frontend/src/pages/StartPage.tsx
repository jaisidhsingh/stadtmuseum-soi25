import * as React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  getLanguageMode,
  setLanguageMode,
  t,
  type LanguageMode,
} from "@/lib/localization";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import topImage from "../assets/bg_start_page_2.jpg";

const StartPage = () => {
  const navigate = useNavigate();
  const [isConsentOpen, setIsConsentOpen] = React.useState(false);
  const [acceptedConsent, setAcceptedConsent] = React.useState(false);
  const [languageMode, setLanguageModeState] =
    React.useState<LanguageMode>(getLanguageMode());
  const [visibleSteps, setVisibleSteps] = React.useState(0);

  React.useEffect(() => {
    const timeoutId = window.setTimeout(
      () => {
        if (visibleSteps < 4) {
          setVisibleSteps((previousValue) => previousValue + 1);
          return;
        }

        setVisibleSteps(0);
      },
      visibleSteps === 0 ? 600 : visibleSteps < 4 ? 3000 : 10000,
    );

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [visibleSteps]);

  const handleLanguageChange = (mode: "en" | "de") => {
    setLanguageMode(mode);
    setLanguageModeState(mode);
  };

  const startExperience = () => {
    setAcceptedConsent(false);
    setIsConsentOpen(true);
  };

  const proceedToCamera = () => {
    if (!acceptedConsent) {
      return;
    }

    setIsConsentOpen(false);
    navigate("/camera");
  };

  const steps = [
    {
      titleEn: "Take your photo",
      titleDe: "Foto aufnehmen",
      descriptionEn: "Capture your portrait silhouette.",
      descriptionDe: "Nimm deine Portrait-Silhouette auf.",
      accent: "border-film-blue/25 bg-film-blue/10 text-film-blue",
    },
    {
      titleEn: "Style your silhouette",
      titleDe: "Silhouette gestalten",
      descriptionEn: "Mix character-inspired body parts.",
      descriptionDe: "Mische Figuren-inspirierte Koerperteile.",
      accent: "border-film-green/25 bg-film-green/10 text-film-green",
    },
    {
      titleEn: "Pick story scenes",
      titleDe: "Hintergruende waehlen",
      descriptionEn: "Choose your favorite backgrounds.",
      descriptionDe: "Waehle deine Lieblingshintergruende.",
      accent: "border-film-red/25 bg-film-red/10 text-film-red",
    },
    {
      titleEn: "Share by QR",
      titleDe: "Per QR teilen",
      descriptionEn: "Scan one code and save your gallery.",
      descriptionDe: "Scanne einen Code und speichere deine Galerie.",
      accent: "border-film-yellow/35 bg-film-yellow/20 text-film-black",
    },
  ];

  return (
    <div className="h-full min-h-0 exhibit-shell flex flex-col overflow-hidden overflow-x-hidden">
      <section className="relative min-h-[56vh] overflow-hidden md:min-h-[62vh]">
        <img
          src={topImage}
          alt={t(
            "Prince Achmed inspired scene",
            "Von Prinz Achmed inspirierte Szene",
          )}
          className="absolute inset-0 h-full w-full object-cover object-[center_26%]"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/30 to-black/15" />

        <div className="absolute right-4 top-4 z-20 rounded-xl border border-white/40 bg-black/45 p-2 backdrop-blur-sm md:right-8 md:top-7 md:p-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-white/90">
            {t("Language", "Sprache")}
          </p>
          <div className="mt-1 grid grid-cols-2 gap-1.5">
            <button
              type="button"
              onClick={() => handleLanguageChange("en")}
              className={`rounded-md px-3 py-1.5 text-sm font-semibold transition-colors ${languageMode === "en" ? "bg-film-blue text-white" : "bg-white/85 text-foreground hover:bg-white"}`}
            >
              English
            </button>
            <button
              type="button"
              onClick={() => handleLanguageChange("de")}
              className={`rounded-md px-3 py-1.5 text-sm font-semibold transition-colors ${languageMode === "de" ? "bg-film-blue text-white" : "bg-white/85 text-foreground hover:bg-white"}`}
            >
              Deutsch
            </button>
          </div>
        </div>

        <div className="absolute inset-x-0 bottom-0 px-4 pb-8 md:px-8 md:pb-10">
          <span className="film-tag border-white/35 bg-white/20 text-white">
            {t("Interactive Kiosk", "Interaktiver Kiosk")}
          </span>
          <h1
            className={`exhibit-title mt-4 max-w-none text-4xl text-white md:text-5xl lg:text-6xl ${languageMode === "en" ? "lg:whitespace-nowrap" : ""}`}
          >
            {t(
              "The Adventures of Prince Achmed",
              "Die Abenteuer des Prinzen Achmed",
            )}
          </h1>
          <p
            className={`mt-3 max-w-none text-base text-white/95 md:text-xl lg:text-2xl ${languageMode === "en" ? "lg:whitespace-nowrap" : ""}`}
          >
            {t(
              "Step into Lotte Reiniger's world and become part of the silhouette story.",
              "Tauche in Lotte Reinigers Welt ein und werde Teil der Silhouetten-Geschichte.",
            )}
          </p>
        </div>
      </section>

      <section className="flex-1 px-4 py-4 md:px-8 md:py-6">
        <div className="mx-auto grid max-w-7xl grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
          <div className="exhibit-panel rounded-2xl p-4 md:p-6">
            <h2 className="text-2xl font-semibold md:text-3xl">
              {t("Your 4-step journey", "Deine 4-Schritte-Reise")}
            </h2>
            <p className="mt-1 text-sm text-muted-foreground md:text-base">
              {t(
                "Watch the flow preview, then tap Start.",
                "Sieh dir die Ablaufvorschau an und tippe dann auf Start.",
              )}
            </p>

            <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
              {steps.map((step, index) => (
                <div
                  key={step.titleEn}
                  className={`rounded-xl border px-4 py-3 transition-all duration-500 ${step.accent} ${index < visibleSteps ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2"}`}
                >
                  <p className="text-sm font-semibold uppercase tracking-wide">
                    {t(`Step ${index + 1}`, `Schritt ${index + 1}`)}
                  </p>
                  <p className="text-lg font-semibold md:text-xl">
                    {t(step.titleEn, step.titleDe)}
                  </p>
                  <p className="mt-1 text-sm text-foreground/85 md:text-base">
                    {t(step.descriptionEn, step.descriptionDe)}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <aside className="exhibit-panel flex flex-col rounded-2xl p-4 md:p-6">
            <h3 className="text-2xl font-semibold md:text-3xl">
              {t("Ready to begin?", "Bereit zu starten?")}
            </h3>
            <p className="mt-2 text-sm text-muted-foreground md:text-base">
              {t(
                "Select your language above, then open consent and start.",
                "Waehle oben deine Sprache und oeffne dann die Zustimmung.",
              )}
            </p>

            <Button
              size="xl"
              onClick={startExperience}
              className="cta-step-1 mt-auto w-full text-lg font-semibold text-white md:text-xl"
            >
              {t("Start Experience", "Erlebnis starten")}
            </Button>

            <p className="mt-2 text-center text-xs text-muted-foreground md:text-sm">
              {t(
                "Consent appears on next tap.",
                "Die Zustimmung erscheint beim naechsten Schritt.",
              )}
            </p>
          </aside>
        </div>
      </section>

      <Dialog open={isConsentOpen} onOpenChange={setIsConsentOpen}>
        <DialogContent className="max-w-2xl rounded-2xl border border-film-blue/25 p-6 md:p-8">
          <DialogHeader className="text-left">
            <DialogTitle className="exhibit-title text-2xl md:text-3xl">
              {t("Before we open the camera", "Bevor wir die Kamera oeffnen")}
            </DialogTitle>
            <DialogDescription className="text-sm md:text-base">
              {t(
                "We use your photo only to create this temporary exhibit artwork.",
                "Wir verwenden dein Foto nur fuer dieses temporaere Kunstwerk in der Ausstellung.",
              )}
            </DialogDescription>
          </DialogHeader>

          <div className="rounded-xl border border-film-red/20 bg-film-red/10 p-4 text-sm text-film-black md:text-base">
            {t(
              "Your source photo and generated data are deleted automatically after 15 minutes.",
              "Dein Ausgangsfoto und alle generierten Daten werden automatisch nach 15 Minuten geloescht.",
            )}
          </div>

          <div className="flex items-start gap-3 rounded-xl border border-border bg-muted/40 p-4">
            <Checkbox
              id="consent"
              checked={acceptedConsent}
              onCheckedChange={(checked) => setAcceptedConsent(checked === true)}
            />
            <label
              htmlFor="consent"
              className="cursor-pointer text-sm md:text-base"
            >
              {t(
                "I agree that my image can be processed for this installation.",
                "Ich stimme zu, dass mein Bild fuer diese Installation verarbeitet wird.",
              )}
            </label>
          </div>

          <DialogFooter className="mt-2 flex-col gap-2 sm:flex-row sm:justify-between sm:gap-3">
            <Button
              variant="outline"
              size="xl"
              onClick={() => setIsConsentOpen(false)}
              className="w-full sm:w-auto"
            >
              {t("Cancel", "Abbrechen")}
            </Button>
            <Button
              size="xl"
              onClick={proceedToCamera}
              disabled={!acceptedConsent}
              className="cta-step-1 w-full font-semibold text-white sm:w-auto"
            >
              {t("Open Camera", "Kamera oeffnen")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default StartPage;
