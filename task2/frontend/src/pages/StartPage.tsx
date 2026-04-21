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
import { ChevronRight } from "lucide-react";
import frontpageVideo from "../assets/frontpage_2.mp4";
import journeyFlowVideo from "../assets/frontpage_4.mp4";

const StartPage = () => {
  const navigate = useNavigate();
  const [isConsentOpen, setIsConsentOpen] = React.useState(false);
  const [acceptedConsent, setAcceptedConsent] = React.useState(false);
  const [languageMode, setLanguageModeState] =
    React.useState<LanguageMode>(getLanguageMode());
  const [visibleSteps, setVisibleSteps] = React.useState(0);

  React.useEffect(() => {
    if (isConsentOpen) {
      return;
    }

    const timeoutId = window.setTimeout(
      () => {
        if (visibleSteps < 4) {
          setVisibleSteps((previousValue) => previousValue + 1);
          return;
        }

        setVisibleSteps(0);
      },
      visibleSteps === 0 ? 300 : visibleSteps < 4 ? 600 : 10000,
    );

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [visibleSteps, isConsentOpen]);

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
      descriptionEn: "Capture your silhouette.",
      descriptionDe: "Nimm deine Portrait-Silhouette auf.",
      accent: "bg-film-blue/10",
    },
    {
      titleEn: "Style your silhouette",
      titleDe: "Silhouette gestalten",
      descriptionEn: "Select character-inspired body parts.",
      descriptionDe: "Mische Figuren-inspirierte Koerperteile.",
      accent: "bg-film-green/10",
    },
    {
      titleEn: "Pick movie backgrounds",
      titleDe: "Hintergruende waehlen",
      descriptionEn: "Choose your favorite backgrounds.",
      descriptionDe: "Waehle deine Lieblingshintergruende.",
      accent: "bg-film-red/10",
    },
    {
      titleEn: "Download with QR",
      titleDe: "Mit QR downloaden",
      descriptionEn: "Scan code and save your scenes.",
      descriptionDe: "Scanne einen Code und speichere deine Galerie.",
      accent: "bg-film-yellow/20",
    },
  ];

  return (
    <div className="start-page-shell exhibit-shell flex min-h-0 flex-1 flex-col overflow-hidden overflow-x-hidden">
      <section className="relative mx-auto w-full max-w-[1920px] shrink-0 overflow-hidden aspect-[1920/400]">
        <video
          className="absolute inset-0 h-full w-full object-cover object-center"
          src={frontpageVideo}
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          aria-label={t(
            "Prince Achmed inspired scene",
            "Von Prinz Achmed inspirierte Szene",
          )}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/30 to-black/15" />

        {/* <div className="absolute inset-x-0 bottom-[6.25rem] z-20 md:bottom-[7.5rem]">
          <div className="mx-auto ml-4 max-w-5xl px-4 md:ml-6 md:px-8">
            <div className="w-fit rounded-xl border border-white/40 bg-black/45 p-2 backdrop-blur-sm md:p-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-white/90">
                {t("Interactive Exhibit", "Interaktiver Kiosk")}
              </p>
            </div>
          </div>
        </div> */}

        <h1
          className={`exhibit-title hero-title-yellow-glow absolute bottom-2 inset-x-0 z-20 mx-auto max-w-5xl px-4 pb-1 text-center text-4xl text-white md:px-8 md:pb-2 md:text-5xl lg:text-6xl ${languageMode === "en" ? "lg:whitespace-nowrap" : ""}`}
        >
          {t(
            "Immerse yourself in Achmed's world!",
            "Tauchen Sie ein in Achmeds Welt!",
          )}
        </h1>

        <div className="absolute right-4 bottom-2 z-30 rounded-xl border border-white/40 bg-black/45 p-2 backdrop-blur-sm md:right-8 md:bottom-2 md:p-3">
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
      </section>

      <section className="shrink-0 -mt-2 px-4 pt-2 pb-2 md:-mt-3 md:px-8 md:pt-3 md:pb-4">
        <div className="grid w-full grid-cols-1 items-stretch gap-3 lg:grid-cols-[minmax(0,1fr)_270px] xl:grid-cols-[minmax(0,1fr)_300px]">
          <div className="exhibit-panel flex h-full min-h-0 flex-col rounded-2xl p-3 md:p-5">
            <h2 className="text-xl font-semibold md:text-2xl">
              {t("Become a part of the first animated film in 4 easy steps!", "Werde in 4 einfachen Schritten Teil des ersten Trickfilms!")}
            </h2>

            <div className="mt-4 w-full overflow-x-auto md:mt-5 md:overflow-visible">
              <div className="flex min-w-max items-stretch gap-2 md:min-w-0 md:w-full md:gap-3">
                {steps.map((step, index) => (
                  <React.Fragment key={step.titleEn}>
                    <div
                      className={`flex min-w-0 flex-1 basis-0 ${index >= steps.length - 2 ? "justify-end" : "justify-start"}`}
                    >
                      <div
                        className={`flex h-full max-w-full min-w-[9rem] flex-col rounded-xl border border-border px-2 py-1.5 text-left text-film-black shadow-[0_10px_34px_hsl(var(--film-black)_/_0.12)] backdrop-blur-[6px] transition-[opacity,transform] duration-500 sm:min-w-[10rem] md:px-2.5 md:py-1.5 ${index === 1 ? "w-[calc(100%-0.125rem)] md:w-[calc(100%-0.375rem)]" : "w-[calc(100%-0.625rem)] md:w-[calc(100%-0.875rem)]"} ${step.accent} ${index < visibleSteps ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2"}`}
                      >
                        <p className="min-w-0 text-[0.9375rem] font-semibold uppercase leading-tight tracking-wide md:text-[1.0625rem]">
                          {t(`Step ${index + 1}`, `Schritt ${index + 1}`)}
                        </p>
                        <p className="mt-0.5 min-w-0 text-[1.1875rem] font-semibold leading-snug md:text-[1.375rem]">
                          {t(step.titleEn, step.titleDe)}
                        </p>
                        <p className="mt-0.5 min-w-0 text-[1.0625rem] leading-snug md:text-[1.1875rem]">
                          {t(step.descriptionEn, step.descriptionDe)}
                        </p>
                      </div>
                    </div>
                    {index < steps.length - 1 ? (
                      <div
                        className={`flex w-4 shrink-0 items-center justify-center self-stretch text-film-black/55 md:w-5 ${index === 0 ? "-translate-x-1 md:-translate-x-1.5" : ""} ${index === steps.length - 2 ? "translate-x-1 md:translate-x-1.5" : ""}`}
                        aria-hidden
                      >
                        <ChevronRight className="h-5 w-5 shrink-0" />
                      </div>
                    ) : null}
                  </React.Fragment>
                ))}
              </div>
            </div>
          </div>

          <aside className="exhibit-panel flex h-full min-h-0 flex-col rounded-2xl p-3 md:p-5">
            <h3 className="w-full text-center text-xl font-semibold md:text-2xl">
              {t("Ready to begin?", "Bereit zu starten?")}
            </h3>

            <Button
              size="xl"
              onClick={startExperience}
              className="cta-step-2 cta-start-outlined mt-3 w-full min-h-[4.75rem] rounded-2xl py-4 text-xl font-bold uppercase tracking-[0.18em] text-white md:mt-4 md:min-h-[5.75rem] md:py-6 md:text-3xl"
            >
              {t("START", "START")}
            </Button>
          </aside>
        </div>
      </section>

      <section className="relative -mt-2 mx-auto w-full max-w-[1920px] min-h-[200px] flex-1 overflow-hidden bg-black md:-mt-3 md:min-h-[240px]">
        <video
          className="absolute inset-0 h-full w-full object-cover object-center"
          src={journeyFlowVideo}
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          aria-label={t(
            "Preview of the four-step exhibit flow",
            "Vorschau des vierstufigen Ausstellungsablaufs",
          )}
        />
      </section>

      <Dialog
        modal={false}
        open={isConsentOpen}
        onOpenChange={setIsConsentOpen}
      >
        <DialogContent
          overlayClassName="data-[state=open]:!animate-none data-[state=closed]:!animate-none"
          className="max-w-2xl rounded-2xl border border-film-blue/25 p-6 ring-1 ring-inset ring-background md:p-8 data-[state=open]:!animate-none data-[state=closed]:!animate-none [transform:translateZ(0)]"
          onOpenAutoFocus={(event) => event.preventDefault()}
        >
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
