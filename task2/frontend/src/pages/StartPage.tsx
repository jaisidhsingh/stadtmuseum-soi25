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
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ChevronRight } from "lucide-react";
import {
  ExhibitStepCard,
  exhibitStepDefaultDescriptionClass,
} from "@/components/ExhibitStepCard";
import { EXHIBIT_STEPS, type ExhibitStepIndex } from "@/lib/exhibitFlow";
import {
  continueButtonClassName,
  retakeButtonClassName,
} from "@/lib/flowCtaClassNames";
import { cn } from "@/lib/utils";
import {
  PrivacyPolicyDocumentDialog,
  TermsDocumentDialog,
} from "@/components/LegalDocumentDialogs";
import frontpageVideo from "../assets/frontpage_5.mp4";
import journeyFlowVideo from "../assets/frontpage_4.mp4";

/**
 * Separate aspects per asset so the box matches each clip (less arbitrary crop with object-cover).
 * Hero: 1920×305. Journey: 1920×320. Tweak to match `ffprobe -show_entries stream=width,height` on the files.
 * `flex-none` + `min-h-0` avoids flex stretching when all children are position:absolute.
 */
const VIDEO_STRIP_BASE =
  "relative mx-auto w-full max-w-[1920px] flex-none min-h-0 shrink-0 overflow-hidden mt-2";

const HERO_VIDEO_STRIP_CLASS = `${VIDEO_STRIP_BASE} aspect-[1920/305]`;
// No top margin: strip position follows the step cards; spacing to cards comes from the middle block + pb.
const JOURNEY_VIDEO_STRIP_CLASS = `${VIDEO_STRIP_BASE} aspect-[1920/320] bg-black`;

const StartPage = () => {
  const navigate = useNavigate();
  const [isConsentOpen, setIsConsentOpen] = React.useState(false);
  const [acceptedProcessing, setAcceptedProcessing] = React.useState(false);
  const [acceptedPrivacyTerms, setAcceptedPrivacyTerms] =
    React.useState(false);
  const [showPrivacyDoc, setShowPrivacyDoc] = React.useState(false);
  const [showTermsDoc, setShowTermsDoc] = React.useState(false);
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
    setAcceptedProcessing(false);
    setAcceptedPrivacyTerms(false);
    setIsConsentOpen(true);
  };

  const proceedToCamera = () => {
    if (!acceptedProcessing || !acceptedPrivacyTerms) {
      return;
    }

    setIsConsentOpen(false);
    navigate("/camera");
  };

  const canOpenCamera = acceptedProcessing && acceptedPrivacyTerms;

  return (
    <div className="start-page-shell exhibit-shell flex h-full min-h-0 flex-1 flex-col overflow-hidden overflow-x-hidden">
      <section className={HERO_VIDEO_STRIP_CLASS}>
        <video
          className="absolute inset-0 h-full w-full object-cover object-top"
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
      </section>

      {/* Title + language are absolute. Grey cards stay in flex flow (bottom) so h-full / grid stretch match the pre-refactor layout. */}
      <section className="relative z-0 flex min-h-0 flex-1 flex-col justify-end px-4 md:-mt-3 md:px-8">
        {/* 1) Headline: tune `top` / horizontal padding only — does not move the cards or language. */}
        <div className="pointer-events-none absolute left-0 right-0 top-2 z-20 px-4 sm:top-3 md:px-8">
          <h1
            className={`exhibit-title hero-title-yellow-glow pointer-events-auto mx-auto max-w-5xl translate-y-10 px-2 text-center text-3xl text-white sm:px-4 md:px-8 md:pr-40 md:text-4xl md:text-5xl lg:pr-44 lg:text-6xl ${languageMode === "en" ? "lg:whitespace-nowrap" : ""}`}
          >
            {t(
              "Immerse yourself in Achmed's world!",
              "Tauchen Sie ein in Achmeds Welt!",
            )}
          </h1>
        </div>

        {/* 2) Language: tune `top` / `right` / `md:right-*` only — independent of headline & cards. */}
        <div className="absolute left-0 right-0 top-[7.5rem] z-30 translate-y-8 flex justify-center sm:top-[6.75rem] md:top-3 md:left-auto md:right-4 md:w-auto md:justify-end lg:right-8">
          <div className="pointer-events-auto flex w-fit min-w-0 max-w-full flex-col items-center gap-1.5 rounded-xl border border-white bg-black/45 px-2.5 py-1.5 backdrop-blur-sm md:px-3 md:py-2">
            <p className="-mt-0.5 w-full text-center text-xs font-semibold uppercase leading-tight tracking-wide text-white/90">
              {t("Language", "Sprache")}
            </p>
            <div className="grid w-full grid-cols-2 gap-1.5">
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
        </div>

        {/* 3) Grey cards: in-flow, `mt-auto` + `justify-end` on section pins to bottom; size matches earlier flex layout. */}
        <div className="relative z-10 mt-auto w-full min-h-0 shrink-0 pb-1 md:pb-10">
          <div className="grid w-full min-h-0 grid-cols-1 items-stretch gap-3 lg:grid-cols-[minmax(0,1fr)_270px] xl:grid-cols-[minmax(0,1fr)_300px]">
            <div className="flex h-full min-h-0 flex-col rounded-2xl border border-white bg-black p-3 md:p-5">
              <h2 className="exhibit-title hero-title-yellow-glow text-xl font-semibold text-white md:text-2xl">
                {t("Become a part of the first animated film in 4 easy steps!", "Werde in 4 einfachen Schritten Teil des ersten Trickfilms!")}
              </h2>

              <div className="mt-4 w-full overflow-x-auto md:mt-5 md:overflow-visible">
                <div className="flex min-w-max items-stretch gap-2 md:min-w-0 md:w-full md:gap-3">
                  {EXHIBIT_STEPS.map((step, index) => (
                    <React.Fragment key={step.titleEn}>
                      <div
                        className={`flex min-w-0 flex-1 basis-0 ${index >= EXHIBIT_STEPS.length - 2 ? "justify-end" : "justify-start"}`}
                      >
                        <ExhibitStepCard
                          stepIndex={index as ExhibitStepIndex}
                          size="default"
                          className={`h-full max-w-full min-w-[9rem] transition-[opacity,transform] duration-500 sm:min-w-[10rem] md:px-2.5 md:py-1.5 ${index === 1 ? "w-[calc(100%-0.125rem)] md:w-[calc(100%-0.375rem)]" : "w-[calc(100%-0.625rem)] md:w-[calc(100%-0.875rem)]"} ${index < visibleSteps ? "translate-y-0 opacity-100" : "translate-y-2 opacity-0"}`}
                        />
                      </div>
                      {index < EXHIBIT_STEPS.length - 1 ? (
                        <div
                          className={`flex w-4 shrink-0 items-center justify-center self-stretch text-white/80 md:w-5 ${index === 0 ? "-translate-x-1 md:-translate-x-1.5" : ""} ${index === EXHIBIT_STEPS.length - 2 ? "translate-x-1 md:translate-x-1.5" : ""}`}
                          aria-hidden
                        >
                          <ChevronRight className="hero-title-yellow-glow-icon h-5 w-5 shrink-0" />
                        </div>
                      ) : null}
                    </React.Fragment>
                  ))}
                </div>
              </div>
            </div>

            <aside className="flex h-full min-h-0 flex-col items-stretch rounded-2xl border border-white bg-black p-3 md:p-5">
              <h3 className="exhibit-title hero-title-yellow-glow w-full shrink-0 text-center text-xl font-semibold text-white md:text-2xl">
                {t("Ready to begin?", "Bereit zu starten?")}
              </h3>

              <Button
                size="xl"
                onClick={startExperience}
                className="cta-step-2 cta-start-outlined mt-3 w-full min-h-14 rounded-2xl py-3 text-xl font-bold uppercase tracking-[0.18em] text-white md:mt-4 md:text-3xl lg:mt-4 lg:flex-1 lg:basis-0 lg:py-4"
              >
                {t("START", "START")}
              </Button>
            </aside>
          </div>
        </div>
      </section>

      <section className={JOURNEY_VIDEO_STRIP_CLASS}>
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
              {t("Before we start the camera", "Bevor wir die Kamera einschalten")}
            </DialogTitle>
          </DialogHeader>

          <div
            className={cn(
              "rounded-xl border border-film-red/20 bg-film-red/10 p-4 text-film-black",
              exhibitStepDefaultDescriptionClass,
            )}
          >
            {t(
              "We use your photo only to create this temporary exhibit artwork. Your source photo and generated data are deleted automatically 15 minutes after the QR code is created.",
              "Wir verwenden dein Foto nur fuer dieses temporaere Kunstwerk in der Ausstellung.Dein Ausgangsfoto und alle generierten Daten werden automatisch 15 Minuten nach dem QR-Code erstellt geloescht.",
            )}
          </div>

          <div className="space-y-3">
            <div className="flex items-start gap-3 rounded-xl border border-border bg-muted/40 p-4">
              <Checkbox
                id="consent-processing"
                checked={acceptedProcessing}
                onCheckedChange={(checked) =>
                  setAcceptedProcessing(checked === true)
                }
              />
              <label
                htmlFor="consent-processing"
                className={cn(
                  "cursor-pointer text-foreground",
                  exhibitStepDefaultDescriptionClass,
                )}
              >
                {t(
                  "I agree that my image can be processed for this installation.",
                  "Ich stimme zu, dass mein Bild fuer diese Installation verarbeitet wird.",
                )}
              </label>
            </div>

            <div className="flex items-start gap-3 rounded-xl border border-border bg-muted/40 p-4">
              <Checkbox
                id="consent-legal"
                checked={acceptedPrivacyTerms}
                onCheckedChange={(checked) =>
                  setAcceptedPrivacyTerms(checked === true)
                }
                aria-label={t(
                  "I agree to the privacy policy and terms and conditions",
                  "Ich stimme der Datenschutzerklaerung und den Nutzungsbedingungen zu",
                )}
              />
              <div
                className={cn("min-w-0 text-foreground", exhibitStepDefaultDescriptionClass)}
              >
                <span>
                  {t("I agree to the ", "Ich stimme der ")}
                </span>
                <Button
                  type="button"
                  variant="link"
                  className={cn(
                    "h-auto min-h-0 p-0 font-semibold text-film-blue",
                    exhibitStepDefaultDescriptionClass,
                  )}
                  onClick={() => setShowPrivacyDoc(true)}
                >
                  {t("Privacy Policy", "Datenschutzerklaerung")}
                </Button>
                <span>{t(" and ", " und den ")}</span>
                <Button
                  type="button"
                  variant="link"
                  className={cn(
                    "h-auto min-h-0 p-0 font-semibold text-film-blue",
                    exhibitStepDefaultDescriptionClass,
                  )}
                  onClick={() => setShowTermsDoc(true)}
                >
                  {t("Terms and Conditions", "Nutzungsbedingungen")}
                </Button>
                <span>{t(".", " zu.")}</span>
              </div>
            </div>
          </div>

          <DialogFooter className="mt-2 flex flex-col gap-3 sm:flex-row sm:items-stretch sm:gap-4">
            <Button
              type="button"
              size="xl"
              onClick={() => setIsConsentOpen(false)}
              className={cn(
                retakeButtonClassName,
                "w-full min-w-0 sm:flex-1",
              )}
            >
              {t("Cancel", "Abbrechen")}
            </Button>
            <Button
              type="button"
              size="xl"
              onClick={proceedToCamera}
              disabled={!canOpenCamera}
              className={cn(
                continueButtonClassName,
                "w-full min-w-0 sm:flex-1",
              )}
            >
              {t("Start camera", "Kamera starten")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <PrivacyPolicyDocumentDialog
        open={showPrivacyDoc}
        onOpenChange={setShowPrivacyDoc}
      />
      <TermsDocumentDialog open={showTermsDoc} onOpenChange={setShowTermsDoc} />
    </div>
  );
};

export default StartPage;
