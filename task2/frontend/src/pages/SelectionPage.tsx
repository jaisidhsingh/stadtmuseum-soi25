import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { t } from "@/lib/localization";
import { ArrowLeft, Check, Circle, LogOut, X } from "lucide-react";

type Background = {
  id: string;
  title: string;
  url: string;
  positions?: [number, number][];
  max_w?: number;
  max_h?: number;
  bg_w?: number;
  bg_h?: number;
  silhouette_color?: [number, number, number];
};

type SilhouetteData = {
  id: string;
  url: string;
};

const SelectionPage = () => {
  const navigate = useNavigate();
  const [selectedCards, setSelectedCards] = useState<string[]>([]);
  const [backgrounds, setBackgrounds] = useState<Background[]>([]);
  const [previewBgId, setPreviewBgId] = useState<string | null>(null);
  const [silhouette, setSilhouette] = useState<SilhouetteData | null>(null);

  useEffect(() => {
    const storedSil = sessionStorage.getItem("silhouetteData");
    if (storedSil) {
      setSilhouette(JSON.parse(storedSil));
    } else {
      navigate("/camera");
    }

    const fetchBackgrounds = async () => {
      try {
        const res = await fetch("http://localhost:8000/backgrounds");
        if (res.ok) {
          const data = await res.json();
          setBackgrounds(data);
          if (data.length > 0) {
            setPreviewBgId(data[0].id);
          }
        }
      } catch (error) {
        console.error("Failed to fetch backgrounds", error);
      }
    };

    fetchBackgrounds();
  }, [navigate]);

  const toggleSelection = (id: string) => {
    setSelectedCards((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id],
    );
  };

  const handleProceed = () => {
    sessionStorage.setItem(
      "selectedBackgroundIds",
      JSON.stringify(selectedCards),
    );
    const selectedBackgrounds = backgrounds.filter((bg) =>
      selectedCards.includes(bg.id),
    );
    sessionStorage.setItem(
      "selectedBackgrounds",
      JSON.stringify(selectedBackgrounds),
    );
    sessionStorage.removeItem("preComposites");
    navigate("/confirmation");
  };

  const handleExitSession = async () => {
    try {
      await fetch("http://localhost:8000/clear", { method: "POST" });
    } catch (error) {
      console.error("Failed to clear backend session", error);
    }

    sessionStorage.removeItem("selectedBackgroundIds");
    sessionStorage.removeItem("selectedBackgrounds");
    sessionStorage.removeItem("preComposites");
    sessionStorage.removeItem("silhouetteData");
    sessionStorage.removeItem("capturedImage");
    sessionStorage.removeItem("silhouetteStyles");
    navigate("/");
  };

  const currentPreviewBg = useMemo(
    () => backgrounds.find((background) => background.id === previewBgId),
    [backgrounds, previewBgId],
  );

  const selectedBackgrounds = useMemo(
    () =>
      backgrounds.filter((background) => selectedCards.includes(background.id)),
    [backgrounds, selectedCards],
  );

  const isCurrentPreviewSelected = currentPreviewBg
    ? selectedCards.includes(currentPreviewBg.id)
    : false;

  return (
    <div className="h-full min-h-0 exhibit-shell flex flex-col overflow-hidden overflow-x-hidden px-4 py-2 md:px-8 md:py-4">
      <div className="mb-3 flex items-center justify-between gap-4">
        <Button
          variant="outline"
          size="xl"
          className="exhibit-panel gap-2 border-film-blue/20 bg-white/70"
          onClick={() => navigate("/silhouette")}
        >
          <ArrowLeft className="h-5 w-5" />
          {t("Back to Silhouette", "Zurueck zur Silhouette")}
        </Button>
        <div className="flex flex-wrap items-center justify-end gap-3">
          <span className="film-tag">{t("Step 3 of 4", "Schritt 3 von 4")}</span>
          <Button
            variant="outline"
            size="xl"
            onClick={handleExitSession}
            className="border-film-red/40 bg-white/80 text-film-red hover:bg-film-red/10 hover:text-film-red"
          >
            <LogOut className="h-5 w-5" />
            {t("Exit Session", "Sitzung beenden")}
          </Button>
        </div>
      </div>

      <div className="grid flex-1 min-h-0 grid-cols-1 gap-3 xl:grid-cols-[340px_minmax(0,1fr)_360px]">
        <aside className="exhibit-panel flex min-h-0 flex-col rounded-2xl p-4 md:p-5">
          <h1 className="exhibit-title text-2xl md:text-3xl">
            {t("Pick Backgrounds to Share", "Hintergruende zum Teilen waehlen")}
          </h1>
          <p className="mt-2 text-sm text-foreground/90 md:text-base">
            {t(
              "Choose scenes for your final QR gallery.",
              "Waehle Szenen fuer deine finale QR-Galerie.",
            )}
          </p>

          <div className="mt-4 space-y-2 text-sm md:text-base">
            <div className="rounded-xl border border-film-blue/20 bg-film-blue/10 px-3 py-2 font-medium text-film-blue">
              {t(
                "1. Tap thumbnail to preview.",
                "1. Miniatur antippen fuer Vorschau.",
              )}
            </div>
            <div className="rounded-xl border border-film-green/20 bg-film-green/10 px-3 py-2 font-medium text-film-green">
              {t(
                "2. Select scene for QR export.",
                "2. Szene fuer QR-Export auswaehlen.",
              )}
            </div>
            <div className="rounded-xl border border-film-red/20 bg-film-red/10 px-3 py-2 font-medium text-film-red">
              {t(
                "3. Continue to review and share.",
                "3. Weiter zu Vorschau und Teilen.",
              )}
            </div>
          </div>
        </aside>

        <section className="exhibit-panel flex min-h-0 flex-col rounded-2xl p-3 md:p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-xl font-semibold md:text-2xl">
                {t("Live Scene Preview", "Szenen-Vorschau")}
              </h2>
              <p className="text-sm text-muted-foreground md:text-base">
                {t(
                  "Your moving silhouette stays exactly as designed.",
                  "Deine bewegte Silhouette bleibt exakt wie gestaltet.",
                )}
              </p>
            </div>
            <Button
              size="xl"
              disabled={!currentPreviewBg}
              onClick={() =>
                currentPreviewBg ? toggleSelection(currentPreviewBg.id) : null
              }
              className={
                isCurrentPreviewSelected
                  ? "bg-film-green text-white hover:bg-film-green/90"
                  : "cta-step-3 text-white"
              }
            >
              {isCurrentPreviewSelected ? (
                <>
                  <Check className="h-5 w-5" />
                  {t("Selected for QR", "Fuer QR ausgewaehlt")}
                </>
              ) : (
                <>
                  <Circle className="h-5 w-5" />
                  {t("Select this scene", "Szene auswaehlen")}
                </>
              )}
            </Button>
          </div>

          <div className="flex flex-1 min-h-0 items-center justify-center">
            {currentPreviewBg && silhouette ? (
              <div className="relative flex w-full max-w-6xl justify-center">
                {(() => {
                  const bw = currentPreviewBg.bg_w || 1920;
                  const bh = currentPreviewBg.bg_h || 1080;
                  const maxWidthFromHeightVh = (58 * bw) / bh;

                  return (
                    <div
                      className={`relative h-full w-full overflow-hidden rounded-2xl border-4 shadow-2xl transition-all ${isCurrentPreviewSelected ? "border-film-green ring-4 ring-film-green/30" : "border-border"}`}
                      style={{
                        width: `min(100%, ${maxWidthFromHeightVh}vh)`,
                        aspectRatio: `${bw} / ${bh}`,
                      }}
                    >
                      <img
                        src={`http://localhost:8000${currentPreviewBg.url}`}
                        alt={currentPreviewBg.title}
                        className="absolute inset-0 h-full w-full object-contain"
                      />

                      {(() => {
                        const pos = currentPreviewBg.positions?.[0] || [
                          bw / 2,
                          bh,
                        ];
                        const mw = currentPreviewBg.max_w;
                        const mh = currentPreviewBg.max_h;
                        const tint = currentPreviewBg.silhouette_color || [
                          0, 0, 0,
                        ];
                        const tintCss = `rgb(${tint[0]}, ${tint[1]}, ${tint[2]})`;
                        const silhouetteUrl = `http://localhost:8000${silhouette.url}`;
                        const widthPct = mw ? `${(mw / bw) * 100}%` : "30%";
                        const heightPct = mh ? `${(mh / bh) * 100}%` : "60%";

                        return (
                          <div
                            aria-label={t("My silhouette", "Meine Silhouette")}
                            className="absolute pointer-events-none transition-all duration-500"
                            style={{
                              left: `${(pos[0] / bw) * 100}%`,
                              top: `${(pos[1] / bh) * 100}%`,
                              transform: "translate(-50%, -100%)",
                              width: widthPct,
                              height: heightPct,
                              backgroundColor: tintCss,
                              WebkitMaskImage: `url(${silhouetteUrl})`,
                              maskImage: `url(${silhouetteUrl})`,
                              WebkitMaskRepeat: "no-repeat",
                              maskRepeat: "no-repeat",
                              WebkitMaskPosition: "center bottom",
                              maskPosition: "center bottom",
                              WebkitMaskSize: "contain",
                              maskSize: "contain",
                            }}
                          />
                        );
                      })()}

                      <div className="absolute left-3 top-3 rounded-full bg-black/55 px-4 py-1.5 text-sm font-semibold text-white md:text-base">
                        {t("Preview", "Vorschau")}: {currentPreviewBg.title}
                      </div>
                      <div className="absolute bottom-3 left-3 rounded-full bg-black/55 px-4 py-1.5 text-xs font-medium text-white md:text-sm">
                        {t(
                          "Only selected scenes go to QR export.",
                          "Nur ausgewaehlte Szenen gehen in den QR-Export.",
                        )}
                      </div>
                      {currentPreviewBg ? (
                        <button
                          type="button"
                          onClick={() => toggleSelection(currentPreviewBg.id)}
                          className="absolute right-3 top-3 rounded-full bg-white/85 p-2 text-foreground shadow-lg transition-colors hover:bg-white"
                          title={
                            isCurrentPreviewSelected
                              ? t("Unselect scene", "Szene abwaehlen")
                              : t("Select scene", "Szene auswaehlen")
                          }
                        >
                          {isCurrentPreviewSelected ? (
                            <Check className="h-6 w-6 text-film-green" />
                          ) : (
                            <Circle className="h-6 w-6 text-muted-foreground" />
                          )}
                        </button>
                      ) : null}
                    </div>
                  );
                })()}
              </div>
            ) : (
              <div className="flex aspect-[16/9] w-full max-w-5xl items-center justify-center rounded-2xl bg-muted/60">
                <span className="text-lg text-muted-foreground">
                  {t("Loading scene preview...", "Szenen-Vorschau laedt...")}
                </span>
              </div>
            )}
          </div>
        </section>

        <aside className="exhibit-panel flex min-h-0 flex-col rounded-2xl p-4 md:p-5">
          <h2 className="text-xl font-semibold">
            {t("Export Queue", "Export-Liste")}
          </h2>
          <p className="mt-1 text-sm text-muted-foreground md:text-base">
            {t(
              "Selected scenes become your QR gallery on the next page.",
              "Ausgewaehlte Szenen erscheinen auf der naechsten QR-Seite.",
            )}
          </p>

          <div className="mt-4 rounded-xl border border-film-blue/20 bg-film-blue/10 p-4">
            <p className="text-3xl font-bold text-film-blue">
              {selectedCards.length}
            </p>
            <p className="text-sm font-semibold uppercase tracking-wide text-film-blue">
              {t("Selected", "Ausgewaehlt")}
            </p>
            <p className="mt-1 text-sm text-foreground/80">
              {t("of", "von")} {backgrounds.length}{" "}
              {t("available scenes", "verfuegbaren Szenen")}
            </p>
          </div>

          <div className="soft-scroll mt-4 min-h-[120px] max-h-[30vh] space-y-2 overflow-y-auto pr-1">
            {selectedBackgrounds.length === 0 ? (
              <div className="rounded-xl border border-dashed border-border bg-muted/40 p-4 text-sm text-muted-foreground">
                {t("No scenes selected yet.", "Noch keine Szenen ausgewaehlt.")}
              </div>
            ) : (
              selectedBackgrounds.map((background) => (
                <div
                  key={background.id}
                  className={`flex items-center justify-between rounded-xl border bg-white/70 px-3 py-2 ${previewBgId === background.id ? "border-film-blue bg-film-blue/10" : ""}`}
                >
                  <button
                    type="button"
                    onClick={() => setPreviewBgId(background.id)}
                    className="min-w-0 flex-1 pr-2 text-left"
                  >
                    <span className="block truncate text-sm font-medium">
                      {background.title}
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      toggleSelection(background.id);
                    }}
                    className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
                    title={t("Remove from export", "Aus Export entfernen")}
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>
              ))
            )}
          </div>

          <div className="mt-4 border-t pt-4">
            <Button
              size="xl"
              className="cta-step-4 w-full font-semibold"
              onClick={handleProceed}
              disabled={selectedCards.length === 0}
            >
              {t("Continue to review and QR", "Weiter zu Vorschau und QR")}
            </Button>
            {selectedCards.length === 0 ? (
              <p className="mt-2 text-center text-sm text-muted-foreground">
                {t(
                  "Select at least one scene to continue.",
                  "Waehle mindestens eine Szene, um fortzufahren.",
                )}
              </p>
            ) : null}
          </div>
        </aside>
      </div>

      <section className="exhibit-panel mt-3 rounded-2xl p-3 md:p-4">
        <div className="mb-3 flex items-center gap-3">
          <h2 className="text-lg font-semibold md:text-xl">
            {t("Scene Strip", "Szenen-Leiste")}
          </h2>
        </div>
        <div className="custom-scrollbar flex w-full snap-x snap-mandatory items-start gap-4 overflow-x-auto pb-2">
          {backgrounds.map((background) => {
            const selected = selectedCards.includes(background.id);
            const isPreviewing = previewBgId === background.id;

            return (
              <button
                type="button"
                key={background.id}
                className="group snap-start text-left"
                onClick={() => setPreviewBgId(background.id)}
              >
                <div
                  className={`relative h-[124px] w-[220px] overflow-hidden rounded-xl border-4 transition-all duration-200 ${isPreviewing ? "border-film-blue shadow-xl" : "border-transparent opacity-80 group-hover:opacity-100"} ${selected ? "ring-2 ring-film-green/40" : ""}`}
                >
                  <img
                    src={`http://localhost:8000${background.url}`}
                    alt={background.title}
                    className="h-full w-full object-cover"
                  />
                  {isPreviewing ? (
                    <span className="absolute left-2 top-2 rounded-full bg-film-blue px-2 py-0.5 text-xs font-semibold text-white">
                      {t("Preview", "Vorschau")}
                    </span>
                  ) : null}
                  {selected ? (
                    <span className="absolute right-2 top-2 rounded-full bg-film-green px-2 py-0.5 text-xs font-semibold text-white">
                      {t("Selected", "Ausgewaehlt")}
                    </span>
                  ) : null}
                </div>
                <p
                  className={`mt-1.5 max-w-[220px] truncate text-sm font-medium ${isPreviewing ? "text-film-blue" : "text-foreground/80"}`}
                >
                  {background.title}
                </p>
              </button>
            );
          })}
        </div>
      </section>
    </div>
  );
};

export default SelectionPage;
