import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { t } from "@/lib/localization";
import {
  ArrowLeft,
  CheckCircle2,
  QrCode,
  ScanLine,
  X,
  ZoomIn,
} from "lucide-react";
import { toast } from "@/hooks/use-toast";

type CompositeItem = {
  id: string;
  originalBgId: string;
  title: string;
  url: string;
};

type ShareInfo = {
  shareUrl: string;
  expiresAt: string;
};

const ConfirmationPage = () => {
  const navigate = useNavigate();
  const [compositedItems, setCompositedItems] = useState<CompositeItem[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [enlargedImage, setEnlargedImage] = useState<CompositeItem | null>(null);
  const [shareInfo, setShareInfo] = useState<ShareInfo | null>(null);
  const [isCreatingShare, setIsCreatingShare] = useState(false);

  const isShareLocked = shareInfo !== null;

  useEffect(() => {
    const silhouetteDataStr = sessionStorage.getItem("silhouetteData");
    const bgIdsStr = sessionStorage.getItem("selectedBackgroundIds");
    const bgObjectsStr = sessionStorage.getItem("selectedBackgrounds");
    const preCompositesStr = sessionStorage.getItem("preComposites");

    if (!silhouetteDataStr || !bgIdsStr) {
      navigate("/selection");
      return;
    }

    const silhouette = JSON.parse(silhouetteDataStr);
    const bgIds: string[] = JSON.parse(bgIdsStr);
    const bgObjects = bgObjectsStr ? JSON.parse(bgObjectsStr) : [];
    const preComposites: Record<string, string> = preCompositesStr
      ? JSON.parse(preCompositesStr)
      : {};

    const generateComposites = async () => {
      setIsProcessing(true);
      try {
        const promises = bgIds.map(async (bgId) => {
          const bgObj = bgObjects.find((b: { id: string }) => b.id === bgId);
          const title = bgObj
            ? bgObj.title
            : t(`Background ${bgId}`, `Hintergrund ${bgId}`);

          if (preComposites[bgId]) {
            return {
              id: `pre_${bgId}`,
              originalBgId: bgId,
              title,
              url: preComposites[bgId],
            } as CompositeItem;
          }

          const res = await fetch("http://localhost:8000/composite", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              silhouette_id: silhouette.id,
              background_id: bgId,
            }),
          });
          if (!res.ok) throw new Error("Composition failed");
          const data = await res.json();

          return {
            id: data.result.id,
            originalBgId: bgId,
            title,
            url: `http://localhost:8000${data.result.url}`,
          } as CompositeItem;
        });

        const results = await Promise.all(promises);
        setCompositedItems(results);
      } catch (error) {
        console.error(error);
        toast({
          title: t("Error", "Fehler"),
          description: t(
            "Failed to generate some composite images.",
            "Einige Kompositionsbilder konnten nicht erzeugt werden.",
          ),
          variant: "destructive",
        });
      } finally {
        setIsProcessing(false);
      }
    };

    generateComposites();
  }, [navigate]);

  const removeItem = (id: string) => {
    if (isShareLocked) {
      return;
    }

    setCompositedItems((prev) => prev.filter((item) => item.id !== id));
  };

  const handleCreateShare = async () => {
    if (isShareLocked) {
      return;
    }

    if (compositedItems.length === 0) {
      toast({
        title: t("No images", "Keine Bilder"),
        description: t(
          "Generate at least one composition first.",
          "Bitte zuerst mindestens eine Komposition erzeugen.",
        ),
        variant: "destructive",
      });
      return;
    }

    setIsCreatingShare(true);
    try {
      const res = await fetch("http://localhost:8000/share/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ids: compositedItems.map((item) => item.id),
          ttl_minutes: 15,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(
          err.detail ||
            t(
              "Failed to create share link",
              "Freigabe-Link konnte nicht erstellt werden",
            ),
        );
      }

      const data = await res.json();
      setShareInfo({ shareUrl: data.share_url, expiresAt: data.expires_at });
    } catch (error: unknown) {
      toast({
        title: t("Share link failed", "Freigabe-Link fehlgeschlagen"),
        description:
          error instanceof Error
            ? error.message
            : t(
                "Could not create QR share link.",
                "QR-Freigabe-Link konnte nicht erstellt werden.",
              ),
        variant: "destructive",
      });
    } finally {
      setIsCreatingShare(false);
    }
  };

  const handleStartOver = async () => {
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

  return (
    <div className="h-full min-h-0 exhibit-shell flex flex-col overflow-hidden overflow-x-hidden px-4 py-4 md:px-8 md:py-7">
      <div className="mb-4 flex items-center justify-between gap-3">
        {!isShareLocked ? (
          <Button
            variant="outline"
            size="xl"
            className="exhibit-panel gap-2 border-film-blue/20 bg-white/70"
            onClick={() => navigate("/selection")}
          >
            <ArrowLeft className="h-5 w-5" />
            {t("Back to Selection", "Zurueck zur Auswahl")}
          </Button>
        ) : (
          <span className="film-tag">
            {t("Share Locked", "Freigabe gesperrt")}
          </span>
        )}
        <span className="film-tag">{t("Step 4 of 4", "Schritt 4 von 4")}</span>
      </div>

      <header className="exhibit-panel mb-4 rounded-2xl p-4 md:p-6">
        <h1 className="exhibit-title text-3xl md:text-5xl">
          {t("Share Your Artwork", "Kunstwerk teilen")}
        </h1>
        <p className="mt-2 text-base text-foreground/90 md:text-lg">
          {t(
            "Review your selected scenes, then create one QR code to take your gallery with you.",
            "Pruefe deine ausgewaehlten Szenen und erstelle dann einen QR-Code, um deine Galerie mitzunehmen.",
          )}
        </p>
      </header>

      <div className="grid flex-1 min-h-0 grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_390px]">
        <section className="exhibit-panel flex min-h-0 flex-col rounded-2xl p-4 md:p-5">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h2 className="text-xl font-semibold md:text-2xl">
              {t("Your Gallery", "Deine Galerie")}
            </h2>
            <span className="rounded-full bg-film-blue/10 px-3 py-1 text-sm font-semibold text-film-blue">
              {compositedItems.length} {t("scene(s)", "Szene(n)")}
            </span>
          </div>

          {isProcessing ? (
            <div className="flex flex-1 items-center justify-center rounded-2xl bg-muted/50 p-6 text-center">
              <p className="animate-pulse text-lg font-medium text-muted-foreground">
                {t("Generating artwork...", "Bilder werden erstellt...")}
              </p>
            </div>
          ) : compositedItems.length === 0 ? (
            <div className="flex flex-1 flex-col items-center justify-center gap-4 rounded-2xl border border-dashed border-border bg-muted/40 p-6 text-center">
              <p className="text-base text-muted-foreground md:text-lg">
                {t(
                  "No selected scenes found.",
                  "Keine ausgewaehlten Szenen gefunden.",
                )}
              </p>
              {!isShareLocked ? (
                <Button onClick={() => navigate("/selection")} size="xl">
                  {t("Return to Selection", "Zurueck zur Auswahl")}
                </Button>
              ) : null}
            </div>
          ) : (
            <div className="soft-scroll w-full max-h-[56vh] overflow-y-auto pr-1">
              <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-4">
                {compositedItems.map((item) => (
                  <Card
                    key={item.id}
                    className="relative h-[352px] overflow-hidden rounded-xl border bg-white/85 shadow-md"
                  >
                    <div className="h-[276px] w-full overflow-hidden bg-muted">
                      <img
                        src={item.url}
                        alt={item.title}
                        className="h-full w-full object-cover"
                      />
                    </div>

                    <div className="flex h-[76px] items-center justify-between gap-2 p-2.5">
                      <span className="truncate text-sm font-medium md:text-base">
                        {item.title}
                      </span>
                      <div className="flex gap-1">
                        <button
                          type="button"
                          className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                          onClick={() => setEnlargedImage(item)}
                          title={t("Zoom", "Vergruessern")}
                        >
                          <ZoomIn className="h-5 w-5" />
                        </button>
                        <button
                          type="button"
                          className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive disabled:opacity-40"
                          onClick={() => removeItem(item.id)}
                          title={t("Remove", "Entfernen")}
                          disabled={isShareLocked}
                        >
                          <X className="h-5 w-5" />
                        </button>
                      </div>
                    </div>

                    {isShareLocked ? (
                      <div className="absolute left-2 top-2 rounded-full bg-film-green px-2 py-0.5 text-xs font-semibold text-white">
                        {t("Locked", "Gesperrt")}
                      </div>
                    ) : null}
                  </Card>
                ))}
              </div>
            </div>
          )}
        </section>

        <aside className="exhibit-panel flex min-h-0 flex-col rounded-2xl p-4 md:p-5">
          {!shareInfo ? (
            <>
              <h2 className="text-xl font-semibold md:text-2xl">
                {t("Final Share Step", "Letzter Freigabe-Schritt")}
              </h2>
              <p className="mt-2 text-sm text-muted-foreground md:text-base">
                {t(
                  "Create one QR code for all selected scenes.",
                  "Erstelle einen QR-Code fuer alle ausgewaehlten Szenen.",
                )}
              </p>

              <div className="mt-4 space-y-2 rounded-xl border border-film-blue/20 bg-film-blue/10 p-4 text-sm md:text-base">
                <p className="font-semibold text-film-blue">
                  {t("1. Review images", "1. Bilder pruefen")}
                </p>
                <p className="font-semibold text-film-blue">
                  {t('2. Tap "Create QR"', '2. "QR erstellen" antippen')}
                </p>
                <p className="font-semibold text-film-blue">
                  {t("3. Scan with phone", "3. Mit Smartphone scannen")}
                </p>
              </div>

              <div className="mt-4 rounded-xl border border-film-red/20 bg-film-red/10 p-4 text-sm text-film-black md:text-base">
                {t(
                  "QR link expires in 15 minutes and all generated data is deleted after 15 minutes.",
                  "Der QR-Link laeuft nach 15 Minuten ab und alle generierten Daten werden nach 15 Minuten geloescht.",
                )}
              </div>

              <div className="mt-auto space-y-3 pt-6">
                <Button
                  size="xl"
                  onClick={handleCreateShare}
                  disabled={isCreatingShare || compositedItems.length === 0}
                  className="cta-step-4 w-full font-semibold"
                >
                  <QrCode className="h-5 w-5" />
                  {isCreatingShare
                    ? t("Creating QR...", "QR wird erstellt...")
                    : t("Create QR Share", "QR-Freigabe erstellen")}
                </Button>
                <Button
                  variant="outline"
                  size="xl"
                  className="w-full"
                  onClick={handleStartOver}
                >
                  {t(
                    "Start over and clear data",
                    "Neu starten und Daten loeschen",
                  )}
                </Button>
              </div>
            </>
          ) : (
            <>
              <div className="mb-3 inline-flex items-center gap-2 rounded-full bg-film-green/10 px-3 py-1 text-sm font-semibold text-film-green">
                <CheckCircle2 className="h-4 w-4" />
                {t("QR ready and locked", "QR bereit und gesperrt")}
              </div>
              <h2 className="text-xl font-semibold md:text-2xl">
                {t("Scan and save", "Scannen und speichern")}
              </h2>
              <p className="mt-1 text-sm text-muted-foreground md:text-base">
                {t(
                  "Use your phone camera to open and save your artwork.",
                  "Nutze deine Smartphone-Kamera, um dein Kunstwerk zu speichern.",
                )}
              </p>

              <div className="mt-4 rounded-2xl border bg-white/86 p-4 shadow-lg">
                <div className="flex items-center justify-center">
                  <img
                    src={`https://api.qrserver.com/v1/create-qr-code/?size=320x320&data=${encodeURIComponent(shareInfo.shareUrl)}`}
                    alt={t("Share QR code", "QR-Code teilen")}
                    className="h-64 w-64"
                  />
                </div>
                <div className="mt-3 space-y-2 text-sm md:text-base">
                  <p className="flex items-start gap-2">
                    <ScanLine className="mt-0.5 h-4 w-4 shrink-0 text-film-blue" />
                    {t(
                      "1. Open phone camera and scan the QR code.",
                      "1. Smartphone-Kamera oeffnen und QR-Code scannen.",
                    )}
                  </p>
                  <p className="flex items-start gap-2">
                    <ScanLine className="mt-0.5 h-4 w-4 shrink-0 text-film-blue" />
                    {t(
                      "2. Tap the link to open your gallery.",
                      "2. Link antippen, um die Galerie zu oeffnen.",
                    )}
                  </p>
                </div>
              </div>

              <div className="mt-4 rounded-xl border border-film-red/20 bg-film-red/10 p-4 text-sm text-film-black md:text-base">
                {t(
                  `QR expires at ${new Date(shareInfo.expiresAt).toLocaleTimeString()}. All generated data is deleted after 15 minutes.`,
                  `QR gueltig bis ${new Date(shareInfo.expiresAt).toLocaleTimeString()}. Alle generierten Daten werden nach 15 Minuten geloescht.`,
                )}
              </div>

              <div className="mt-auto pt-6">
                <Button
                  size="xl"
                  onClick={handleStartOver}
                  className="w-full bg-film-red text-white hover:bg-film-red/90"
                >
                  {t("Close and clear data", "Schliessen und Daten loeschen")}
                </Button>
              </div>
            </>
          )}
        </aside>
      </div>

      <Dialog open={!!enlargedImage} onOpenChange={() => setEnlargedImage(null)}>
        <DialogContent className="max-w-2xl border-none bg-transparent px-0 shadow-none">
          {enlargedImage ? (
            <img
              src={enlargedImage.url}
              alt={enlargedImage.title}
              className="h-auto w-full rounded"
            />
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default ConfirmationPage;
