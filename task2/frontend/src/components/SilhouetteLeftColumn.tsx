import { CameraStepCard } from "@/components/CameraStepCard";
import { type ExhibitStepIndex } from "@/lib/exhibitFlow";
import { cn } from "@/lib/utils";
import { t } from "@/lib/localization";

/** Same as `CameraPage` / step-1 list body (`CAMERA_STEP1_DESC_INSTRUCTION_CLASS`). */
const stepInstructionListTextClass =
  "text-[1.0625rem] leading-snug md:text-[1.4rem]";

/** Same card root as `CameraPage` `cameraStepCardClassName`. */
const cameraStepCardClassName =
  "h-full max-w-full min-h-0 min-w-0 w-full";

const SILHOUETTE_STEP2_INSTRUCTION_LINES: { en: string; de: string }[] = [
  {
    en: "Transform your silhouette with body parts and accessories from the movie!",
    de: "Stelle deine Silhouette mit Figuren-Teilen und Accessoires aus dem Film dar!",
  },
  {
    en: 'Click on the "SURPRISE ME" button to apply a unique mix of accessories or choose your own combination.',
    de: 'Tippe auf "UEBERRASCH MICH" um eine zufaellige Accessoire-Kombination anzuwenden oder waehle deine eigenen Teile selbst aus.',
  },
];

const greyedStepClass =
  "opacity-50 grayscale transition-[filter,opacity] duration-200";

/**
 * Left column for `/silhouette` — same structure and typography as `/camera`’s left panel
 * (`CameraStepCard` + list styling); step 1 done, step 2 current with instructions, 3–4 greyed.
 */
export function SilhouetteLeftColumn() {
  return (
    <aside className="flex min-h-0 min-w-0 flex-col overflow-hidden lg:h-full">
      <div
        className={cn(
          "exhibit-panel flex h-full min-h-0 w-full min-w-0 max-h-full flex-col overflow-hidden rounded-2xl p-2 sm:p-3 md:p-4",
          "max-lg:max-h-[30vh] lg:max-h-none",
        )}
      >
        <div className="exhibit-title flex min-h-0 w-full min-w-0 flex-1 flex-col gap-2 overflow-y-auto overscroll-contain [scrollbar-gutter:stable] sm:gap-2 md:gap-3">
          <div
            className={cn("w-full min-w-0 max-w-full shrink-0", greyedStepClass)}
          >
            <CameraStepCard
              stepIndex={0}
              className={cameraStepCardClassName}
            />
          </div>

          <div
            className="w-full min-w-0 max-w-full shrink-0"
            aria-current="step"
          >
            <CameraStepCard
              stepIndex={1}
              className={cameraStepCardClassName}
            >
              <ol
                className={cn(
                  "exhibit-title mt-2 list-inside list-decimal space-y-1.5 border-t border-border/50 pl-0 pt-2 text-left text-film-black [overflow-wrap:anywhere] marker:font-medium",
                  stepInstructionListTextClass,
                )}
              >
                {SILHOUETTE_STEP2_INSTRUCTION_LINES.map((line, i) => (
                  <li key={i}>{t(line.en, line.de)}</li>
                ))}
              </ol>
            </CameraStepCard>
          </div>

          {([2, 3] as const).map((stepIndex) => (
            <div
              key={stepIndex}
              className={cn(
                "w-full min-w-0 max-w-full shrink-0",
                greyedStepClass,
              )}
            >
              <CameraStepCard
                stepIndex={stepIndex as ExhibitStepIndex}
                className={cameraStepCardClassName}
              />
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
