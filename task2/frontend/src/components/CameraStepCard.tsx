import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { t } from "@/lib/localization";
import { EXHIBIT_STEPS, EXHIBIT_STEP_ACCENTS, type ExhibitStepIndex } from "@/lib/exhibitFlow";

/**
 * Step card for `/camera` only — same structure and copy source as `ExhibitStepCard`, but
 * typography is local so you can resize without affecting the start page strip.
 */
type CameraStepCardProps = {
  stepIndex: ExhibitStepIndex;
  className?: string;
  id?: string;
  children?: ReactNode;
};

/** "Step n" / "Schritt n" — edit font size here for the camera page */
const cameraStepLabelClass =
  "text-[0.9375rem] font-semibold uppercase leading-tight tracking-wide md:text-[1.5rem]";

/** Step title (e.g. Take your photo) */
const cameraStepHeadingClass =
  "text-[1.1875rem] font-semibold leading-snug md:text-[2rem]";

/** Description under the title (e.g. Capture your silhouette.) */
const cameraStepDescriptionClass =
  "text-[1.0625rem] leading-snug md:text-[1.4rem]";

export function CameraStepCard({
  stepIndex,
  className,
  id,
  children,
}: CameraStepCardProps) {
  const step = EXHIBIT_STEPS[stepIndex];
  const accent = EXHIBIT_STEP_ACCENTS[stepIndex];

  return (
    <div
      id={id}
      className={cn(
        "flex flex-col rounded-xl border border-border text-left text-film-black",
        accent,
        "px-2 py-1.5",
        className,
      )}
    >
      <p className={cn("min-w-0", cameraStepLabelClass)}>
        {t(`Step ${stepIndex + 1}`, `Schritt ${stepIndex + 1}`)}
      </p>
      <p className={cn("min-w-0 mt-0.5", cameraStepHeadingClass)}>
        {t(step.titleEn, step.titleDe)}
      </p>
      <p className={cn("min-w-0 mt-0.5", cameraStepDescriptionClass)}>
        {t(step.descriptionEn, step.descriptionDe)}
      </p>
      {children}
    </div>
  );
}
