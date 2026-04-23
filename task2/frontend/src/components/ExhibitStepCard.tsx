import { cn } from "@/lib/utils";
import { t } from "@/lib/localization";
import { EXHIBIT_STEPS, EXHIBIT_STEP_ACCENTS, type ExhibitStepIndex } from "@/lib/exhibitFlow";

type Size = "default" | "large";

type ExhibitStepCardProps = {
  stepIndex: ExhibitStepIndex;
  size?: Size;
  className?: string;
};

const defaultTitle =
  "text-[0.9375rem] font-semibold uppercase leading-tight tracking-wide md:text-[1.0625rem]";
const defaultHeading =
  "text-[1.1875rem] font-semibold leading-snug md:text-[1.375rem]";
const defaultDescription = "text-[1.0625rem] leading-snug md:text-[1.1875rem]";

const largeLabel =
  "text-xs font-semibold uppercase leading-tight tracking-wide sm:text-sm md:text-base";
const largeHeading =
  "text-sm font-semibold leading-snug sm:text-lg sm:leading-snug md:text-2xl";
const largeDescription =
  "text-xs leading-snug sm:text-base sm:leading-snug md:text-lg";

/**
 * One step card — same look as the Start page strip; `large` is for side panels on flow pages.
 */
export function ExhibitStepCard({
  stepIndex,
  size = "default",
  className,
}: ExhibitStepCardProps) {
  const step = EXHIBIT_STEPS[stepIndex];
  const accent = EXHIBIT_STEP_ACCENTS[stepIndex];

  return (
    <div
      className={cn(
        "flex flex-col rounded-xl border border-border text-left text-film-black shadow-[0_10px_34px_hsl(var(--film-black)_/_0.2)]",
        accent,
        size === "default" && "px-2 py-1.5",
        size === "large" &&
          "gap-0.5 rounded-2xl px-2 py-2 sm:gap-1 sm:px-4 sm:py-3 md:gap-1.5 md:px-6 md:py-5",
        className,
      )}
    >
      <p
        className={cn("min-w-0", size === "default" ? defaultTitle : largeLabel)}
      >
        {t(`Step ${stepIndex + 1}`, `Schritt ${stepIndex + 1}`)}
      </p>
      <p
        className={cn(
          "min-w-0 mt-0.5",
          size === "default" ? defaultHeading : largeHeading,
        )}
      >
        {t(step.titleEn, step.titleDe)}
      </p>
      <p
        className={cn(
          "min-w-0 mt-0.5",
          size === "default" ? defaultDescription : largeDescription,
        )}
      >
        {t(step.descriptionEn, step.descriptionDe)}
      </p>
    </div>
  );
}
