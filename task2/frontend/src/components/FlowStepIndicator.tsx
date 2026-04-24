import { flowCtaLabelTypographyClassName } from "@/lib/flowCtaClassNames";
import { EXHIBIT_STEP_ACCENTS, type ExhibitStepIndex } from "@/lib/exhibitFlow";
import { t } from "@/lib/localization";
import { cn } from "@/lib/utils";

const FILM_HSL = {
  0: "var(--film-blue)",
  1: "var(--film-green)",
  2: "var(--film-red)",
  3: "var(--film-yellow)",
} as const satisfies Record<ExhibitStepIndex, string>;

const STEP_I18N: { en: string; de: string }[] = [
  { en: "Step 1 of 4", de: "Schritt 1 / 4" },
  { en: "Step 2 of 4", de: "Schritt 2 / 4" },
  { en: "Step 3 of 4", de: "Schritt 3 / 4" },
  { en: "Step 4 of 4", de: "Schritt 4 von 4" },
];

type FlowStepIndicatorProps = {
  activeStepIndex: ExhibitStepIndex;
  className?: string;
};

/**
 * Rounded “pill” in the same scale family as the EXIT CTA, with a 4-slice disc that fills
 * by ¼ per step; background tint matches the active step card (`EXHIBIT_STEP_ACCENTS`).
 */
export function FlowStepIndicator({
  activeStepIndex,
  className,
}: FlowStepIndicatorProps) {
  const n = (activeStepIndex + 1) * 90;
  const c = FILM_HSL[activeStepIndex];
  const label = STEP_I18N[activeStepIndex];
  const conic = `conic-gradient(from -90deg, hsl(${c} / 0.9) 0deg, hsl(${c} / 0.9) ${n}deg, hsl(0 0% 0% / 0.1) ${n}deg, hsl(0 0% 0% / 0.1) 360deg)`;

  return (
    <div
      className={cn(
        "exhibit-title inline-flex h-14 w-fit max-w-full min-w-0 items-center justify-center gap-3 rounded-2xl border border-border px-2 text-film-black shadow-sm md:h-16 md:px-3",
        EXHIBIT_STEP_ACCENTS[activeStepIndex],
        className,
      )}
    >
      <div
        className="relative h-6 w-6 shrink-0 overflow-hidden rounded-full border border-border md:h-7 md:w-7"
        aria-hidden
      >
        <div className="absolute inset-0 rounded-full" style={{ background: conic }} />
        <svg
          className="absolute inset-0 h-full w-full text-film-black/25"
          viewBox="0 0 100 100"
          fill="none"
          aria-hidden
        >
          <line
            x1="50"
            y1="0"
            x2="50"
            y2="100"
            stroke="currentColor"
            strokeWidth={2.5}
            vectorEffect="non-scaling-stroke"
          />
          <line
            x1="0"
            y1="50"
            x2="100"
            y2="50"
            stroke="currentColor"
            strokeWidth={2.5}
            vectorEffect="non-scaling-stroke"
          />
        </svg>
      </div>
      <span
        className={cn(
          "exhibit-title min-w-0 shrink text-left",
          flowCtaLabelTypographyClassName,
        )}
      >
        {t(label.en, label.de)}
      </span>
    </div>
  );
}
