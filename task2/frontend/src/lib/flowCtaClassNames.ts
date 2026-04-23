/**
 * Shared CTA bar styles for flow pages (camera bar + EXIT in header on camera/silhouette).
 */

const cameraBarCtaClassName =
  "exhibit-title !h-14 !min-h-14 !w-full min-w-0 !shrink-0 !items-center !justify-center !gap-3 !rounded-2xl !px-2 !py-0 !text-xl !font-bold !uppercase !leading-tight !tracking-[0.18em] !text-white !box-border md:!h-16 md:!min-h-16 md:!px-3 md:!text-3xl [&_svg]:!h-6 [&_svg]:!w-6 md:[&_svg]:!h-7 md:[&_svg]:!w-7";

/** Bar geometry for dark text (step 4 = film yellow) — no `!text-white`, no `cta-start-outlined`. */
const cameraBarCtaClassNameForStep4 =
  "exhibit-title !h-14 !min-h-14 !w-full min-w-0 !shrink-0 !items-center !justify-center !gap-3 !rounded-2xl !px-2 !py-0 !text-xl !font-bold !uppercase !leading-tight !tracking-[0.18em] !box-border md:!h-16 md:!min-h-16 md:!px-3 md:!text-3xl [&_svg]:!h-6 [&_svg]:!w-6 md:[&_svg]:!h-7 md:[&_svg]:!w-7";

export const retakeButtonClassName = `cta-step-3 cta-start-outlined ${cameraBarCtaClassName}`;

export const continueButtonClassName = `cta-step-2 cta-start-outlined ${cameraBarCtaClassName}`;

export const captureButtonClassName = `cta-step-2 cta-start-outlined ${cameraBarCtaClassName}`;

export const flowExitButtonClassName = `${retakeButtonClassName} !w-auto !max-w-none shrink-0`;

/**
 * Last step / QR — same bar size as Go Back, `cta-step-4` (4th step card, film yellow).
 * `cta-qr-steady` keeps the same shadow when disabled (no shadow “pop” on enable); opacity +
 * grayscale still grey out the disabled state.
 */
export const continueToQrButtonClassName = `cta-step-4 cta-qr-steady ${cameraBarCtaClassNameForStep4}`;

/**
 * Same type scale and letter style as the EXIT CTA / camera bar, for dark text on
 * light panels (e.g. `FlowStepIndicator` next to EXIT).
 */
export const flowCtaLabelTypographyClassName =
  "text-xl font-bold uppercase leading-tight tracking-[0.18em] md:text-3xl";

/**
 * Blue film CTA — same `cameraBarCta` scale as EXIT / bottom bar; use for non-button previews.
 * Pair with a `div` + `inline-flex` (or `Button` when wired up).
 */
export const filmBlueCtaClassName = `cta-step-1 cta-start-outlined inline-flex ${cameraBarCtaClassName} w-full cursor-default select-none pointer-events-none`;
