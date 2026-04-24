/**
 * Copy + tints for the 4 exhibit steps (StartPage strip + per-step flow pages).
 * Accent strings are opaque color-mix on white; see index.css / StartPage.
 */
export const EXHIBIT_STEPS = [
  {
    titleEn: "Take your photo",
    titleDe: "Foto aufnehmen",
    descriptionEn: "Capture your silhouette.",
    descriptionDe: "Nehmen Sie Ihr Foto auf.",
  },
  {
    titleEn: "Style your silhouette",
    titleDe: "Silhouette gestalten",
    descriptionEn: "Select character-inspired body parts.",
    descriptionDe: "Mische Figuren-inspirierte Koerperteile.",
  },
  {
    titleEn: "Pick movie backgrounds",
    titleDe: "Hintergruende waehlen",
    descriptionEn: "Choose your favorite backgrounds.",
    descriptionDe: "Waehle deine Lieblingshintergruende.",
  },
  {
    titleEn: "Download with QR",
    titleDe: "Mit QR downloaden",
    descriptionEn: "Scan code and save your scenes.",
    descriptionDe: "Scanne den Code und speichere deine Galerie.",
  },
] as const;

export const EXHIBIT_STEP_ACCENTS = [
  "bg-[color-mix(in_srgb,hsl(var(--film-blue))_16%,white)]",
  "bg-[color-mix(in_srgb,hsl(var(--film-green))_16%,white)]",
  "bg-[color-mix(in_srgb,hsl(var(--film-red))_16%,white)]",
  "bg-[color-mix(in_srgb,hsl(var(--film-yellow))_20%,white)]",
] as const;

export type ExhibitStepIndex = 0 | 1 | 2 | 3;
