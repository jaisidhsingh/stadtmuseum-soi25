export type LanguageMode = "bilingual" | "en" | "de";
export type LanguageOrder = "en-first" | "de-first";

const LANGUAGE_MODE_STORAGE_KEY = "soi25-language-mode";

const isLanguageMode = (value: string): value is LanguageMode =>
  value === "bilingual" || value === "en" || value === "de";

const getInitialLanguageMode = (): LanguageMode => {
  if (typeof window === "undefined") {
    return "en";
  }

  const storedMode = window.localStorage.getItem(LANGUAGE_MODE_STORAGE_KEY);
  if (storedMode && isLanguageMode(storedMode)) {
    return storedMode;
  }

  return "en";
};

export const languageSettings: {
  mode: LanguageMode;
  order: LanguageOrder;
  separator: string;
} = {
  // Defaults to English; use controls in the Start page to switch.
  mode: getInitialLanguageMode(),
  // Change order to "de-first" to show German before English.
  order: "en-first",
  separator: " / ",
};

export const getLanguageMode = (): LanguageMode => languageSettings.mode;

export const setLanguageMode = (mode: LanguageMode): void => {
  languageSettings.mode = mode;

  if (typeof window !== "undefined") {
    window.localStorage.setItem(LANGUAGE_MODE_STORAGE_KEY, mode);
  }
};

export const t = (en: string, de: string): string => {
  if (languageSettings.mode === "en") {
    return en;
  }

  if (languageSettings.mode === "de") {
    return de;
  }

  if (languageSettings.order === "de-first") {
    return `${de}${languageSettings.separator}${en}`;
  }

  return `${en}${languageSettings.separator}${de}`;
};
