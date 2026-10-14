/// <reference lib="dom" />
/// <reference lib="dom.iterable" />
/** Browser interactions for the Wallace introduction page. */

export const FEATURES = {
  creative: {
    title: "Creative projects",
    description: "Brainstorm ideas, shape a concept, outline a story, or turn a blank page into a practical starting point.",
  },
  practical: {
    title: "Practical tasks",
    description: "Break down everyday work, compare approaches, and turn a complex to-do into manageable next steps.",
  },
  productivity: {
    title: "Productivity workflows",
    description: "Plan repeatable workflows, organize priorities, and use tools to keep work moving with less friction.",
  },
  development: {
    title: "Software development",
    description: "Explore a codebase, plan a change, implement it, and check the result with appropriate tests.",
  },
} as const;

export type ThemeChoice = "light" | "dark";

export function selectFeature(
  featureId: string,
  cards: readonly HTMLButtonElement[],
  title: HTMLElement,
  description: HTMLElement,
): boolean {
  if (!Object.hasOwn(FEATURES, featureId)) return false;
  const selected = FEATURES[featureId as keyof typeof FEATURES];
  for (const card of cards) {
    const active = card.dataset.feature === featureId;
    card.classList.toggle("is-active", active);
    card.setAttribute("aria-pressed", String(active));
  }
  title.textContent = selected.title;
  description.textContent = selected.description;
  return true;
}

export function chooseInitialTheme(stored: string | null, systemDark: boolean): ThemeChoice {
  if (stored === "light" || stored === "dark") return stored;
  if (stored === "true") return "dark";
  if (stored === "false") return "light";
  return systemDark ? "dark" : "light";
}

export function applyTheme(root: HTMLElement, toggle: HTMLButtonElement, theme: ThemeChoice): void {
  root.setAttribute("data-theme", theme);
  toggle.setAttribute("aria-pressed", String(theme === "dark"));
  toggle.textContent = theme === "dark" ? "Light mode" : "Dark mode";
}

export function initializePage(root: Document = document): void {
  const cards = [...root.querySelectorAll<HTMLButtonElement>("[data-feature]")];
  const title = root.getElementById("detail-title");
  const description = root.getElementById("detail-description");
  if (title && description && cards.length > 0) {
    for (const card of cards) {
      card.addEventListener("click", () => {
        const featureId = card.dataset.feature;
        if (featureId) selectFeature(featureId, cards, title, description);
      });
    }
    selectFeature("creative", cards, title, description);
  }

  const toggle = root.getElementById("theme-toggle") as HTMLButtonElement | null;
  if (!toggle) return;
  const storageKey = "wallace-theme-preference";
  let stored: string | null = null;
  try {
    stored = window.localStorage.getItem(storageKey);
  } catch {
    // Storage can be disabled; the toggle still works for this page view.
  }
  const systemDark = typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-color-scheme: dark)").matches;
  let theme = chooseInitialTheme(stored, systemDark);
  applyTheme(root.documentElement, toggle, theme);
  toggle.addEventListener("click", () => {
    theme = theme === "dark" ? "light" : "dark";
    applyTheme(root.documentElement, toggle, theme);
    try {
      window.localStorage.setItem(storageKey, theme);
    } catch {
      // A blocked storage API should not break the visible theme control.
    }
  });
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initializePage(), { once: true });
  } else {
    initializePage();
  }
}
