/// <reference lib="dom" />
import { applyTheme, chooseInitialTheme, initializePage, selectFeature } from "./app.ts";

Deno.test("saved light or dark choice takes priority over the system theme", () => {
  if (chooseInitialTheme("light", true) !== "light") throw new Error("saved light choice was ignored");
  if (chooseInitialTheme("dark", false) !== "dark") throw new Error("saved dark choice was ignored");
  if (chooseInitialTheme("false", true) !== "light") throw new Error("legacy light choice was ignored");
  if (chooseInitialTheme("true", false) !== "dark") throw new Error("legacy dark choice was ignored");
  if (chooseInitialTheme(null, true) !== "dark") throw new Error("system dark choice was ignored");
  if (chooseInitialTheme(null, false) !== "light") throw new Error("light default was ignored");
});

Deno.test("theme button communicates its current state", () => {
  const rootAttributes = new Map<string, string>();
  const buttonAttributes = new Map<string, string>();
  const root = { setAttribute(name: string, value: string) { rootAttributes.set(name, value); } } as HTMLElement;
  const toggle = {
    textContent: "",
    setAttribute(name: string, value: string) { buttonAttributes.set(name, value); },
  } as HTMLButtonElement;
  applyTheme(root, toggle, "dark");
  if (rootAttributes.get("data-theme") !== "dark" || buttonAttributes.get("aria-pressed") !== "true" || String(toggle.textContent) !== "Light mode") {
    throw new Error("dark state is not reflected in the page and button");
  }
  applyTheme(root, toggle, "light");
  if (rootAttributes.get("data-theme") !== "light" || buttonAttributes.get("aria-pressed") !== "false" || String(toggle.textContent) !== "Dark mode") {
    throw new Error("light state is not reflected in the page and button");
  }
});

function card(feature: string) {
  const classes = new Set<string>();
  const attributes = new Map<string, string>();
  let onClick: (() => void) | undefined;
  const element = {
    dataset: { feature },
    classList: { toggle(name: string, active: boolean) {
      if (active) classes.add(name);
      else classes.delete(name);
    } },
    setAttribute(name: string, value: string) { attributes.set(name, value); },
    addEventListener(name: string, listener: () => void) {
      if (name === "click") onClick = listener;
    },
  } as unknown as HTMLButtonElement;
  return { element, classes, attributes, click: () => onClick?.() };
}

Deno.test("selecting a feature updates the detail and pressed state", () => {
  const creative = card("creative");
  const development = card("development");
  const title = { textContent: "" } as HTMLElement;
  const description = { textContent: "" } as HTMLElement;
  const cards = [creative.element, development.element];

  if (!selectFeature("development", cards, title, description)) throw new Error("valid feature was rejected");
  if (title.textContent !== "Software development") throw new Error("detail title was not updated");
  if (!description.textContent?.includes("codebase")) throw new Error("detail description was not updated");
  if (!development.classes.has("is-active") || creative.classes.has("is-active")) throw new Error("active card was not updated");
  if (development.attributes.get("aria-pressed") !== "true" || creative.attributes.get("aria-pressed") !== "false") throw new Error("pressed state was not updated");

  if (selectFeature("unknown", cards, title, description)) throw new Error("unknown feature was accepted");
  if (title.textContent !== "Software development") throw new Error("unknown feature changed the detail");
});

Deno.test("the page wires card clicks to the visible detail", () => {
  const creative = card("creative");
  const practical = card("practical");
  const title = { textContent: "" } as HTMLElement;
  const description = { textContent: "" } as HTMLElement;
  const root = {
    querySelectorAll: () => [creative.element, practical.element],
    getElementById: (id: string) => id === "detail-title" ? title : id === "detail-description" ? description : null,
  } as unknown as Document;

  initializePage(root);
  if (String(title.textContent) !== "Creative projects") throw new Error("initial detail is missing");
  practical.click();
  if (String(title.textContent) !== "Practical tasks") throw new Error("click did not switch the detail");
  if (practical.attributes.get("aria-pressed") !== "true") throw new Error("clicked card was not pressed");
});

Deno.test("the emitted browser script wires card clicks", async () => {
  const creative = card("creative");
  const development = card("development");
  const title = { textContent: "" } as HTMLElement;
  const description = { textContent: "" } as HTMLElement;
  const root = {
    readyState: "complete",
    querySelectorAll: () => [creative.element, development.element],
    getElementById: (id: string) => id === "detail-title" ? title : id === "detail-description" ? description : null,
  } as unknown as Document;
  const original = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", { configurable: true, value: root });
  try {
    const emitted = await import("./app.js");
    if (emitted.chooseInitialTheme("light", true) !== "light") {
      throw new Error("emitted script is missing the saved-theme behavior");
    }
    development.click();
    if (String(title.textContent) !== "Software development") throw new Error("emitted script did not wire the click");
  } finally {
    if (original) Object.defineProperty(globalThis, "document", original);
    else Reflect.deleteProperty(globalThis, "document");
  }
});
