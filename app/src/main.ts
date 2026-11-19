import { mount } from "svelte";
import "./app.css";
import App from "./App.svelte";

const target = document.getElementById("app");
if (!target) {
  throw new Error("Target #app element not found");
}

// Prevent default drag and drop behavior across window
if (typeof window !== "undefined") {
  window.addEventListener("dragover", (e) => e.preventDefault(), false);
  window.addEventListener("drop", (e) => e.preventDefault(), false);
}

const app = mount(App, {
  target
});

export default app;
