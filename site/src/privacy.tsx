import React from "react";
import ReactDOM from "react-dom/client";
import { Privacy } from "./components/Legal.tsx";
import "./index.css";

const rootElement = document.getElementById("root");
if (rootElement) {
  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      <Privacy />
    </React.StrictMode>
  );
}
