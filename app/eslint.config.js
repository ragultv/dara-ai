import js from "@eslint/js";
import ts from "typescript-eslint";
import svelte from "eslint-plugin-svelte";
import prettier from "eslint-config-prettier";
import globals from "globals";

export default ts.config(
  js.configs.recommended,
  ...ts.configs.strict,
  ...svelte.configs["flat/recommended"],
  prettier,
  ...svelte.configs["flat/prettier"],
  {
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node
      }
    }
  },
  {
    files: ["**/*.svelte"],
    languageOptions: {
      parserOptions: {
        parser: ts.parser
      }
    },
    rules: {
      // Svelte a11y rules as errors per docs/ui/06-coding-standards.md
      "svelte/valid-compile": "error"
    }
  },
  {
    ignores: [
      "build/",
      ".svelte-kit/",
      "dist/",
      "src-tauri/",
      "src/tokens.css",
      "node_modules/",
      "test-results/",
      "playwright-report/"
    ]
  }
);
