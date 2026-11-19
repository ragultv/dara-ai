<script lang="ts">
  import type { Snippet } from "svelte";
  import type { HTMLButtonAttributes } from "svelte/elements";

  interface Props extends Omit<HTMLButtonAttributes, "aria-label"> {
    "aria-label": string;
    variant?: "primary" | "secondary" | "ghost" | "danger";
    size?: "sm" | "md";
    isLoading?: boolean;
    children?: Snippet;
  }

  const {
    "aria-label": ariaLabel,
    variant = "ghost",
    size = "md",
    isLoading = false,
    disabled = false,
    class: className = "",
    type = "button",
    children,
    ...rest
  }: Props = $props();

  const variantClasses = {
    primary:
      "bg-primary text-on-primary hover:bg-primary-hover active:bg-primary-active border-transparent",
    secondary:
      "bg-surface text-text hover:bg-surface-sunken active:bg-surface-raised border border-border-strong",
    ghost:
      "bg-transparent text-text hover:bg-surface-sunken active:bg-surface-raised border-transparent",
    danger:
      "bg-danger text-on-primary hover:bg-danger-subtle hover:text-danger active:opacity-90 border-transparent"
  };

  const sizeClasses = {
    sm: "w-8 h-8 rounded-sm p-1.5",
    md: "w-10 h-10 rounded-md p-2"
  };
</script>

<button
  {type}
  aria-label={ariaLabel}
  disabled={disabled || isLoading}
  aria-busy={isLoading}
  class="inline-flex items-center justify-center transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed select-none focus-visible:outline-2 focus-visible:outline-focus focus-visible:outline-offset-2 {variantClasses[
    variant
  ]} {sizeClasses[size]} {className}"
  {...rest}
>
  {#if isLoading}
    <svg
      class="animate-spin h-4 w-4"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"
      ></circle>
      <path
        class="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      ></path>
    </svg>
  {:else if children}
    {@render children()}
  {/if}
</button>
