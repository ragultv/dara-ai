<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    variant?: "neutral" | "purple" | "success" | "warning" | "danger";
    size?: "sm" | "md";
    icon?: Snippet;
    class?: string;
    children?: Snippet;
  }

  const {
    variant = "neutral",
    size = "sm",
    icon,
    class: className = "",
    children
  }: Props = $props();

  const variantClasses = {
    neutral: "bg-surface-sunken text-text border-border",
    purple: "bg-primary-subtle text-on-primary-subtle border-primary",
    success: "bg-success-subtle text-success border-success",
    warning: "bg-warning-subtle text-warning border-warning",
    danger: "bg-danger-subtle text-danger border-danger"
  };

  const sizeClasses = {
    sm: "px-2 py-0.5 text-xs gap-1",
    md: "px-2.5 py-1 text-sm gap-1.5"
  };
</script>

<span
  class="inline-flex items-center font-medium rounded-full border border-opacity-30 select-none {variantClasses[
    variant
  ]} {sizeClasses[size]} {className}"
>
  {#if icon}
    {@render icon()}
  {/if}
  {#if children}
    {@render children()}
  {/if}
</span>
