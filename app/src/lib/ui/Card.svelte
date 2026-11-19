<script lang="ts">
  import type { Snippet } from "svelte";
  import type { HTMLAttributes } from "svelte/elements";

  interface Props extends HTMLAttributes<HTMLDivElement> {
    variant?: "surface" | "raised" | "sunken";
    padding?: "none" | "sm" | "md" | "lg";
    children?: Snippet;
  }

  const {
    variant = "surface",
    padding = "md",
    class: className = "",
    children,
    ...rest
  }: Props = $props();

  const variantClasses = {
    surface: "bg-surface border-border",
    raised: "bg-surface-raised border-border",
    sunken: "bg-surface-sunken border-border"
  };

  const paddingClasses = {
    none: "p-0",
    sm: "p-3",
    md: "p-5",
    lg: "p-6"
  };
</script>

<div
  class="rounded-lg border {variantClasses[variant]} {paddingClasses[padding]} {className}"
  {...rest}
>
  {#if children}
    {@render children()}
  {/if}
</div>
