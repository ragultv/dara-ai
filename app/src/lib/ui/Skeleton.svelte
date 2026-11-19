<script lang="ts">
  interface Props {
    width?: string;
    height?: string;
    rounded?: "sm" | "md" | "lg" | "full";
    class?: string;
  }

  const {
    width = "w-full",
    height = "h-4",
    rounded = "md",
    class: className = ""
  }: Props = $props();

  const roundedClasses = {
    sm: "rounded-sm",
    md: "rounded-md",
    lg: "rounded-lg",
    full: "rounded-full"
  };
</script>

<div
  aria-hidden="true"
  class="{width} {height} {roundedClasses[
    rounded
  ]} bg-surface-sunken border border-border animate-pulse {className}"
></div>
