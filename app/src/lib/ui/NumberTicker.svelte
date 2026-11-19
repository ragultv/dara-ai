<script lang="ts">
  interface Props {
    value: number;
    decimalPlaces?: number;
    prefix?: string;
    suffix?: string;
    class?: string;
  }

  const {
    value = 0,
    decimalPlaces = 0,
    prefix = "",
    suffix = "",
    class: className = ""
  }: Props = $props();

  const formatted = $derived.by(() => {
    if (decimalPlaces > 0) {
      return value.toFixed(decimalPlaces);
    }
    return Math.round(value).toLocaleString();
  });
</script>

<span class="inline-flex items-center font-mono tabular-nums leading-none {className}" aria-live="polite">
  {#if prefix}<span>{prefix}</span>{/if}<span>{formatted}</span>{#if suffix}<span>{suffix}</span>{/if}
</span>
