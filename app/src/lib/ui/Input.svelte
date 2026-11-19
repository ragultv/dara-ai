<script lang="ts">
  import type { HTMLInputAttributes } from "svelte/elements";

  interface Props extends Omit<HTMLInputAttributes, "value"> {
    id?: string;
    label?: string;
    hint?: string;
    error?: string;
    value?: string | number;
    oninput?: (e: Event & { currentTarget: HTMLInputElement }) => void;
  }

  let {
    id,
    label,
    hint,
    error,
    value = $bindable(""),
    disabled = false,
    class: className = "",
    type = "text",
    oninput,
    ...rest
  }: Props = $props();

  const generatedId = `input-${Math.random().toString(36).substring(2, 9)}`;
  const inputId = $derived(id ?? generatedId);
  const hintId = $derived(hint ? `${inputId}-hint` : undefined);
  const errorId = $derived(error ? `${inputId}-error` : undefined);
  const describedBy = $derived([hintId, errorId].filter(Boolean).join(" ") || undefined);
</script>

<div class="flex flex-col space-y-1.5 w-full text-left">
  {#if label}
    <label for={inputId} class="text-xs font-medium text-text">
      {label}
    </label>
  {/if}

  <input
    id={inputId}
    {type}
    bind:value
    {disabled}
    aria-invalid={Boolean(error)}
    aria-describedby={describedBy}
    class="w-full h-10 px-3 py-2 text-sm rounded-md bg-surface border transition-colors text-text placeholder:text-text-subtle disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-2 focus-visible:outline-focus focus-visible:outline-offset-2 {error
      ? 'border-danger focus-visible:outline-danger'
      : 'border-border focus-visible:border-primary'} {className}"
    {oninput}
    {...rest}
  />

  {#if error}
    <p id={errorId} class="text-xs text-danger font-medium" role="alert">
      {error}
    </p>
  {:else if hint}
    <p id={hintId} class="text-xs text-text-muted">
      {hint}
    </p>
  {/if}
</div>
