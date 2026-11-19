<script lang="ts">
  import type { HTMLTextareaAttributes } from "svelte/elements";

  interface Props extends Omit<HTMLTextareaAttributes, "value"> {
    id?: string;
    label?: string;
    hint?: string;
    error?: string;
    value?: string;
    maxRows?: number;
    oninput?: (e: Event & { currentTarget: HTMLTextAreaElement }) => void;
  }

  let {
    id,
    label,
    hint,
    error,
    value = $bindable(""),
    disabled = false,
    rows = 3,
    maxRows = 10,
    class: className = "",
    oninput,
    ...rest
  }: Props = $props();

  let textareaElement: HTMLTextAreaElement | null = $state(null);

  const generatedId = `textarea-${Math.random().toString(36).substring(2, 9)}`;
  const textareaId = $derived(id ?? generatedId);
  const hintId = $derived(hint ? `${textareaId}-hint` : undefined);
  const errorId = $derived(error ? `${textareaId}-error` : undefined);
  const describedBy = $derived([hintId, errorId].filter(Boolean).join(" ") || undefined);

  function adjustHeight(el: HTMLTextAreaElement) {
    el.style.height = "auto";
    const lineHeight = 20; // approximate px per row
    const maxHeight = maxRows * lineHeight + 16;
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
  }

  function handleInput(e: Event & { currentTarget: HTMLTextAreaElement }) {
    adjustHeight(e.currentTarget);
    oninput?.(e);
  }

  $effect(() => {
    if (textareaElement && value !== undefined) {
      adjustHeight(textareaElement);
    }
  });
</script>

<div class="flex flex-col space-y-1.5 w-full text-left">
  {#if label}
    <label for={textareaId} class="text-xs font-medium text-text">
      {label}
    </label>
  {/if}

  <textarea
    id={textareaId}
    bind:this={textareaElement}
    bind:value
    {rows}
    {disabled}
    aria-invalid={Boolean(error)}
    aria-describedby={describedBy}
    class="w-full px-3 py-2 text-sm rounded-md bg-surface border transition-colors text-text placeholder:text-text-subtle resize-none disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-2 focus-visible:outline-focus focus-visible:outline-offset-2 {error
      ? 'border-danger focus-visible:outline-danger'
      : 'border-border focus-visible:border-primary'} {className}"
    oninput={handleInput}
    {...rest}
  ></textarea>

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
