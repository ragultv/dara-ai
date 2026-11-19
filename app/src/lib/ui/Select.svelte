<script lang="ts">
  import type { HTMLSelectAttributes } from "svelte/elements";

  export interface SelectOption {
    value: string;
    label: string;
    disabled?: boolean;
  }

  interface Props extends Omit<HTMLSelectAttributes, "value"> {
    id?: string;
    label?: string;
    hint?: string;
    error?: string;
    value?: string;
    options: SelectOption[];
    onchange?: (e: Event & { currentTarget: HTMLSelectElement }) => void;
  }

  let {
    id,
    label,
    hint,
    error,
    value = $bindable(""),
    options,
    disabled = false,
    class: className = "",
    onchange,
    ...rest
  }: Props = $props();

  const generatedId = `select-${Math.random().toString(36).substring(2, 9)}`;
  const selectId = $derived(id ?? generatedId);
  const hintId = $derived(hint ? `${selectId}-hint` : undefined);
  const errorId = $derived(error ? `${selectId}-error` : undefined);
  const describedBy = $derived([hintId, errorId].filter(Boolean).join(" ") || undefined);
</script>

<div class="flex flex-col space-y-1.5 w-full text-left">
  {#if label}
    <label for={selectId} class="text-xs font-medium text-text">
      {label}
    </label>
  {/if}

  <div class="relative">
    <select
      id={selectId}
      bind:value
      {disabled}
      aria-invalid={Boolean(error)}
      aria-describedby={describedBy}
      class="w-full h-10 pl-3 pr-10 py-2 text-sm rounded-md bg-surface border transition-colors text-text appearance-none disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-2 focus-visible:outline-focus focus-visible:outline-offset-2 cursor-pointer {error
        ? 'border-danger focus-visible:outline-danger'
        : 'border-border focus-visible:border-primary'} {className}"
      {onchange}
      {...rest}
    >
      {#each options as opt (opt.value)}
        <option value={opt.value} disabled={opt.disabled ?? false}>
          {opt.label}
        </option>
      {/each}
    </select>

    <!-- Chevron icon -->
    <div
      class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-text-muted"
    >
      <svg
        class="w-4 h-4"
        viewBox="0 0 20 20"
        fill="none"
        stroke="currentColor"
        stroke-width="1.5"
        aria-hidden="true"
      >
        <path d="M6 8l4 4 4-4" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
    </div>
  </div>

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
