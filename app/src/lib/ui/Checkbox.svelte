<script lang="ts">
  import type { HTMLInputAttributes } from "svelte/elements";

  interface Props extends Omit<HTMLInputAttributes, "type" | "checked"> {
    id?: string;
    label: string;
    description?: string;
    checked?: boolean;
    indeterminate?: boolean;
    onchange?: (e: Event & { currentTarget: HTMLInputElement }) => void;
  }

  let {
    id,
    label,
    description,
    checked = $bindable(false),
    indeterminate = false,
    disabled = false,
    class: className = "",
    onchange,
    ...rest
  }: Props = $props();

  let inputRef: HTMLInputElement | null = $state(null);

  $effect(() => {
    if (inputRef) {
      inputRef.indeterminate = indeterminate;
    }
  });

  const generatedId = `checkbox-${Math.random().toString(36).substring(2, 9)}`;
  const checkboxId = $derived(id ?? generatedId);
  const descId = $derived(description ? `${checkboxId}-desc` : undefined);
</script>

<div class="flex items-start space-x-3 text-left {className}">
  <div class="flex items-center h-5">
    <input
      bind:this={inputRef}
      id={checkboxId}
      type="checkbox"
      bind:checked
      {disabled}
      aria-describedby={descId}
      class="w-4 h-4 rounded text-primary bg-surface border-border-strong focus-visible:outline-2 focus-visible:outline-focus focus-visible:outline-offset-2 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed accent-primary"
      {onchange}
      {...rest}
    />
  </div>

  <div class="flex flex-col text-sm">
    <label
      for={checkboxId}
      class="font-medium text-text select-none cursor-pointer {disabled
        ? 'opacity-50 cursor-not-allowed'
        : ''}"
    >
      {label}
    </label>
    {#if description}
      <p id={descId} class="text-xs text-text-muted">
        {description}
      </p>
    {/if}
  </div>
</div>
