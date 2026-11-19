<script lang="ts">
  interface Props {
    id?: string;
    label: string;
    description?: string;
    checked?: boolean | undefined;
    disabled?: boolean;
    class?: string;
    onchange?: (checked: boolean) => void;
  }

  let {
    id,
    label,
    description,
    checked = $bindable(false),
    disabled = false,
    class: className = "",
    onchange
  }: Props = $props();

  const generatedId = `toggle-${Math.random().toString(36).substring(2, 9)}`;
  const toggleId = $derived(id ?? generatedId);
  const descId = $derived(description ? `${toggleId}-desc` : undefined);

  function handleClick() {
    if (disabled) return;
    checked = !checked;
    onchange?.(checked);
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (disabled) return;
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      handleClick();
    }
  }
</script>

<div class="flex items-center justify-between space-x-4 text-left {className}">
  <div class="flex flex-col">
    <span
      id="{toggleId}-label"
      class="text-sm font-medium text-text select-none {disabled ? 'opacity-50' : ''}"
    >
      {label}
    </span>
    {#if description}
      <span id={descId} class="text-xs text-text-muted">
        {description}
      </span>
    {/if}
  </div>

  <button
    id={toggleId}
    type="button"
    role="switch"
    aria-checked={checked}
    aria-labelledby="{toggleId}-label"
    aria-describedby={descId}
    {disabled}
    onclick={handleClick}
    onkeydown={handleKeyDown}
    class="relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-base focus-visible:outline-2 focus-visible:outline-focus focus-visible:outline-offset-2 disabled:opacity-50 disabled:cursor-not-allowed {checked
      ? 'bg-primary'
      : 'bg-surface-sunken border-border'}"
  >
    <span
      class="pointer-events-none inline-block h-5 w-5 transform rounded-full bg-surface shadow-sm transition duration-base {checked
        ? 'translate-x-5 bg-on-primary'
        : 'translate-x-0'}"
    ></span>
  </button>
</div>
