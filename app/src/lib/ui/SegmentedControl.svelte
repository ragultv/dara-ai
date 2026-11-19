<script lang="ts">
  export interface SegmentOption {
    value: string;
    label: string;
    description?: string;
    disabled?: boolean;
  }

  interface Props {
    id?: string;
    label?: string;
    options: SegmentOption[];
    value?: string;
    disabled?: boolean;
    class?: string;
    onchange?: (value: string) => void;
  }

  let {
    id,
    label,
    options,
    value = $bindable(options[0]?.value ?? ""),
    disabled = false,
    class: className = "",
    onchange
  }: Props = $props();

  const generatedId = `segment-${Math.random().toString(36).substring(2, 9)}`;
  const controlId = $derived(id ?? generatedId);

  function selectOption(val: string) {
    if (disabled) return;
    value = val;
    onchange?.(val);
  }

  function handleKeyDown(e: KeyboardEvent, index: number) {
    if (disabled) return;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") {
      e.preventDefault();
      const nextIndex = (index + 1) % options.length;
      const nextOpt = options[nextIndex];
      if (nextOpt && !nextOpt.disabled) {
        selectOption(nextOpt.value);
      }
    } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
      e.preventDefault();
      const prevIndex = (index - 1 + options.length) % options.length;
      const prevOpt = options[prevIndex];
      if (prevOpt && !prevOpt.disabled) {
        selectOption(prevOpt.value);
      }
    }
  }
</script>

<div class="flex flex-col space-y-1.5 text-left {className}">
  {#if label}
    <span id="{controlId}-label" class="text-xs font-medium text-text">
      {label}
    </span>
  {/if}

  <div
    id={controlId}
    role="radiogroup"
    aria-labelledby={label ? `${controlId}-label` : undefined}
    class="inline-flex p-1 rounded-md bg-surface-sunken border border-border"
  >
    {#each options as opt, i (opt.value)}
      {@const isSelected = value === opt.value}
      {@const isDisabled = disabled || (opt.disabled ?? false)}
      <button
        type="button"
        role="radio"
        aria-checked={isSelected}
        aria-label={opt.label}
        disabled={isDisabled}
        tabindex={isSelected ? 0 : -1}
        onclick={() => selectOption(opt.value)}
        onkeydown={(e) => handleKeyDown(e, i)}
        class="flex-1 py-1.5 px-3 rounded text-xs font-medium transition-colors cursor-pointer select-none focus-visible:outline-2 focus-visible:outline-focus focus-visible:outline-offset-1 disabled:opacity-50 disabled:cursor-not-allowed {isSelected
          ? 'bg-surface text-text shadow-sm border border-border-strong font-semibold'
          : 'text-text-muted hover:text-text hover:bg-surface-raised'}"
      >
        <span>{opt.label}</span>
        {#if opt.description}
          <span class="block text-[10px] text-text-subtle font-normal mt-0.5"
            >{opt.description}</span
          >
        {/if}
      </button>
    {/each}
  </div>
</div>
