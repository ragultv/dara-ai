<script lang="ts">
  interface Props {
    id?: string;
    label: string;
    min?: number;
    max?: number;
    step?: number;
    value?: number | undefined;
    unit?: string;
    disabled?: boolean;
    class?: string;
    oninput?: (value: number) => void;
  }

  let {
    id,
    label,
    min = 0,
    max = 100,
    step = 1,
    value = $bindable(50),
    unit = "",
    disabled = false,
    class: className = "",
    oninput
  }: Props = $props();

  const generatedId = `slider-${Math.random().toString(36).substring(2, 9)}`;
  const sliderId = $derived(id ?? generatedId);

  function handleInput(e: Event & { currentTarget: HTMLInputElement }) {
    const num = parseFloat(e.currentTarget.value);
    value = num;
    oninput?.(num);
  }
</script>

<div class="flex flex-col space-y-2 w-full text-left {className}">
  <div class="flex items-center justify-between">
    <label for={sliderId} class="text-xs font-medium text-text">
      {label}
    </label>
    <span class="text-xs font-mono tabular-nums text-text-muted">
      {value}{unit}
    </span>
  </div>

  <input
    id={sliderId}
    type="range"
    {min}
    {max}
    {step}
    {value}
    {disabled}
    aria-valuenow={value}
    aria-valuemin={min}
    aria-valuemax={max}
    aria-label={label}
    oninput={handleInput}
    class="w-full h-2 rounded-full appearance-none bg-surface-sunken cursor-pointer accent-primary focus-visible:outline-2 focus-visible:outline-focus focus-visible:outline-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
  />
</div>
