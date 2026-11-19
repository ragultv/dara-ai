<script lang="ts" module>
  export interface ToastItem {
    id: string;
    message: string;
    type?: "info" | "success" | "warning" | "danger";
    durationMs?: number;
  }

  class ToastStore {
    toasts = $state<ToastItem[]>([]);

    show(message: string, type: ToastItem["type"] = "info", durationMs = 3500) {
      const id = Math.random().toString(36).substring(2, 9);
      const item: ToastItem = { id, message, type, durationMs };
      this.toasts = [...this.toasts, item];

      if (durationMs > 0) {
        setTimeout(() => {
          this.dismiss(id);
        }, durationMs);
      }
    }

    dismiss(id: string) {
      this.toasts = this.toasts.filter((t) => t.id !== id);
    }
  }

  export const toast = new ToastStore();
</script>

<script lang="ts">
  const typeClasses = {
    info: "bg-surface-raised border-border text-text",
    success: "bg-surface-raised border-success text-success",
    warning: "bg-surface-raised border-warning text-warning",
    danger: "bg-surface-raised border-danger text-danger"
  };
</script>

<div
  role="region"
  aria-label="Notifications"
  class="fixed bottom-4 right-4 z-toast flex flex-col space-y-2 pointer-events-none max-w-sm w-full"
>
  {#each toast.toasts as t (t.id)}
    <div
      role="status"
      aria-live="polite"
      class="pointer-events-auto p-3 rounded-lg border shadow-md flex items-center justify-between text-xs font-medium space-x-3 animate-fade-in {typeClasses[
        t.type ?? 'info'
      ]}"
    >
      <span class="truncate">{t.message}</span>
      <button
        type="button"
        aria-label="Dismiss toast"
        onclick={() => toast.dismiss(t.id)}
        class="p-0.5 rounded text-text-subtle hover:text-text cursor-pointer focus-visible:outline-2 focus-visible:outline-focus"
      >
        <svg
          class="w-3.5 h-3.5"
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          stroke-width="1.5"
          aria-hidden="true"
        >
          <path d="M4 4l8 8M12 4L4 12" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </button>
    </div>
  {/each}
</div>
