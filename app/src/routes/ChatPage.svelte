<script lang="ts">
  import { serverStore } from "../lib/stores/server.svelte.ts";
  import EmptyState from "../lib/ui/EmptyState.svelte";
  import Button from "../lib/ui/Button.svelte";
</script>

<div class="flex-1 flex flex-col h-full overflow-hidden text-left">
  <!-- Chat Main Area -->
  <div class="flex-1 overflow-y-auto p-6 flex flex-col items-center justify-center">
    <EmptyState
      title="How can Dara help today?"
      description="Private, local AI running directly on your Windows GPU. Your prompts never leave this PC."
    >
      {#snippet actions()}
        <div class="flex items-center space-x-2">
          <Button
            size="sm"
            variant="secondary"
            onclick={() => {
              /* TODO(M3): start sample chat */
            }}
          >
            Summarize a document
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onclick={() => {
              /* TODO(M3): start sample chat */
            }}
          >
            Draft Python code
          </Button>
        </div>
      {/snippet}
    </EmptyState>
  </div>

  <!-- Chat Composer Placeholder (M3) -->
  <div class="p-4 border-t border-border bg-surface shrink-0 max-w-3xl w-full mx-auto">
    <div
      class="flex items-center justify-between p-2 rounded-xl border border-border bg-surface-sunken"
    >
      <input
        type="text"
        placeholder="Type a message (Shift+Enter for new line)..."
        class="w-full bg-transparent px-3 py-1.5 text-sm text-text placeholder:text-text-subtle outline-none"
      />
      <Button size="sm" variant="primary" disabled={!serverStore.isReady}>Send</Button>
    </div>
  </div>
</div>
