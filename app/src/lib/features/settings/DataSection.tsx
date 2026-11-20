import React, { useState } from "react";
import { storeApi } from "../../api/store-api.ts";
import { useChatStore } from "../../stores/useChatStore.ts";
import { Button } from "../../ui/Button.tsx";
import { Dialog } from "../../ui/Dialog.tsx";
import { toast } from "../../ui/Toast.tsx";
import { SettingsSection } from "./SettingsSection.tsx";
import type { ServerFacts } from "./useServerFacts.ts";

interface DataSectionProps {
  facts: ServerFacts;
}

/** Saves a JSON document through a temporary link; no file-system access is needed. */
function downloadJson(data: unknown, fileName: string): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const DataSection: React.FC<DataSectionProps> = ({ facts }) => {
  const [isExporting, setIsExporting] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const store = facts.store;

  const exportAll = async (): Promise<void> => {
    setIsExporting(true);
    try {
      const data = await storeApi.exportAll();
      const stamp = new Date().toISOString().slice(0, 10);
      downloadJson(data, `dara-export-${stamp}.json`);
      toast.show("Export saved to your downloads.", "success");
    } catch (err: unknown) {
      toast.show(`Export failed: ${err instanceof Error ? err.message : "unknown error"}`, "error");
    } finally {
      setIsExporting(false);
    }
  };

  const deleteAll = async (): Promise<void> => {
    setIsDeleting(true);
    try {
      await storeApi.deleteAllData();
      await useChatStore.getState().loadConversations();
      setIsConfirmOpen(false);
      facts.refresh();
      toast.show("All conversations, images and saved model setups were deleted.", "success");
    } catch (err: unknown) {
      toast.show(
        `Could not delete: ${err instanceof Error ? err.message : "unknown error"}`,
        "error"
      );
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <SettingsSection
      id="settings-data"
      icon="database"
      title="Data"
      description="Your chats stay on this computer, in a database that the engine manages."
    >
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-text">Conversations</h3>
          <p className="mt-0.5 text-xs text-text-muted" role="status">
            {store
              ? `${store.conversations} saved conversation${store.conversations === 1 ? "" : "s"}, ${store.messages} message${store.messages === 1 ? "" : "s"}.`
              : (facts.storeError ?? "Counting...")}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            disabled={!store || isExporting}
            onClick={() => void exportAll()}
          >
            {isExporting ? "Exporting..." : "Export all"}
          </Button>
          <Button
            variant="danger"
            size="sm"
            disabled={!store}
            onClick={() => setIsConfirmOpen(true)}
          >
            Delete all data
          </Button>
        </div>
      </div>
      <p className="text-[11px] text-text-muted">
        Export saves one JSON file with your settings, saved model setups and every conversation.
      </p>

      <Dialog
        isOpen={isConfirmOpen}
        title="Delete all data?"
        description="This cannot be undone. Export first if you want a copy."
        onclose={() => setIsConfirmOpen(false)}
        actions={
          <>
            <Button variant="ghost" size="sm" onClick={() => setIsConfirmOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              size="sm"
              disabled={isDeleting}
              onClick={() => void deleteAll()}
            >
              {isDeleting ? "Deleting..." : "Delete everything"}
            </Button>
          </>
        }
      >
        <p>You will lose:</p>
        <ul className="list-disc space-y-1 pl-5">
          <li>
            all {store ? store.conversations : ""} conversations and their{" "}
            {store ? store.messages : ""} messages
          </li>
          <li>every image you attached to a message</li>
          <li>the K/V and Flash setup saved for each model</li>
        </ul>
        <p>Your settings on this page and your downloaded models are kept.</p>
      </Dialog>
    </SettingsSection>
  );
};
