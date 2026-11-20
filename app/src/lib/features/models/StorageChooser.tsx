import React, { useEffect, useState } from "react";
import { useEngineStore } from "../../stores/useEngineStore.ts";
import { useModelsStore } from "../../stores/useModelsStore.ts";
import { Badge } from "../../ui/Badge.tsx";
import { Icon } from "../../ui/Icon.tsx";
import { formatGb } from "../../utils/format.ts";

interface DriveInfo {
  root: string;
  free_gb: number;
  total_gb: number;
}

/** Folder Dara creates on the chosen drive. */
function folderOn(root: string): string {
  return `${root}Dara\models`;
}

function driveOf(path: string): string {
  return path.slice(0, 3).toUpperCase();
}

/**
 * First-run step: models are several GB each, so let the user put them on the drive with room.
 * Only shown inside the desktop app (it needs the engine's folder setting and the drive list).
 */
export const StorageChooser: React.FC = () => {
  const isAvailable = useEngineStore((s) => s.isAvailable);
  const info = useEngineStore((s) => s.info);
  const isRestarting = useEngineStore((s) => s.isRestarting);
  const load = useEngineStore((s) => s.load);
  const save = useEngineStore((s) => s.save);
  const restart = useEngineStore((s) => s.restart);
  const fetchModels = useModelsStore((s) => s.fetchModels);

  const [drives, setDrives] = useState<DriveInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isAvailable) return;
    void load();
    void import("@tauri-apps/api/core")
      .then(({ invoke }) => invoke<DriveInfo[]>("list_drives"))
      .then(setDrives)
      .catch(() => setDrives([]));
  }, [isAvailable, load]);

  if (!isAvailable || !info || drives.length === 0) return null;

  const current = info.running.models_dir;
  const currentDrive = driveOf(current);
  const roomiest = drives.reduce((a, b) => (b.free_gb > a.free_gb ? b : a));

  const choose = async (drive: DriveInfo): Promise<void> => {
    setError(null);
    const dir = drive.root === driveOf(info.default_models_dir) ? "" : folderOn(drive.root);
    const ok = await save({ ...info.saved, models_dir: dir });
    if (!ok) {
      setError("That folder could not be used. Pick another drive.");
      return;
    }
    if (await restart(false)) await fetchModels();
  };

  return (
    <section
      aria-labelledby="storage-title"
      className="w-full max-w-3xl rounded-md border border-border bg-surface p-5 space-y-3"
    >
      <div className="flex items-center gap-2">
        <Icon name="hard-drive" size={16} className="text-text" />
        <h2 id="storage-title" className="text-sm font-bold text-text">
          Where should Dara keep its models?
        </h2>
      </div>
      <p className="text-xs text-text-muted leading-relaxed">
        Each model takes several GB. Choose the drive with the most free space.
      </p>
      <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {drives.map((d) => {
          const isCurrent = d.root === currentDrive;
          return (
            <li key={d.root}>
              <button
                type="button"
                disabled={isRestarting || isCurrent}
                onClick={() => void choose(d)}
                aria-pressed={isCurrent}
                className={`w-full text-left rounded-md border p-3 flex items-center justify-between gap-2 cursor-pointer disabled:cursor-default ${
                  isCurrent ? "border-border-strong bg-surface-raised" : "border-border hover:bg-surface-raised"
                }`}
              >
                <span>
                  <span className="block text-sm font-semibold text-text">{d.root}</span>
                  <span className="block text-xs text-text-muted tabular-nums">
                    {formatGb(d.free_gb)} free of {formatGb(d.total_gb)}
                  </span>
                </span>
                <span className="flex flex-col items-end gap-1">
                  {d.root === roomiest.root && (
                    <Badge variant="success" size="sm">
                      Most space
                    </Badge>
                  )}
                  {isCurrent && (
                    <Badge variant="primary" size="sm">
                      In use
                    </Badge>
                  )}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
      <p className="text-xs text-text-muted truncate" title={current}>
        Current folder: <span className="font-mono">{current}</span>
      </p>
      {isRestarting && (
        <p role="status" className="text-xs text-text-muted">
          Restarting the engine with the new folder...
        </p>
      )}
      {error && (
        <p role="alert" className="text-xs font-medium text-danger">
          {error}
        </p>
      )}
    </section>
  );
};
