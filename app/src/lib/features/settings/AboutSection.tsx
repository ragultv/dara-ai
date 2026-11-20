import React, { useEffect, useState } from "react";
import { getAppVersion } from "../../utils/app-version.ts";
import { Badge } from "../../ui/Badge.tsx";
import { KeyValue } from "../monitor/KeyValue.tsx";
import { SettingsSection } from "./SettingsSection.tsx";
import type { ServerFacts } from "./useServerFacts.ts";

interface AboutSectionProps {
  facts: ServerFacts;
}

export const AboutSection: React.FC<AboutSectionProps> = ({ facts }) => {
  const [appVersion, setAppVersion] = useState<string | undefined>(undefined);

  useEffect(() => {
    let active = true;
    void getAppVersion().then((v) => {
      if (active) setAppVersion(v);
    });
    return () => {
      active = false;
    };
  }, []);

  const { version } = facts;
  return (
    <SettingsSection id="settings-about" icon="info" title="About">
      <div className="flex items-center gap-2.5">
        <img src="/dara.png" alt="" className="h-6 w-6 shrink-0 object-contain" />
        <span className="text-sm font-bold text-text">Dara</span>
        {appVersion && (
          <Badge variant="primary" size="sm">
            v{appVersion}
          </Badge>
        )}
      </div>
      <KeyValue
        rows={[
          { label: "App version", value: appVersion },
          { label: "Engine version", value: version?.version },
          { label: "Engine API", value: version ? String(version.api) : undefined },
          { label: "Engine build", value: version?.built }
        ]}
      />
      <p className="text-xs leading-relaxed text-text-muted">
        Dara runs language models on your own graphics card. Nothing you write leaves this computer.
      </p>
    </SettingsSection>
  );
};
