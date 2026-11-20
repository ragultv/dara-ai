import React, { useState } from "react";
import { Button } from "../lib/ui/Button.tsx";
import { IconButton } from "../lib/ui/IconButton.tsx";
import { Input } from "../lib/ui/Input.tsx";
import { Textarea } from "../lib/ui/Textarea.tsx";
import { Checkbox } from "../lib/ui/Checkbox.tsx";
import { Toggle } from "../lib/ui/Toggle.tsx";
import { Slider } from "../lib/ui/Slider.tsx";
import { Select } from "../lib/ui/Select.tsx";
import { SegmentedControl } from "../lib/ui/SegmentedControl.tsx";
import { Card } from "../lib/ui/Card.tsx";
import { Badge } from "../lib/ui/Badge.tsx";
import { Tooltip } from "../lib/ui/Tooltip.tsx";
import { Dialog } from "../lib/ui/Dialog.tsx";
import { ProgressBar } from "../lib/ui/ProgressBar.tsx";
import { Skeleton } from "../lib/ui/Skeleton.tsx";
import { Tabs } from "../lib/ui/Tabs.tsx";
import { EmptyState } from "../lib/ui/EmptyState.tsx";
import { ErrorState } from "../lib/ui/ErrorState.tsx";
import { Markdown } from "../lib/ui/Markdown.tsx";
import { Reasoning } from "../lib/components/assistant-ui/Reasoning.tsx";
import { toast } from "../lib/ui/Toast.tsx";

export const KitPage: React.FC = () => {
  const [inputValue, setInputValue] = useState("");
  const [toggleChecked, setToggleChecked] = useState(true);
  const [sliderVal, setSliderVal] = useState(70);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [activeTab, setActiveTab] = useState("tab1");

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-10 text-left overflow-y-auto custom-scrollbar select-text">
      <header className="border-b border-border pb-5">
        <h1 className="text-xl font-bold text-text">Dara UI Design Kit</h1>
        <p className="text-xs text-text-muted mt-0.5">Component suite & atomic primitives</p>
      </header>

      {/* Buttons */}
      <section className="space-y-4">
        <h2 className="text-xs font-bold uppercase tracking-wider text-text-subtle">Buttons</h2>
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="primary">Primary</Button>
          <Button variant="secondary">Secondary</Button>
          <Button variant="ghost">Ghost</Button>
          <Button variant="danger">Danger</Button>
          <IconButton icon="copy" label="Copy" />
          <Button onClick={() => toast.show("Notification triggered!", "info")}>
            Trigger Toast
          </Button>
        </div>
      </section>

      {/* Badges */}
      <section className="space-y-4">
        <h2 className="text-xs font-bold uppercase tracking-wider text-text-subtle">Badges</h2>
        <div className="flex flex-wrap gap-2">
          <Badge variant="primary">Primary</Badge>
          <Badge variant="success">Success</Badge>
          <Badge variant="warning">Warning</Badge>
          <Badge variant="danger">Danger</Badge>
          <Badge variant="neutral">Neutral</Badge>
        </div>
      </section>

      {/* Inputs */}
      <section className="space-y-4">
        <h2 className="text-xs font-bold uppercase tracking-wider text-text-subtle">
          Inputs & Controls
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Input
            label="Sample Input"
            placeholder="Type here..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
          />
          <Select
            label="Sample Select"
            options={[
              { value: "opt1", label: "Option 1" },
              { value: "opt2", label: "Option 2" }
            ]}
          />
        </div>
        <Slider label="Slider" min={0} max={100} value={sliderVal} oninput={setSliderVal} />
        <Toggle label="Toggle Switch" checked={toggleChecked} onchange={setToggleChecked} />
      </section>

      {/* Reasoning (the model's thinking) */}
      <section className="space-y-4">
        <h2 className="text-xs font-bold uppercase tracking-wider text-text-subtle">Reasoning</h2>
        <Reasoning
          streaming
          text={
            "Thinking Process:\n\n1.  **Analyze the Request:** The user wants the product of 17 and 23.\n2.  **Calculate:** 17 x 20 = 340, 17 x 3 = 51, so 340 + 51 = 391.\n3.  **Check the result:** 23 x 17 = 230 + 161 = 391. Both agree."
          }
        />
        <Reasoning
          streaming={false}
          durationMs={12400}
          text={
            "Thinking Process:\n\n1.  **Analyze the Request:** The user wants the product of 17 and 23.\n2.  **Calculate:** 17 x 20 = 340, 17 x 3 = 51, so 340 + 51 = 391."
          }
        />
        <Reasoning
          streaming={false}
          durationMs={61000}
          cut
          text={
            "Let me plan the trip day by day. Day 1: arrive in Tokyo, settle in, evening walk in Shinjuku..."
          }
        />
      </section>

      {/* Markdown Showcase */}
      <section className="space-y-4">
        <h2 className="text-xs font-bold uppercase tracking-wider text-text-subtle">
          Markdown & Code Blocks
        </h2>
        <div className="font-[family-name:var(--font-reply)]">
          <Markdown
            content={`I've made a full royal purple theme with both light and dark modes. Switch between them by setting \`data-theme="dark"\` on \`<html>\`.\n\n### Core colors\n\n| Role | Color | Hex |\n| --- | --- | --- |\n| Primary | Royal violet | \`#7C3AED\` (hover \`#6D28D9\`, pressed \`#5B21B6\`) |\n| Deep royal | Dark violet for headings and gradients | \`#2E1065\` |\n| Secondary | Soft orchid violet, for highlights, active states and glows | \`#C084FC\` |\n| Accent (optional) | Gold, for badges and "pro" or "new" tags | \`#E8C468\` |\n\n**Dark mode** uses a near-black with a purple tint, which looks more premium than flat black:\n\n- Background \`#09050F\`, sidebar \`#0D0818\`, cards \`#120B1E\`, input \`#150E24\`\n- Borders \`#241A38\`, text \`#F4EFFB\`, muted text \`#A99CC0\`\n\n**Light mode** uses a soft lavender-white:\n\n- Background \`#FBF9FF\`, sidebar \`#F3EEFC\`, cards \`#FFFFFF\`\n- Borders \`#E4DBF2\`, text \`#1A1128\`\n\nHere is a code block:\n\`\`\`python\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    return merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))\n\`\`\``}
          />
        </div>
      </section>
    </div>
  );
};
