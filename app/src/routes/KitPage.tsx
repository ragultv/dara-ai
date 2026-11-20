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

      {/* Markdown Showcase */}
      <section className="space-y-4">
        <h2 className="text-xs font-bold uppercase tracking-wider text-text-subtle">
          Markdown & Code Blocks
        </h2>
        <Markdown
          content={`Here is a code block:\n\`\`\`python\ndef merge_sort(arr):\n    # Base case: single element is sorted\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\`\`\``}
        />
      </section>
    </div>
  );
};
