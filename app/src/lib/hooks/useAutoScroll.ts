import { useEffect, useRef } from "react";

export function useAutoScroll<T extends HTMLElement>(dependencies: unknown[] = [], enabled = true) {
  const containerRef = useRef<T | null>(null);

  useEffect(() => {
    if (!enabled || !containerRef.current) return;
    const el = containerRef.current;
    const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distFromBottom < 300) {
      el.scrollTop = el.scrollHeight;
    }
  }, [...dependencies, enabled]);

  return containerRef;
}
