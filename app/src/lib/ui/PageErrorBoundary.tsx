import React from "react";
import { ErrorState } from "./ErrorState.tsx";

interface Props {
  /** what the page is called, for the message */
  page: string;
  children: React.ReactNode;
}
interface State {
  error: Error | null;
}

/**
 * A page that throws while it draws would otherwise leave the whole window blank. This catches it, says which page and why, and offers a
 * retry; the rest of the app (the side bar, the other pages) keeps working. The details can be copied to report the problem.
 */
export class PageErrorBoundary extends React.Component<Props, State> {
  override state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  override componentDidCatch(error: Error, info: React.ErrorInfo): void {
    // keep it in the console too: the stack is what a developer needs
    console.error(`The ${this.props.page} page crashed:`, error, info.componentStack);
  }

  private details(): string {
    const e = this.state.error;
    return [`Page: ${this.props.page}`, `Error: ${e?.message ?? ""}`, e?.stack ?? ""].join("\n");
  }

  override render(): React.ReactNode {
    if (!this.state.error) return this.props.children;
    return (
      <div className="mx-auto flex h-full max-w-xl flex-col items-center justify-center gap-3 px-6">
        <ErrorState
          title={`The ${this.props.page} page ran into a problem`}
          message={
            this.state.error.message || "An unexpected error stopped this page from drawing."
          }
          onRetry={() => this.setState({ error: null })}
        />
        <button
          type="button"
          onClick={() => void navigator.clipboard?.writeText(this.details()).catch(() => {})}
          className="cursor-pointer text-xs text-text-subtle underline-offset-2 hover:text-text hover:underline"
        >
          Copy details
        </button>
      </div>
    );
  }
}
