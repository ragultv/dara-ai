import React, { useState, useEffect } from "react";
import { create } from "zustand";

interface ToastMessage {
  id: string;
  text: string;
  type: "info" | "success" | "error";
}

interface ToastStore {
  toasts: ToastMessage[];
  show: (text: string, type?: "info" | "success" | "error") => void;
  dismiss: (id: string) => void;
}

export const useToastStore = create<ToastStore>((set) => ({
  toasts: [],
  show: (text, type = "info") => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).substring(2, 5)}`;
    set((state) => ({ toasts: [...state.toasts, { id, text, type }] }));
    setTimeout(() => {
      set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) }));
    }, 3000);
  },
  dismiss: (id) => {
    set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) }));
  }
}));

export const toast = {
  show: (text: string, type?: "info" | "success" | "error") =>
    useToastStore.getState().show(text, type)
};

export const ToastContainer: React.FC = () => {
  const toasts = useToastStore((s) => s.toasts);
  const dismiss = useToastStore((s) => s.dismiss);

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none">
      {toasts.map((t) => (
        <div
          key={t.id}
          className="pointer-events-auto flex items-center justify-between px-3.5 py-2.5 rounded-xl border border-border bg-surface-raised shadow-lg text-xs font-medium text-text animate-in fade-in slide-in-from-bottom-2 duration-150"
        >
          <span>{t.text}</span>
          <button
            type="button"
            onClick={() => dismiss(t.id)}
            className="ml-3 text-text-subtle hover:text-text cursor-pointer"
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
};
