import React from "react";

interface ComposerAttachmentsProps {
  images: string[];
  onRemove: (index: number) => void;
  className?: string;
}

export const ComposerAttachments: React.FC<ComposerAttachmentsProps> = ({
  images,
  onRemove,
  className = ""
}) => {
  if (images.length === 0) return null;

  return (
    <div className={`flex flex-wrap gap-2 pb-2 border-b border-border/40 ${className}`}>
      {images.map((img, i) => (
        <div key={i} className="relative group/thumb">
          <img
            src={img}
            alt="Attachment thumbnail"
            className="w-14 h-14 object-cover rounded-xl border border-border shadow-xs"
          />
          <button
            type="button"
            aria-label="Remove image"
            onClick={() => onRemove(i)}
            className="absolute -top-1.5 -right-1.5 bg-text text-bg hover:bg-danger hover:text-white rounded-full w-4 h-4 flex items-center justify-center text-[10px] cursor-pointer shadow-sm transition-colors"
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
};
