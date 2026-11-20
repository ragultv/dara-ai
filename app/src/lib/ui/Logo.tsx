import React from "react";

interface LogoProps {
  className?: string;
  alt?: string;
}

/**
 * The Dara mark. The file has a near-white half that would vanish on a light page, so there are two images
 * (`dara.png` for dark surfaces, `dara-light.png` for light ones) and app.css shows the one that matches `data-theme`.
 */
export const Logo: React.FC<LogoProps> = ({ className = "", alt = "Dara" }) => (
  <>
    <img
      src="/dara-light.png"
      alt={alt}
      className={`logo-on-light ${className}`}
      draggable={false}
    />
    <img
      src="/dara.png"
      alt=""
      aria-hidden="true"
      className={`logo-on-dark ${className}`}
      draggable={false}
    />
  </>
);
