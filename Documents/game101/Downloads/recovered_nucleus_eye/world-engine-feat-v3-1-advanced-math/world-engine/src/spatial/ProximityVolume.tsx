// Invisible spatial click handler (use for buttons/3D zones)
import * as React from "react";

type Props = {
  radius?: number;
  thickness?: number;
  children: React.ReactNode;
  onEnter?: () => void;
  onLeave?: () => void;
  onClick?: () => void;
  className?: string;
};

export function ProximityVolume({
  radius = 120,
  thickness = 8,
  children,
  onEnter,
  onLeave,
  onClick,
  className = "",
}: Props) {
  const ref = React.useRef<HTMLDivElement>(null);
  const [isInside, setIsInside] = React.useState(false);

  React.useEffect(() => {
    const div = ref.current;
    if (!div) return;

    const check = (evt: MouseEvent) => {
      const rect = div.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dx = evt.clientX - cx;
      const dy = evt.clientY - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);

      const inner = radius - thickness;
      const outer = radius;
      const inside = dist >= inner && dist <= outer;

      if (inside !== isInside) {
        setIsInside(inside);
        if (inside) onEnter?.();
        else onLeave?.();
      }
    };

    const handleClick = (evt: MouseEvent) => {
      const rect = div.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dx = evt.clientX - cx;
      const dy = evt.clientY - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);

      const inner = radius - thickness;
      const outer = radius;
      const inside = dist >= inner && dist <= outer;

      if (inside) {
        evt.preventDefault();
        evt.stopPropagation();
        onClick?.();
      }
    };

    document.addEventListener("mousemove", check);
    document.addEventListener("click", handleClick);

    return () => {
      document.removeEventListener("mousemove", check);
      document.removeEventListener("click", handleClick);
    };
  }, [radius, thickness, isInside, onEnter, onLeave, onClick]);

  return (
    <div
      ref={ref}
      className={`relative ${className}`}
      style={{
        width: radius * 2,
        height: radius * 2,
        cursor: isInside ? "pointer" : "default",
      }}
    >
      {/* Proximity zone (invisible) */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          borderRadius: "50%",
          border: isInside ? `1px solid rgba(100, 255, 100, 0.3)` : "none",
        }}
      />

      {/* Content area */}
      <div className="absolute inset-0 flex items-center justify-center">
        {children}
      </div>
    </div>
  );
}
