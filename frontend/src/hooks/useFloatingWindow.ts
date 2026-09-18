import { useCallback, useEffect, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent, RefObject } from "react";

export interface FloatRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FloatOptions {
  minW?: number;
  minH?: number;
  margin?: number;
  center?: boolean;
}

/**
 * Move (via a title-bar mousedown), resize (via the bottom-right grip) and
 * clamp a floating window inside its container, so it can never be dragged
 * out of reach.
 */
export function useFloatingWindow(
  initial: FloatRect,
  containerRef: RefObject<HTMLElement | null>,
  { minW = 320, minH = 200, margin = 8, center = false }: FloatOptions = {},
) {
  const [rect, setRect] = useState<FloatRect>(initial);
  const didCenter = useRef(false);
  const drag = useRef<{
    mode: "move" | "resize";
    ox: number;
    oy: number;
    sx: number;
    sy: number;
  } | null>(null);

  useEffect(() => {
    const clampRect = (r: FloatRect): FloatRect => {
      const el = containerRef.current;
      const cw = el ? el.clientWidth : window.innerWidth;
      const ch = el ? el.clientHeight : window.innerHeight;
      const w = Math.min(Math.max(r.w, minW), Math.max(minW, cw - margin * 2));
      const h = Math.min(Math.max(r.h, minH), Math.max(minH, ch - margin * 2));
      const x = Math.min(Math.max(r.x, margin), Math.max(margin, cw - w - margin));
      const y = Math.min(Math.max(r.y, margin), Math.max(margin, ch - h - margin));
      return { x, y, w, h };
    };

    const onMove = (e: MouseEvent) => {
      const d = drag.current;
      if (!d) return;
      if (d.mode === "move") {
        setRect((r) =>
          clampRect({ ...r, x: d.sx + (e.clientX - d.ox), y: d.sy + (e.clientY - d.oy) }),
        );
      } else {
        setRect((r) =>
          clampRect({ ...r, w: d.sx + (e.clientX - d.ox), h: d.sy + (e.clientY - d.oy) }),
        );
      }
    };
    const onUp = () => {
      drag.current = null;
    };
    const onResize = () =>
      setRect((r) => {
        const el = containerRef.current;
        const cw = el ? el.clientWidth : window.innerWidth;
        const ch = el ? el.clientHeight : window.innerHeight;
        let next = r;
        if (center && !didCenter.current) {
          didCenter.current = true;
          next = { ...r, x: (cw - r.w) / 2, y: (ch - r.h) / 2 };
        }
        return clampRect(next);
      });

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    window.addEventListener("resize", onResize);
    onResize(); // initial clamp / centering
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      window.removeEventListener("resize", onResize);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [containerRef, minW, minH, margin, center]);

  const onTitleMouseDown = useCallback(
    (e: ReactMouseEvent) => {
      drag.current = { mode: "move", ox: e.clientX, oy: e.clientY, sx: rect.x, sy: rect.y };
    },
    [rect.x, rect.y],
  );

  const onResizeMouseDown = useCallback(
    (e: ReactMouseEvent) => {
      e.stopPropagation();
      drag.current = { mode: "resize", ox: e.clientX, oy: e.clientY, sx: rect.w, sy: rect.h };
    },
    [rect.w, rect.h],
  );

  return { rect, onTitleMouseDown, onResizeMouseDown };
}
