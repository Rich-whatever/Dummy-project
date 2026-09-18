import { useEffect, useRef } from "react";
import type { OrbState } from "../types";

const N = 90;
const ORB_R = 170;

function fibSphere(n: number): [number, number, number][] {
  const pts: [number, number, number][] = [];
  const g = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    pts.push([r * Math.cos(g * i), y, r * Math.sin(g * i)]);
  }
  return pts;
}

const BASE = fibSphere(N);

const EDGES: [number, number][] = [];
for (let i = 0; i < N; i++) {
  for (let j = i + 1; j < N; j++) {
    const d = BASE[i][0] * BASE[j][0] + BASE[i][1] * BASE[j][1] + BASE[i][2] * BASE[j][2];
    if (d > 0.93) EDGES.push([i, j]);
  }
}

// ─── Neural Orb ───────────────────────────────────────────────────────────────

interface Fire { i: number; j: number; life: number }

export function NeuralOrb({ orbState, isDark }: { orbState: OrbState; isDark: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const orbStateRef = useRef(orbState);
  const isDarkRef = useRef(isDark);
  const animRef = useRef<number>(0);

  useEffect(() => { orbStateRef.current = orbState; }, [orbState]);
  useEffect(() => { isDarkRef.current = isDark; }, [isDark]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let W = 0, H = 0;
    let angleY = 0, angleX = 0;
    let speed = 0.003;
    let time = 0;
    const fires: Fire[] = [];
    const pulses = new Map<number, number>(); // particle index → life

    function resize() {
      W = canvas!.offsetWidth;
      H = canvas!.offsetHeight;
      canvas!.width = W;
      canvas!.height = H;
    }
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    function rotatePoint(
      [x, y, z]: [number, number, number],
      ay: number,
      ax: number
    ): [number, number, number] {
      const x1 = x * Math.cos(ay) - z * Math.sin(ay);
      const z1 = x * Math.sin(ay) + z * Math.cos(ay);
      const y2 = y * Math.cos(ax) - z1 * Math.sin(ax);
      const z2 = y * Math.sin(ax) + z1 * Math.cos(ax);
      return [x1, y2, z2];
    }

    function project([x, y, z]: [number, number, number], cx: number, cy: number): [number, number, number] {
      const fov = 420;
      const s = fov / (fov + z * ORB_R);
      return [cx + x * ORB_R * s, cy + y * ORB_R * s, s];
    }

    function draw() {
      if (!ctx || W === 0) { animRef.current = requestAnimationFrame(draw); return; }

      ctx.clearRect(0, 0, W, H);

      const thinking = orbStateRef.current === "thinking";
      const dark = isDarkRef.current;
      const accent = dark ? "79,144,232" : "150,108,20";
      const cx = W / 2;
      const cy = H / 2-15;
      if (dark) {
        const disc = ctx.createRadialGradient(cx, cy, 0, cx, cy, ORB_R * 1.7);
        disc.addColorStop(0.5, "rgba(40, 36, 120, 0.08)");
        disc.addColorStop(0.7, "rgba(5, 60, 181, 0.04)");
        disc.addColorStop(1, "rgba(20, 8, 240, 0)");
        ctx.fillStyle = disc;
        ctx.fillRect(0, 0, W, H);
      }
      // Light mode: a faint warm backdrop disc so the gold particles have
      // something to sit against (defines the sphere on a near-white bg).
      if (!dark) {
        const disc = ctx.createRadialGradient(cx, cy, 0, cx, cy, ORB_R * 1.7);
        disc.addColorStop(0.5, "rgba(150,108,20,0.08)");
        disc.addColorStop(0.7, "rgba(150,108,20,0.04)");
        disc.addColorStop(1, "rgba(150,108,20,0)");
        ctx.fillStyle = disc;
        ctx.fillRect(0, 0, W, H);
      }

      // Smooth speed
      const targetSpeed = thinking ? 0.013 : 0.003;
      speed += (targetSpeed - speed) * 0.04;
      angleY += speed;
      angleX = Math.sin(time * 0.0008) * 0.35;
      time++;

      // Project all points
      const proj: [number, number, number][] = BASE.map(p =>
        project(rotatePoint(p, angleY, angleX), cx, cy)
      );

      // Glow beneath orb
      const glowPulse = thinking
        ? 0.13 + 0.05 * Math.sin(time * 0.08)
        : 0.055 + 0.015 * Math.sin(time * 0.025);
      const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, ORB_R * 1.35);
      glow.addColorStop(0, `rgba(${accent},${glowPulse * (dark ? 1 : 0.85)})`);
      glow.addColorStop(0.5, `rgba(${accent},${glowPulse * 0.3 * (dark ? 1 : 0.6)})`);
      glow.addColorStop(1, `rgba(${accent},0)`);
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, W, H);

      // Thinking: add fires
      if (thinking && Math.random() < 0.06) {
        fires.push({ i: Math.random() * N | 0, j: Math.random() * N | 0, life: 1 });
      }
      // Thinking: add pulses
      if (thinking && Math.random() < 0.04) {
        pulses.set(Math.random() * N | 0, 1);
      }
      // Update pulses
      for (const [k, v] of pulses) {
        if (v <= 0) pulses.delete(k); else pulses.set(k, v - 0.03);
      }

      // Draw edges
      for (const [a, b] of EDGES) {
        const [x1, y1, s1] = proj[a];
        const [x2, y2, s2] = proj[b];
        const avgS = (s1 + s2) / 2;
        const alpha = avgS * (thinking ? 0.14 : 0.07) * (dark ? 1 : 1.6);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = `rgba(${accent},${alpha})`;
        ctx.lineWidth = dark ? 0.6 : 0.9;
        ctx.stroke();
      }

      // Draw fires
      for (let f = fires.length - 1; f >= 0; f--) {
        const fire = fires[f];
        const [x1, y1] = proj[fire.i];
        const [x2, y2] = proj[fire.j];
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = `rgba(${accent},${fire.life * 0.55})`;
        ctx.lineWidth = 1;
        ctx.stroke();
        fire.life -= 0.035;
        if (fire.life <= 0) fires.splice(f, 1);
      }

      // Draw particles
      // Sort by z for depth
      const order = proj.map((_, i) => i).sort((a, b) => proj[a][2] - proj[b][2]);
      for (const i of order) {
        const [px, py, s] = proj[i];
        const pulse = pulses.get(i) ?? 0;
        const r = (1 + pulse * 2.5) * Math.max(0.8, s * (dark ? 1.6 : 1.9));
        const alpha = Math.min(1, s * (thinking ? 0.75 : 0.45) * (dark ? 1 : 1.7) + pulse * 0.4);

        if (pulse > 0) {
          ctx.beginPath();
          ctx.arc(px, py, r * 2.5, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${accent},${pulse * 0.2})`;
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${accent},${alpha})`;
        ctx.fill();
      }

      animRef.current = requestAnimationFrame(draw);
    }

    draw();
    return () => {
      cancelAnimationFrame(animRef.current);
      ro.disconnect();
    };
  }, []);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" style={{ zIndex: 0 }} />;
}
