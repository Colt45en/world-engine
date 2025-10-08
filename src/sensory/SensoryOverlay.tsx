// React Three Fiber sensory overlay component
import * as React from "react";
import * as THREE from "three";
import { Html } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { SceneMoment, byChannel, CHANNEL_COLORS, SensoryChannel } from "./sensoryTypes";

type Props = {
  moment: SceneMoment;
  attachTo?: THREE.Object3D;
  color?: string;
  visible?: boolean;
};

export function SensoryOverlay({
  moment,
  attachTo,
  color = "#4ecdc4",
  visible = true
}: Props) {
  const group = React.useRef<THREE.Group>(null!);
  const mat = React.useRef<THREE.MeshStandardMaterial>(null!);

  // Derived channel stacks
  const sight = byChannel(moment, "sight");
  const sound = byChannel(moment, "sound");
  const touch = byChannel(moment, "touch");
  const scent = byChannel(moment, "scent");
  const taste = byChannel(moment, "taste");
  const inner = byChannel(moment, "inner");

  // Subtle visual response: emissive pulse for sight/inner
  useFrame(({ clock }) => {
    if (!group.current || !visible) return;

    const t = clock.getElapsedTime();

    // Follow target (if provided)
    if (attachTo) {
      attachTo.getWorldPosition(group.current.position);
      group.current.position.y += 1.2; // Float above
    }

    // Emissive pulse strength tied to sight + inner strengths
    const sightIntensity = sight[0]?.strength ?? 0.4;
    const innerIntensity = inner[0]?.strength ?? 0.3;
    const combinedIntensity = sightIntensity * 0.8 + innerIntensity * 0.6;

    const glow = 0.3 + Math.abs(Math.sin(t * 1.2)) * combinedIntensity;

    if (mat.current) {
      mat.current.emissiveIntensity = glow;
    }

    // Gentle rotation based on inner perception
    group.current.rotation.y = t * 0.1 * (innerIntensity + 0.2);
  });

  if (!visible) return null;

  return (
    <group ref={group}>
      {/* Central hover ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.4, 0.48, 48]} />
        <meshStandardMaterial
          ref={mat}
          color={color}
          emissive={new THREE.Color(color)}
          emissiveIntensity={0.4}
          transparent
          opacity={0.7}
        />
      </mesh>

      {/* Textual cues arranged radially by channel */}
      <RadialText items={sight} channel="sight" radius={0.8} />
      <RadialText items={sound} channel="sound" radius={1.0} />
      <RadialText items={touch} channel="touch" radius={1.2} />
      <RadialText items={scent} channel="scent" radius={1.4} />
      <RadialText items={taste} channel="taste" radius={1.6} />
      <RadialText items={inner} channel="inner" radius={1.8} weight={600} />
    </group>
  );
}

function RadialText({
  items,
  channel,
  radius,
  weight = 400,
}: {
  items: { description: string; strength?: number }[];
  channel: SensoryChannel;
  radius: number;
  weight?: number;
}) {
  const count = Math.min(items.length, 4); // Limit to avoid clutter
  if (count === 0) return null;

  const hue = CHANNEL_COLORS[channel];

  return (
    <>
      {items.slice(0, count).map((detail, i) => {
        const angle = (i / count) * Math.PI * 2;
        const x = Math.cos(angle) * radius;
        const z = Math.sin(angle) * radius;
        const alpha = 0.6 + (detail.strength ?? 0.4) * 0.4;

        return (
          <Html
            key={`${channel}-${i}`}
            position={[x, 0.1, z]}
            center
            occlude="blending"
            distanceFactor={8}
          >
            <div
              style={{
                color: hue,
                opacity: alpha,
                fontFamily: "ui-monospace, SFMono-Regular, 'SF Mono', monospace",
                fontSize: 11,
                fontWeight: weight,
                background: "rgba(0, 0, 0, 0.7)",
                padding: "3px 8px",
                borderRadius: 6,
                border: `1px solid ${hue}40`,
                backdropFilter: "blur(6px)",
                whiteSpace: "nowrap",
                maxWidth: "200px",
                textAlign: "center",
                pointerEvents: "none",
                userSelect: "none"
              }}
            >
              {detail.description}
            </div>
          </Html>
        );
      })}
    </>
  );
}
