// Nexus Forge: Enhanced Recursive Scene-State Visual Forge + Mythic Symbol Codex + Cube Projection Themes
// Core Tech: React + Three.js + Zustand (for state) + Tailwind CSS + Improved Interactions

import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Accordion, AccordionItem } from '@/components/ui/accordion';
import create from 'zustand';

// Zustand Store
const useStore = create((set) => ({
    cubes: [
        { id: 1, position: [-1, 0, -1], tag: 'Seeker' },
        { id: 2, position: [1, 0, -1], tag: 'Weaver' },
    ],
    spotlight: { enabled: false, position: [0, 2, 0] },
    selectedCube: null,
    theme: 'default',
    toggleSpotlight: () => set((s) => ({ spotlight: { ...s.spotlight, enabled: !s.spotlight.enabled } })),
    addCube: () => set((s) => ({ cubes: [...s.cubes, { id: Date.now(), position: [0, 0, 0], tag: 'Newborn' }] })),
    selectCube: (id) => set((s) => {
        const selected = s.cubes.find(c => c.id === id);
        return { selectedCube: id, theme: selected ? selected.tag : 'default' };
    }),
}));

const themeMap = {
    default: { bg: '#000000', fog: '#222222', title: 'Neutral Void' },
    Seeker: { bg: '#001122', fog: '#223344', title: 'Karma Echoes' },
    Weaver: { bg: '#331144', fog: '#442255', title: 'Threading the Dao' },
    Newborn: { bg: '#0a0a0a', fog: '#1a1a1a', title: 'Blank Origin' },
};

function Cube({ position, id, tag }) {
    const selectCube = useStore((s) => s.selectCube);
    return (
        <mesh position= { position } onClick = {() => selectCube(id)
}>
    <boxGeometry args={ [1, 1, 1] } />
        < meshStandardMaterial color = { 'skyblue'} />
            </mesh>
  );
}

function Spotlight() {
    const spotlight = useStore((s) => s.spotlight);
    if (!spotlight.enabled) return null;
    return (
        <pointLight position= { spotlight.position } intensity = { 1} color = "white" />
  );
}

function MythicCodex() {
    return (
        <Accordion type= "multiple" className = "text-sm" >
            <AccordionItem title="🔸 The Oars of Karma" >
                Each stroke of the oar determines the flow of fate.Some who row too forcefully are thrown into the past, forced to relive their mistakes, while those who drift aimlessly risk being swallowed by the current of Oblivion.
      </AccordionItem>
                    < AccordionItem title = "🔸 The Drowned Monastery" >
                        A sunken temple hidden within the river, where monks who failed to reach enlightenment linger as half - formed phantoms, endlessly chanting the names of their past lives.It is said that within their chants lies a secret Dao… but only those who listen without desire may understand it.
      </AccordionItem>
                            < AccordionItem title = "🔸 The Whirlpool of Forgotten Names" >
                                A celestial vortex said to erase the past entirely.Those who surrender to it are freed from all karma, yet they become nameless wanderers, neither mortal nor divine, trapped outside the cycle of rebirth.
      </AccordionItem>
                                    < AccordionItem title = "🔸 The Void of Echoes" >
                                        Where all thoughts become reality before dissolving back into pure potential.Masters say the void contains every possibility without judgment or attachment.Those who enter find their deepest fears and highest aspirations mirrored in perfect balance.
      </AccordionItem>
                                            < AccordionItem title = "🔸 The Crystal Nexus" >
                                                A convergence point where all timelines intersect.Those who stand at its center can glimpse alternate versions of themselves—paths not taken, choices unmade.The wise understand it is not for changing one's past, but for accepting the perfection of the present.
                                                    </AccordionItem>
                                                    </Accordion>
    );
}

export default function NexusForge() {
    const { cubes, toggleSpotlight, addCube, spotlight, theme } = useStore();
    const activeTheme = themeMap[theme] || themeMap['default'];

    return (
        <div className= "grid grid-cols-4 h-screen" >
        <div className="col-span-3" >
            <Canvas camera={ { position: [0, 2, 5], fov: 60 } } shadows >
                <color attach="background" args = { [activeTheme.bg]} />
                    <fog attach="fog" args = { [activeTheme.fog, 2, 10]} />
                        <ambientLight intensity={ 0.3 } />
                            < Spotlight />
                        {
                            cubes.map((cube) => (
                                <Cube key= { cube.id } { ...cube } />
          ))
                        }
                            < OrbitControls />
                            </Canvas>
                            </div>
                            < div className = "p-4 bg-gray-900 text-white overflow-y-auto" >
                                <Card className="mb-4" >
                                    <CardContent>
                                    <h2 className="text-xl font-bold mb-2" >🧱 Nexus Forge Control </h2>
                                        < p className = "text-xs mb-2 text-gray-400" > Current Theme: { activeTheme.title } </p>
                                            < Button onClick = { addCube } className = "mb-2 w-full" >➕ Add Cube </Button>
                                                < Button onClick = { toggleSpotlight } className = "w-full" >
                                                    { spotlight.enabled ? '💡 Disable Spotlight' : '💡 Enable Spotlight' }
                                                    </Button>
                                                    </CardContent>
                                                    </Card>
                                                    < Card >
                                                    <CardContent>
                                                    <h2 className="text-lg font-bold mb-2" >📜 Mythic Symbol Codex </h2>
                                                        < MythicCodex />
                                                        </CardContent>
                                                        </Card>
                                                        </div>
                                                        </div>
  );
}
