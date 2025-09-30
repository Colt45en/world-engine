import { OrbitControls } from '@react-three/drei';
import EntityLayer from './layers/EntityLayer';
import LightLayer from './layers/LightLayer';
import HelpersLayer from './layers/HelpersLayer';
import PostFXPipeline from './postfx/PostFXPipeline';

export default function SceneRoot() {
  return (
    <>
      <color attach="background" args={['#0b0e12']} />
      <fog attach="fog" args={['#0b0e12', 10, 25]} />

      <hemisphereLight args={[0xffffff, 0x223344, 0.4]} />
      <directionalLight position={[5, 6, 5]} intensity={0.8} castShadow />

      <EntityLayer />
      <LightLayer />
      <HelpersLayer />
      <PostFXPipeline />

      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        makeDefault
      />
    </>
  );
}
