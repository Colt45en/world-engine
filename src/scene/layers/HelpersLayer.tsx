import React from 'react'
import { Grid, Stats } from '@react-three/drei'
import { useUI } from '../../state/ui'

export default function HelpersLayer() {
  const { reducedMotion } = useUI()

  return (
    <>
      {/* Grid helper - more subtle when motion is reduced */}
      <Grid
        position={[0, -2, 0]}
        args={[20, 20]}
        cellSize={1}
        cellThickness={reducedMotion ? 0.5 : 1}
        cellColor="#444444"
        sectionSize={5}
        sectionThickness={reducedMotion ? 1 : 1.5}
        sectionColor="#666666"
        fadeDistance={reducedMotion ? 15 : 25}
        fadeStrength={reducedMotion ? 0.8 : 1}
        infiniteGrid={false}
      />

      {/* Performance stats - less obtrusive when motion is reduced */}
      <Stats
        showPanel={0}
        className={reducedMotion ? 'opacity-60' : 'opacity-80'}
      />
    </>
  )
}
