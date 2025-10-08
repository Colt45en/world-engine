import { useMemo } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';

export function useGroundPointer(onPoint: (position: [number, number, number]) => void) {
    const { camera, gl } = useThree();
    const raycaster = useMemo(() => new THREE.Raycaster(), []);
    const plane = useMemo(() => new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), []);
    const ndc = new THREE.Vector2();

    return (event: any) => {
        if (!event) return;

        let ray = event.ray;

        if (!ray) {
            const rect = gl.domElement.getBoundingClientRect();
            const cx = (event.clientX ?? event?.nativeEvent?.clientX);
            const cy = (event.clientY ?? event?.nativeEvent?.clientY);
            if (typeof cx !== 'number' || typeof cy !== 'number') return;

            ndc.x = ((cx - rect.left) / rect.width) * 2 - 1;
            ndc.y = -((cy - rect.top) / rect.height) * 2 + 1;
            raycaster.setFromCamera(ndc, camera);
            ray = raycaster.ray;
        }

        const hit = new THREE.Vector3();
        ray.intersectPlane(plane, hit);
        if (Number.isFinite(hit.x) && Number.isFinite(hit.y) && Number.isFinite(hit.z)) {
            onPoint([hit.x, 0, hit.z]);
        }
    };
}
