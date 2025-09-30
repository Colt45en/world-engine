import { create } from 'zustand'

type UIState = {
    dyslexiaMode: boolean
    reducedMotion: boolean
    uiScale: number // 1 = 100%
    toggleReaderMode: () => void
    toggleReducedMotion: () => void
    setUiScale: (n: number) => void
    bumpUiScale: (delta: number) => void
}

export const useUI = create<UIState>((set, get) => ({
    dyslexiaMode: false,
    reducedMotion: false,
    uiScale: 1,
    toggleReaderMode: () => set(s => ({ dyslexiaMode: !s.dyslexiaMode })),
    toggleReducedMotion: () => set(s => ({ reducedMotion: !s.reducedMotion })),
    setUiScale: (n) => set({ uiScale: Math.min(1.6, Math.max(0.8, n)) }),
    bumpUiScale: (delta) => {
        const next = get().uiScale + delta
        set({ uiScale: Math.min(1.6, Math.max(0.8, Number(next.toFixed(2)))) })
    },
}))
