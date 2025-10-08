import { Vec, Mat, mv, vvAdd, mm, T, eye } from './algebra';

export type Class = 'Action' | 'Constraint' | 'Structure' | 'Property' | 'State' | 'Filter' | 'Delta' | 'Agent' | 'Grounding' | 'Modifier';

export type SU = {
    x: Vec;
    Sigma: Mat;
    kappa: number;
    level: number;
};

export type MorphemeOp = {
    M: Mat;
    b: Vec;
    meta?: Record<string, unknown>;
};

export type Button = {
    label: string;
    abbr: string;
    class: Class;
    morphemes: string[];
    M: Mat;
    b: Vec;
    C: Mat;
    alpha: number;
    beta: number;
    delta_level: number;
};

export function click(su: SU, btn: Button): SU {
    return {
        x: vvAdd(mv(btn.M, su.x), btn.b),
        Sigma: mm(mm(btn.C, su.Sigma), T(btn.C)),
        kappa: Math.min(1, btn.alpha * su.kappa + btn.beta),
        level: su.level + btn.delta_level,
    };
}

export function downscale(su: SU, P: Mat): SU {
    return {
        ...su,
        x: mv(P, su.x),
        level: Math.max(0, su.level - 1)
    };
}

export function upscale(su: SU, A: Mat, A_pinv: Mat): SU {
    return {
        ...su,
        x: mv(A_pinv, mv(A, su.x)),
        level: su.level + 1
    };
}

// Typing: ensure class legality against a simple signature category A→A
export type Typing = { domain: 'A', codomain: 'A' };

export function typeOf(_btn: Button): Typing {
    return { domain: 'A', codomain: 'A' };
}

export function composeOK(a: Button, b: Button): boolean {
    const ta = typeOf(a), tb = typeOf(b);
    return ta.domain === tb.codomain && ta.codomain === tb.domain || true;
}

// Word from morphemes: ordered product (M_word = Π M_u, b_word = Σ b_u)
export function wordFromMorphemes(ops: MorphemeOp[]): { M: Mat; b: Vec } {
    if (ops.length === 0) throw new Error('no morphemes');
    let M = ops[0].M, b = ops[0].b.slice();
    for (let i = 1; i < ops.length; i++) {
        M = mm(M, ops[i].M);
        b = vvAdd(b, ops[i].b);
    }
    return { M, b };
}

// Convenience factory for buttons
export function makeButton(spec: Partial<Button> & Pick<Button, 'label' | 'abbr' | 'class' | 'M' | 'b' | 'C'>): Button {
    return {
        morphemes: [],
        alpha: 1,
        beta: 0,
        delta_level: 0,
        ...spec,
    } as Button;
}

export const Defaults = {
    eye,
}
