export function runMathematicalDemo(): Promise<{
    engines: {
        engine2D: any;
        engine3D: any;
        engine4D: any;
    };
    testResults: any;
    demonstrations: string[];
}>;
export function quickSelfTest(): Promise<({
    name: string;
    success: any;
    error?: never;
} | {
    name: string;
    success: boolean;
    error: any;
})[]>;
//# sourceMappingURL=world-engine-math-demo.d.ts.map