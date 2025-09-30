# RBF Solver Tool

A standalone Radial Basis Function (RBF) meshless solver for partial differential equations. Solves 2D Poisson equations using scattered point collocation with various RBF kernels.

## Features

- **Multiple Kernels**: IMQ, Gaussian, Multiquadric, Thin Plate Spline
- **Meshless Method**: No grid required - uses scattered collocation points
- **Real-time Visualization**: 3D surface visualization of solution
- **Error Analysis**: L² error computation against analytical solution
- **Interactive Controls**: Adjust all solver parameters in real-time
- **Center Point Visualization**: Show interior/boundary collocation points

## Mathematical Background

Solves the 2D Poisson boundary value problem:
```
-∆u + b·u = f  in Ω
u = g          on ∂Ω
```

Where:
- `Ω = [-1,1] × [-1,1]` (unit square)
- Test problem: `u_true(x,y) = (1-x²)(1-y²)`
- Source term `f` derived from analytical Laplacian

## RBF Kernels

### Inverse Multiquadric (IMQ)
- **Formula**: `φ(r) = (c² + r²)^β`
- **Parameters**: c (shape), β (power, typically -0.5)
- **Best for**: General purpose, good conditioning

### Gaussian
- **Formula**: `φ(r) = exp(-(εr)²)`
- **Parameters**: ε (shape parameter)
- **Best for**: Smooth solutions, exponential convergence

### Multiquadric
- **Formula**: `φ(r) = √(c² + r²)`
- **Parameters**: c (shape parameter)
- **Best for**: Interpolation problems

### Thin Plate Spline
- **Formula**: `φ(r) = r² log(r)`
- **Best for**: Surface fitting, no shape parameter needed

## Controls

### Kernel Settings
- **kernelType**: Choose RBF kernel
- **c**: Shape parameter (IMQ, Multiquadric)
- **beta**: Power parameter (IMQ)
- **eps**: Shape parameter (Gaussian)

### Discretization
- **interiorCount**: Number of interior collocation points
- **boundaryPerEdge**: Points per boundary edge (4 edges total)
- **gridResolution**: Evaluation grid resolution
- **seed**: Random seed for point generation

### PDE Parameters
- **b**: Coefficient in PDE (-∆u + b·u = f)
- **noiseLevel**: Gaussian noise added to source term

### Visualization
- **showCenters**: Display collocation points
- **showError**: Color surface by pointwise error

## Algorithm

1. **Point Generation**: Scatter interior points, uniform boundary points
2. **System Assembly**: Build collocation matrix A and RHS vector
3. **Linear Solve**: Gaussian elimination with partial pivoting
4. **Evaluation**: Reconstruct solution on regular grid
5. **Visualization**: Generate 3D mesh with error coloring

## Performance

- **Complexity**: O(n³) for system solve where n = total points
- **Memory**: O(n²) for system matrix
- **Typical**: ~400 points (196 interior + 96 boundary) solves in <1s

## Usage Tips

1. **Start Simple**: Use IMQ kernel with default parameters
2. **Point Distribution**: More interior points = better accuracy in domain
3. **Boundary Points**: More boundary points = better boundary accuracy
4. **Shape Parameters**: Smaller c/larger ε = more localized influence
5. **Conditioning**: IMQ with β=-0.5 usually well-conditioned

## Integration

```tsx
import RBFSolver from './RBFSolver';

// Standalone app
function MyApp() {
  return <RBFSolver />;
}

// Or use solver function directly
import { solveRBFProblem } from './RBFSolver';
const solution = solveRBFProblem('IMQ', {c: 0.25, beta: -0.5}, 196, 24, 1.0, 0.002, 42);
```

## Mathematical Details

The RBF approximation has the form:
```
u(x,y) ≈ Σⱼ αⱼ φ(‖(x,y) - (xⱼ,yⱼ)‖)
```

Where `αⱼ` are coefficients found by solving the linear system enforcing:
- PDE at interior points: `L[u] = f`
- Boundary conditions at boundary points: `u = g`
