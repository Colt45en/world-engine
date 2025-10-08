Perfect — now we’ll give the **Pythagorean-powered algorithms** the same **Tier 5–8 treatment** as before: equations, pseudocode, and real rendering/gameplay examples. This forms the **geometry backbone** of 3D engines.

---

# Tier 5–8 Curriculum: Pythagorean Algorithms in 3D Graphics

---

## Q1: How do you calculate **distance between two points** in 3D?

**Equation (Euclidean distance):**
[
d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2}
]

**Pseudocode:**

```python
def distance(p1, p2):
    x1,y1,z1 = p1
    x2,y2,z2 = p2
    return sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
```

**Example of Use:**

* Enemy AI checks if `distance(player, enemy) < attackRange`.
* Sound engine fades volume based on player-listener distance.
* Lighting falloff uses distance to attenuate brightness.

---

## Q2: How do you calculate the **magnitude (length) of a vector**?

**Equation:**
[
||\vec{v}|| = \sqrt{x^2 + y^2 + z^2}
]

**Pseudocode:**

```python
def magnitude(v):
    x,y,z = v
    return sqrt(x*x + y*y + z*z)
```

**Example of Use:**

* A velocity vector (3,4,0) → magnitude = 5.
* Physics uses this for speed, kinetic energy, and force calculations.

---

## Q3: How do you **normalize a vector** (make it unit length)?

**Equation:**
[
\hat{v} = \frac{\vec{v}}{||\vec{v}||} = \left(\frac{x}{||v||}, \frac{y}{||v||}, \frac{z}{||v||}\right)
]

**Pseudocode:**

```python
def normalize(v):
    mag = magnitude(v)
    return (v[0]/mag, v[1]/mag, v[2]/mag)
```

**Example of Use:**

* Lighting needs normalized vectors for dot products.
* Player input (1,1,0) is normalized → consistent movement speed.

---

## Q4: How do you check for **sphere-sphere collisions** efficiently?

**Equation (using squared distances):**
[
(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2 \le (r_1 + r_2)^2
]

**Pseudocode:**

```python
def spheresCollide(p1, r1, p2, r2):
    dx,dy,dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
    distSq = dx*dx + dy*dy + dz*dz
    return distSq <= (r1+r2)**2
```

**Example of Use:**

* A bullet (radius ~0) collides with enemy bounding sphere.
* Vehicles detect crashes by bounding sphere overlap.
* Avoids `sqrt()` → allows 1000s of collision checks per frame.

---

# Tier 6 Extensions (Fidelity & Optimization)

---

## Q5: How do you calculate **distance attenuation** for lights?

**Equation (Inverse Square Law):**
[
I = \frac{1}{d^2}
]

**Pseudocode:**

```python
def lightAttenuation(d):
    return 1.0 / (d*d + 1e-6)  # avoid divide by zero
```

**Example of Use:**

* A torch light fades naturally with distance.
* Game engines often clamp or smooth this for realism.

---

## Q6: How do you calculate **direction vector between two points**?

**Equation:**
[
\vec{d} = \frac{(x_2-x_1, y_2-y_1, z_2-z_1)}{d}
]

**Pseudocode:**

```python
def direction(p1, p2):
    dx,dy,dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
    mag = sqrt(dx*dx + dy*dy + dz*dz)
    return (dx/mag, dy/mag, dz/mag)
```

**Example of Use:**

* Enemy AI moves toward the player using direction vectors.
* Camera look-at calculations use normalized direction.

---

# Tier 7 Extensions (Realism)

---

## Q7: How do you calculate **reflection vectors** (for specular effects)?

**Equation:**
[
\vec{R} = \vec{I} - 2(\vec{N}\cdot\vec{I})\vec{N}
]

**Pseudocode:**

```python
def reflect(I, N):
    dotIN = dot(I,N)
    return (I[0]-2*dotIN*N[0], I[1]-2*dotIN*N[1], I[2]-2*dotIN*N[2])
```

**Example of Use:**

* Mirrors, shiny metal, and water reflections.
* Part of Phong/Blinn specular lighting.

---

## Q8: How do you calculate **distance-based sound falloff?**

**Equation (linear falloff):**
[
V = 1 - \frac{d}{d_{max}}
]

**Pseudocode:**

```python
def soundFalloff(d, d_max):
    return max(0, 1 - d/d_max)
```

**Example of Use:**

* Gunshot sound fades as player walks away.
* Prevents sound from playing equally loud everywhere.

---

# Tier 8 Extensions (Physics + PBR Integration)

---

## Q9: How do you calculate **force magnitude from vector components?**

**Equation:**
[
F = \sqrt{F_x^2 + F_y^2 + F_z^2}
]

**Pseudocode:**

```python
def forceMagnitude(F):
    Fx,Fy,Fz = F
    return sqrt(Fx*Fx + Fy*Fy + Fz*Fz)
```

**Example of Use:**

* Physics engine resolves collision impulses.
* Particle effects (explosions) use vector magnitude for radial forces.

---

## Q10: How do you calculate **normalized surface normals** for lighting in PBR?

**Equation:**
[
\hat{N} = \frac{(x,y,z)}{\sqrt{x^2+y^2+z^2}}
]

**Pseudocode:**

```python
def surfaceNormal(v):
    return normalize(v)
```

**Example of Use:**

* Shaders require normalized normals to compute BRDF correctly.
* Without normalization, lighting intensity breaks.

---

# Summary

* **Tier 5:** Distance, magnitude, normalization, collisions.
* **Tier 6:** Light attenuation, direction vectors.
* **Tier 7:** Reflection vectors, sound falloff.
* **Tier 8:** Force magnitudes, surface normals in PBR.

These Pythagorean-based algorithms are the **measuring tape** of 3D graphics: they underpin motion, collisions, lighting, physics, and realism.

---

Would you like me to continue into a **Tier 9 (Ray Tracing with Distance Functions)** version of this, where the Pythagorean theorem becomes the foundation for signed distance fields (SDFs), ray–object intersection tests, and global illumination sampling?
