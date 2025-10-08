init:
  EMA_alpha = 0.2
  norm_ranges = rolling_minmax()  # per feature
  choose shape_mode ∈ {heart, rose, spiral, dragon, lissajous, circle}

per audio frame:
  L  = rms(frame)
  Xk = fft_mag(frame)
  chi = spectral_centroid(Xk)
  f0  = estimate_pitch(frame)      # optional
  Phi = spectral_flux(Xk, Xk_prev)
  B   = stereo_balance(frameL, frameR)
  fb  = BPM/60 or beat_tracker()
  LAM = lfo(rate=fb/d_AM, t)        # AM LFO if desired
  LFM = lfo(rate=d_FM*fb, t)        # FM LFO if desired

  # smooth + normalize
  L~  = ema_norm(L)
  chi~= ema_norm(chi)
  Phi~= ema_norm(Phi)

  # map to parameters (example: Rose)
  n   = max(1, round(1 + 6 * chi~))
  a   = a0 * (0.6 + 0.4 * L~)
  δ   = 2π * phase(fb, t)

  # sample the curve
  pts = []
  if shape_mode == "rose":
      for θ in linspace(0, 2π, M):
          r = a * cos(n*θ + δ)
          x = r * cos(θ); y = r * sin(θ)
          pts.append((x,y))
  elif shape_mode == "heart":
      # parametric heart for stability
      for θ in linspace(0, 2π, M):
          x = S*16*sin(θ)^3
          y = S*(13*cos(θ)-5*cos(2θ)-2*cos(3θ)-cos(4θ))
          x *= 1 + β*LAM; y *= 1 + β*LAM
          pts.append(rot((x,y), 2π*phase(fb,t)))
  ... # spiral, lissajous, circle similar

  draw_polyline(pts, color=hsv(chi~, 0.8, 0.9), thickness=map_thickness(L~))


  import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Audio Settings ---
blocksize = 1024
samplerate = 44100
audio_buffer = np.zeros(blocksize)

def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = indata[:, 0]  # take mono channel

stream = sd.InputStream(callback=audio_callback,
                        channels=1,
                        blocksize=blocksize,
                        samplerate=samplerate)
stream.start()

# --- Feature Extraction ---
def extract_features(buf):
    # RMS loudness
    L = np.sqrt(np.mean(buf**2))
    # Spectrum + centroid
    spectrum = np.abs(np.fft.rfft(buf))
    freqs = np.fft.rfftfreq(len(buf), 1/samplerate)
    if np.sum(spectrum) > 0:
        centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
    else:
        centroid = 0
    return L, centroid

# --- Shape Generators ---
def heart(L, centroid):
    t = np.linspace(0, 2*np.pi, 800)
    x = 16 * np.sin(t)**3 * (0.5+2*L)
    y = (13*np.cos(t) - 5*np.cos(2*t)
         - 2*np.cos(3*t) - np.cos(4*t)) * (0.5+2*L)
    return x, y

def rose(L, centroid):
    n = int(2 + round(6*centroid/(samplerate/2)))  # petals from centroid
    a = 1 + 3*L
    theta = np.linspace(0, 2*np.pi, 800)
    r = a * np.cos(n*theta)
    return r*np.cos(theta), r*np.sin(theta)

def spiral(L, centroid):
    theta = np.linspace(0, 6*np.pi, 1000)
    r = (0.1+L) * theta
    return r*np.cos(theta), r*np.sin(theta)

def dragon(L, centroid, iterations=12):
    # simple iterative dragon curve approximation
    points = [0+0j, 1+0j]
    for i in range(iterations):
        new_points = [points[0]]
        for j in range(len(points)-1):
            mid = (points[j] + points[j+1]) / 2
            rot = (points[j+1] - points[j]) * 1j
            mid = points[j] + rot/2
            new_points.extend([mid, points[j+1]])
        points = new_points
    pts = np.array(points) * (0.5+3*L)
    return pts.real, pts.imag

def lissajous(L, centroid):
    t = np.linspace(0, 2*np.pi, 1000)
    a = 3
    b = int(2 + round(centroid/1000))  # map centroid to frequency ratio
    x = np.sin(a*t) * (1+3*L)
    y = np.cos(b*t) * (1+3*L)
    return x, y

def circle(L, centroid):
    theta = np.linspace(0, 2*np.pi, 500)
    r = 1 + 3*L
    return r*np.cos(theta), r*np.sin(theta)

shapes = [heart, rose, spiral, dragon, lissajous, circle]
shape_names = ["Heart", "Rose", "Spiral", "Dragon", "Lissajous", "Circle"]
current_shape = 0

# --- Matplotlib Visualization ---
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
title = ax.set_title("Shape: " + shape_names[current_shape])

def on_key(event):
    global current_shape
    if event.key in map(str, range(1, 7)):
        current_shape = int(event.key) - 1
        title.set_text("Shape: " + shape_names[current_shape])
fig.canvas.mpl_connect('key_press_event', on_key)

def update(frame):
    global audio_buffer
    L, centroid = extract_features(audio_buffer)
    x, y = shapes[current_shape](L, centroid)
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig, update, interval=50, blit=True)
plt.show()
