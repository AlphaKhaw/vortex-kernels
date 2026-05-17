# `fftconv_func` — HCM's FFT-conv reference

> **Tour target 1 of 12** · [index](README.md) · next: [parallel_fir](parallel_fir.md)

## Source

| | |
|---|---|
| Upstream | `vortex/model/engine.py:38-83` |
| Vendored copy | `vortex_kernels/reference/engine_ref.py:51-96` |
| Called from | `engine.py:176` only (HCM branch in `parallel_fir`) |

## Overall purpose

**Apply a depthwise long convolution `y = u * k` (one filter per channel,
no channel mixing) using FFTs, then add a per-channel skip-residual
`D * u`.**

Why FFT and not direct `conv1d`? At HCM's filter length `K=128` and
sequence `L=8192`:

- Direct conv: `O(L · K)` per channel = ~1 M ops × D channels
- FFT conv: `O(L log L)` per channel = ~110 K ops × D channels (plus
  three cuFFT launches)

FFT wins by ~10×. Long convolutions in Hyena always go FFT.

What this kernel **isn't doing**: any mixing across channels. Each of
the `D=4096` channels has its own independent filter `k[d]` and its
own output. It's `D` separate 1D convolutions running in parallel,
packed into one batched call.

## Realistic input shapes (evo2_7b HCM at L=8192)

```
INPUTS to fftconv_func
══════════════════════

u  ( 1 , 4096 , 8192 )  fp32     ← activations
   └─┬─┘└──┬──┘└──┬──┘
     B   D=4096  L=8192
   batch channels sequence

k  ( 4096 , 1 , 128 )   fp32     ← filter (HCM has K=128 taps per channel)
   └──┬──┘└┬┘└──┬──┘
     D   1  K=128

D  ( 4096 , )           fp32     ← per-channel skip gain
   └──┬──┘                          (NOT the channel count — the bias parameter)
    D=4096
```

> ⚠️ **Naming gotcha**: the parameter `D` (skip gain) and the channel
> dimension `D` share a letter. Different things. In this doc I write
> `D=4096` for the channel count and `D[:, None]` or "the skip gain"
> for the parameter.

## Inputs explained

### `u` — activations

"Activation" is the deep-learning term for **the tensor flowing between
layers**. It started as an analogy to biological neurons firing in
response to input, but in practice it just means "the data that gets
transformed at each layer". In a transformer you'd also call these
"hidden states".

```
Each Hyena layer:

  activations_in  ──►  [ layer math ]  ──►  activations_out
  (B, D, L) bf16        ▒▒▒▒▒▒▒▒▒▒▒        (B, D, L) bf16
                          ▒▒▒▒▒▒▒
                          ▒▒▒▒▒
```

For `fftconv_func`, `u` is the activations *after* the gating prep
(where engine.py computed `u = x1 * v` at line 169). Same shape
`(B, D, L)` going in and coming out — the conv doesn't change the
tensor's dimensions, just the values inside.

### `k` — filter

You already get this. Quick recap: depthwise conv1d filter, one
independent filter per channel, shape `(D, 1, K)`. The "1" is a
vestige of PyTorch's general conv1d weight shape
`(out_channels, in_channels/groups, kernel_size)`; with `groups=D`
("depthwise"), `in_channels/groups = D/D = 1`.

### `D` — per-channel skip gain (the parameter, not the channel count)

A learned `(D,)` fp32 tensor. Element-wise multiplies the input `u`
and adds to the conv output:

```
final_output = conv_output + (D[:, None] * u)
```

**Why is this a separate parameter and not just `h[t=0]`?** Vortex
follows a **state-space model** convention from control theory. In
state-space terms, any linear time-invariant (LTI) system looks like:

```
x(t+1)  =  A · x(t)  +  B · u(t)        ← state evolution
y(t)    =  C · x(t)  +  D · u(t)        ← output equation
                       ───┬───
                          │
                  THIS is the "D" parameter:
                  direct feedthrough from input to output,
                  bypassing the hidden state.
```

Mapping to vortex:
- The modal params `(residues, log_poles)` encode the `A, B, C` matrices
  (the state evolution + how state maps to output)
- The `D` parameter is exactly that direct-feedthrough term — input
  passes straight to the output, scaled per-channel

**Why have it as a separate parameter** instead of letting `h[t=0]`
absorb it?

1. The modal form `h[d, t] = Σ_s residue[s] · exp(log_poles[s] · t)` is
   built to model "memory" (the decay-over-time part). `t=0` is a
   boundary case.
2. Initialization is cleaner: `D` can start at 0 (no pass-through), the
   modal params can start small (no memory yet), and gradient descent
   decides which one to grow.
3. `D` is unconstrained (any real value per channel). The modal form
   has rational-function structure that limits what `h[t=0]` can be.
   Separating them gives the network more expressive freedom.

The naming `D` for direct-feedthrough is **the standard DSP/control-
theory convention**, predating deep learning by decades. Awkward that
it collides with the channel-dimension symbol in this codebase.

## Background: FFT in 60 seconds

If `rfft`/`irfft` feel mysterious, this primer answers the three
common questions in one place.

### What rfft and irfft do (semantically)

**rfft** = Real-input Fast Fourier Transform. Maps a real-valued
time-domain signal to its **frequency spectrum**:

```
TIME DOMAIN                          FREQUENCY DOMAIN
─────────────                        ─────────────────

real-valued signal                   complex-valued spectrum
u[0], u[1], ..., u[N-1]      ─►      u_f[0], u_f[1], ..., u_f[N/2]

N real numbers                       N/2+1 complex numbers
                                     (≈ N real numbers total —
                                      each complex = 2 reals)

  amplitude over time                  magnitude / phase over frequency
```

**irfft** is the exact inverse: take a frequency spectrum, produce the
real time-domain signal. `irfft(rfft(u)) == u` up to floating-point
precision. **Lossless**, no information added or removed.

### Why N/2+1 (answering "why L+1")?

For a length-N real-valued signal, the full FFT would produce N complex
coefficients. But real-valued input has **Hermitian conjugate symmetry**
in its FFT:

```
fft[k] = conj(fft[N - k])     for all k
```

That means bins `0..N/2` carry unique information; bins `N/2+1..N-1`
are just complex conjugates of earlier ones. Storing all N would be
duplication.

`rfft` exploits this and only stores the first `N/2+1` bins:

```
For N = 16384 (= 2L = 2·8192):
  Full FFT output:         16384 complex bins
  rfft output (unique):    16384/2 + 1 = 8193 complex bins
                                          └────┬────┘
                                           That's L+1
```

The +1 is because bin indices run `0, 1, 2, ..., N/2`. That's
`N/2 - 0 + 1 = N/2 + 1` bins. The DC bin (bin 0) is real-valued; the
Nyquist bin (bin N/2) is also real-valued; the bins in between are
complex.

### Is this like a VAE encoder/decoder?

**Surface yes, structure no.**

| | VAE | rfft/irfft |
|---|---|---|
| Direction | encoder → decoder | rfft → irfft |
| Type | Learned **non-linear** | Fixed **linear orthonormal** |
| Output dim | Smaller (bottleneck) | Same info content (lossless) |
| Invertible | No (lossy by design) | Yes (exact inverse) |
| Parameters | Weights to learn | Zero (fixed sinusoid basis) |
| Why use it | Compress to learnable manifold | Convert convolution into multiplication |

The **shared intuition**: "transform to a different representation,
do something cheaper there, transform back". That pattern shows up
everywhere — PCA, autoencoders, FFTs, Laplace transforms, change of
basis in linear algebra. Once you see it once, you'll see it a lot.

What's different here:
- VAEs *learn* the projection (the encoder weights). FFT uses a *fixed*
  basis of sinusoids (no parameters).
- VAEs *lose* information (that's the whole point — bottleneck for
  compression / regularization). FFT preserves information *exactly*.
- VAEs *compress* dimensionally. FFT preserves dimensionality (N real
  numbers in → N/2+1 complex numbers = N real numbers, just re-arranged).

### Concrete analogy — the graphic equalizer

Easiest mental model: a graphic equalizer on a music player.

```
TIME DOMAIN VIEW                       FREQUENCY DOMAIN VIEW
─────────────────                      ─────────────────────

The waveform you hear:                  The equalizer sliders:

  amplitude                                    ┌──┬──┬──┬──┬──┐
       ▲ ╱╲     ╱╲                             │ ▒│ ▒│ ▒│ ▒│ ▒│
       │╱  ╲   ╱  ╲      /                     │ ▒│ ▒│ ▒│ ▒│ ▒│
       │    ╲ ╱    ╲    ╱                      │ ▒│  │  │  │ ▒│
   ────┼─────V──────╲──╱── time                │ ▒│  │  │  │  │
                     ╲╱                        └──┴──┴──┴──┴──┘
                                               bass  mid  treble
   (raw audio samples)                          (gain per freq band)
```

Convolution in time-domain ≡ multiplication in frequency-domain. To
filter a song's bass:

- **Time-domain way**: convolve the audio with a "low-pass impulse
  response" — expensive, O(L·K)
- **Frequency-domain way**: rfft the audio, multiply each frequency
  bin by your gain setting (e.g. 1.5× low frequencies, 0.5× high
  frequencies), irfft back — cheap, O(L log L)

That's exactly what `fftconv_func` does. The filter `k` is the
equalizer's frequency-response settings (learned per layer per channel);
the activations `u` are the "audio". rfft → multiply → irfft is the
cheap path.

### Full pipeline as the equalizer analogy

```
   u (raw "audio")                   y (filtered "audio")
   ─────────────────                 ─────────────────────
      amplitude                          amplitude
           ▲                                  ▲
       ────┼──── time ────►            ──────┼──── time ────►
           │                                  │  smoother
           │                                  │  curve

           │ rfft                             ▲ irfft
           ▼                                  │

   u_f (raw "spectrum")              prod (filtered "spectrum")
   ────────────────────              ──────────────────────────
   ┌──┬──┬──┬──┬──┬──┐                ┌──┬──┬──┬──┬──┬──┐
   │ ▒│ ▒│ ▒│ ▒│ ▒│ ▒│                │ ▒│ ▒│  │  │  │  │
   └──┴──┴──┴──┴──┴──┘  ── × k_f ──►  └──┴──┴──┴──┴──┴──┘
    each bin = how                     k_f attenuates the
    much of that                       higher-frequency bins;
    frequency the                      result has less high-freq
    signal contains                    content (it's smoother in time)
```

In Hyena the filter `k` is learned per layer per channel, but the
underlying math is the same as the equalizer.

## Step-by-step with shape tracking

The HCM-relevant branch (`bidirectional=False`, `k_rev=None`) is the
only path that fires in production.

### Step 1: Set the FFT size

```python
seqlen = u.shape[-1]      # 8192
fft_size = 2 * seqlen     # 16384
```

**Purpose**: Padding factor to convert *circular* convolution (what FFT
naturally computes) into *linear* convolution (what we want). Linear
conv of length-L input with length-K filter needs
`fft_size ≥ L + K - 1`. Vortex uses `2L` which works whenever
`K ≤ L+1` — true for HCM (`K=128 ≤ 8193`).

```
What padding prevents (circular wraparound):

┌──── 8192 ────┐
[u_0 u_1 ... u_L] ─── circular wrap ─── back to u_0
                ↑
              filter tap at lag K would mix with u_0  (BUG)

With fft_size = 16384:
┌──── 8192 ────┐┌─── 8192 zeros ────┐
[u_0 ... u_L  ][   0    0   ...   0]    ← padded length 16384
                ↑
            filter wraps into the padding  (HARMLESS)
```

### Step 2: FFT the filter, pre-normalize

```python
k_f = torch.fft.rfft(k, n=fft_size) / fft_size
#         ┌────────┐    ┌──────────┐
#       (4096,1,128)  pad to 16384 then FFT
```

```
SHAPES
  input  k     ( 4096 , 1 , 128 )            fp32, real
  rfft(k, n=16384) zero-pads last dim to 16384 first, then FFT
  output k_f   ( 4096 , 1 , 8193 )           complex64

  └ rfft (real-input FFT) only stores the first N/2+1 = 8193 complex bins.
    The remaining bins are Hermitian-conjugate redundant (symmetry of FFT
    of real signal) — saves half the memory + half the compute vs full fft.
```

**Purpose**: Get the filter into frequency-domain so convolution becomes
a multiply (Convolution Theorem). The `/fft_size` is a normalization
trick: cuFFT's `rfft` is **unnormalized** (no 1/N factor); cuFFT's
`irfft` *by default* divides by N. By pre-dividing `k_f` here, and
passing `norm="forward"` to `irfft` later (which disables irfft's
auto-division), vortex moves the normalization off the hot inverse
path. Same math, different cost split.

### Step 3: Reshape filter for broadcasting

```python
k_f = adjust_filter_shape_for_broadcast(u, k_f)
```

The helper at `engine_ref.py:38-48`. For `u.shape == (B, D, L)` and
`k_f.shape == (D, 1, L+1)`:

```
  k_f  ( 4096 , 1 , 8193 )   complex64
   │
   │  .squeeze()  → strip the size-1 dim
   ▼
       ( 4096 , 8193 )       complex64
   │
   │  len(u.shape)=3 > len(h.shape)=2  → unsqueeze(0)
   ▼
   k_f  ( 1 , 4096 , 8193 )  complex64    ← now broadcasts against u_f
```

**Purpose**: PyTorch broadcasting needs aligned trailing dims and
explicit size-1 axes elsewhere. Making `k_f` `(1, D, L+1)` lets it
broadcast against `u_f` `(B, D, L+1)` over the batch dimension. **This
is shape-plumbing only; no math happens here.**

### Step 4: FFT the input

```python
u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
#                    └──────┬─────────┘    ┌──────────┐
#                  cast bf16→fp32 here   pad 8192 → 16384 then FFT
```

```
SHAPES
  u            ( 1 , 4096 , 8192 )     bf16 (incoming from autocast scope)
  u.to(fp32)   ( 1 , 4096 , 8192 )     fp32
  rfft(...)    ( 1 , 4096 , 8193 )     complex64
```

**Purpose**: Same as Step 2 but for the input. The explicit
`.to(dtype=k.dtype)` is the dtype up-cast that gives FFT-conv
numerical stability — Hyena filters are sensitive enough that bf16
input math hurts; vortex casts up to fp32 for the entire FFT-conv
pipeline and casts back at the very end.

### Step 5: Convolution theorem in action (one line, three things)

```python
y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
#                   └────┬─────┘   └────────┘  └──────────┘ └────┬────┘
#                  complex mul    inverse FFT  no /N here    trim to L
```

#### 5a: Complex elementwise multiply

```
u_f * k_f
─────────
u_f  ( 1 , 4096 , 8193 )  complex64
k_f  ( 1 , 4096 , 8193 )  complex64    (after step 3's broadcast prep)
─────
prod ( 1 , 4096 , 8193 )  complex64
```

**Purpose**: This is the convolution theorem. In time-domain you'd
compute `y[t] = Σ_τ k[τ] u[t-τ]`. In frequency-domain that becomes a
**pointwise multiply**. The whole reason to FFT was to convert the
expensive convolution into this cheap multiply.

#### 5b: Inverse FFT

```
irfft(prod, n=16384, norm="forward")
─────────────────────────────────────
prod    ( 1 , 4096 , 8193 )   complex64
irfft   ( 1 , 4096 , 16384 )  fp32      ← back to length 2L = 16384, real-valued
```

**Purpose**: Get back to time-domain. `norm="forward"` means "don't
divide by N on the inverse" — because step 2 already divided `k_f` by
N. (`norm="forward"` is one of cuFFT's normalization modes; the default
`"backward"` does `/N` on the inverse).

#### 5c: Trim the padded tail

```
irfft_out  ( 1 , 4096 , 16384 )   ← the padded length
[..., :seqlen]
─────────────
y          ( 1 , 4096 , 8192 )    ← drop the 8192 trailing positions
```

**Purpose**: Step 1's padding made the linear conv work, but the irfft
output is `fft_size=16384` long. Only the first `L=8192` positions are
valid linear-conv outputs; the rest are wrap-around artifacts we don't
want. Slicing keeps just the real conv.

```
The irfft output schematically:

┌───── L=8192 valid conv outputs ─────┐┌──── 8192 wrap artifacts ────┐
[ y[0]  y[1]  y[2]  ...        y[L-1] ][   junk   junk   ...    junk ]
                                       ↑
                              [..., :seqlen] keeps everything left of here
```

### Step 6: Skip-residual

```python
out = y + u * D.unsqueeze(-1)
#         └──┬──┘
#       (4096, 1) — broadcasts against (1, 4096, 8192) over batch & L
```

```
SHAPES
  D                 ( 4096 , )                       fp32
  D.unsqueeze(-1)   ( 4096 , 1 )                     fp32
  u * D[:, None]    ( 1 , 4096 , 8192 )              fp32   (broadcast B and L)
  y                 ( 1 , 4096 , 8192 )              fp32
  out               ( 1 , 4096 , 8192 )              fp32
```

**Purpose**: The Hyena operator wants `y_total = conv(u, k) + D ⊙ u`
where `D` is a learned per-channel skip-connection gain. In DSP terms
`D` represents `h(0)` — the *instantaneous* response that's stored
separately from the conv kernel so the kernel can be parameterized
cleanly (the modal-decomposition form has `Σ residue · exp(...)` which
goes to 0 at t=∞ but the skip term handles t=0). Without this term,
the filter would only see the past; with it, the current sample
contributes directly.

### Step 7: Cast back to bf16

```python
return out.to(dtype=u.dtype)
#                   ┌────┐
#                  bf16 (the original input dtype before step 4's up-cast)
```

```
SHAPES
  out          ( 1 , 4096 , 8192 )   fp32
  out.to(bf16) ( 1 , 4096 , 8192 )   bf16
```

**Purpose**: We're inside an `autocast("cuda")` scope at the call site,
which expects bf16 going forward. Steps 4–6 ran in fp32 for numerical
stability of the FFT math; step 7 commits back to inference dtype for
the rest of the model.

## Why the `bidirectional` and `k_rev` branches exist

The function has two branches HCM never enters but you'll see in the
code. Worth understanding what they do and why HCM skips them.

### The branch structure at a glance

```python
def fftconv_func(u, k, D, ..., bidirectional=False, k_rev=None, ...):
    ...
    k_f = rfft(k, n=fft_size) / fft_size
    k_f = adjust_filter_shape_for_broadcast(u, k_f)
    k = k.squeeze()

    if bidirectional:
        ╔══════════════════════════════════════════════╗
        ║ TWO-DIRECTION CONV — encoder-style models    ║
        ║ Apply forward + backward filters, sum.       ║
        ║ HCM never enters this branch.                ║
        ╚══════════════════════════════════════════════╝

    else:
        if k_rev is not None:
            ╔══════════════════════════════════════════╗
            ║ Add a reverse-filter component to k_f.   ║
            ║ HCM never enters this either.            ║
            ╚══════════════════════════════════════════╝

        u_f = rfft(u.to(dtype=k.dtype), n=fft_size)
        y = irfft(u_f * k_f, ...)[..., :seqlen]
        ╔══════════════════════════════════════════════╗
        ║ ◄── HCM's path: plain forward-only conv      ║
        ╚══════════════════════════════════════════════╝
```

### `if bidirectional:` — what it does

Applies the filter in **both forward and backward time directions** and
sums the results. From lines 70–77 of `engine_ref.py`:

```python
u_f = rfft(u.to(dtype=k.dtype), n=fft_size)
k, k2 = k.split(k.shape[1] // 2, dim=1)      # split filter in half
k2_f = rfft(k2, n=fft_size) / fft_size

y1 = u_f * k_f               # forward direction in freq domain
y2 = u_f.conj() * k2_f.conj()  # backward direction via conjugate trick

y = irfft(y1 + y2, n=fft_size, norm="forward")[..., :seqlen]
```

The **conjugate trick**: in the frequency domain, conjugating the FFT
is equivalent to **time-reversing** the original signal. So
`u_f.conj() * k2_f.conj()` computes "filter `k2` applied to `u` going
backwards in time, then conjugate that result".

When summed, `y` gets contributions from both temporal directions —
each output position sees both past AND future input.

**When this is used**: encoder-style models (BERT, T5) where masked
language modeling allows each token to attend to both past and future
context.

**Why HCM never uses it**: evo2 is autoregressive. If an early token
could see a later token through the conv, the model would leak future
information and break causality (you couldn't autoregressively sample
without re-running the whole forward). At the HCM call site
(`engine.py:182`), `bidirectional=False` is **hardcoded**.

### `if k_rev is not None:` — what it does

A subtler variant that lives inside the `else` (forward-only) branch.
Lines 67–69 of engine_ref.py:

```python
if k_rev is not None:
    k_rev_f = rfft(k_rev, n=fft_size) / fft_size
    k_f = k_f + k_rev_f.conj()
```

This adds a **reverse-filter component** to the forward filter `k_f`
*in the frequency domain*. Effectively a "partial bidirectional"
behavior with a single combined filter rather than two separate ones.
Some Hyena variants experimented with this; evo2's HCM doesn't.

At the HCM call site, `k_rev` argument isn't passed — defaults to
`None` — so the `k_f += k_rev_f.conj()` adjustment never fires.

### What the `else` branch actually routes to

For HCM (`bidirectional=False, k_rev=None`), the function effectively
collapses to just four lines:

```python
seqlen   = u.shape[-1]
fft_size = 2 * seqlen
k_f      = rfft(k, n=fft_size) / fft_size
k_f      = adjust_filter_shape_for_broadcast(u, k_f)
u_f      = rfft(u.to(dtype=k.dtype), n=fft_size)
y        = irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
out      = y + u * D.unsqueeze(-1)
return out.to(dtype=u.dtype)
```

That's the "plain causal forward-only depthwise FFT-conv with skip"
flow — what the step-by-step section walked through. Everything else
in `fftconv_func` is dead code for HCM.

### What this means for the wire-up

Our HCM kernel only needs to implement the **forward-only path** (the
`else` branch with `k_rev=None`). The bidirectional and k_rev paths can
stay as upstream `fftconv_func` — we won't intercept them. Our
wrapper falls back to upstream if anyone passes those args:

```python
def hcm_fft_conv(u, k, D, dropout_mask, *,
                 bidirectional=False, k_rev=None, **kwargs):
    """
    Drop-in replacement for fftconv_func — forward-only path only.
    bidirectional=True or k_rev is not None falls back to upstream.
    """
    if bidirectional or k_rev is not None:
        return _UPSTREAM_FFTCONV(u, k, D, dropout_mask,
                                  bidirectional=bidirectional,
                                  k_rev=k_rev, **kwargs)
    # ... our fused Triton path for the common case
```

This matches the maintainer email's "signature-preserving" constraint
without committing us to optimize paths nothing uses. Document the
fallback in the PR body so reviewers know what's covered vs deferred.

## End-to-end shape pipeline

```
u    (1, 4096, 8192) bf16
            │  cast to fp32
            ▼
     (1, 4096, 8192) fp32
            │  rfft, pad to 16384
            ▼
u_f  (1, 4096, 8193) complex64 ──┐
                                 │
k    (4096, 1, 128)  fp32        │
            │  rfft, pad to 16384│
            ▼                    │
k_f  (4096, 1, 8193) complex64   │
            │  /fft_size         │
            │  adjust shape      │
            ▼                    │
     (1, 4096, 8193) complex64 ──┤  ← broadcastable
                                 │
                          ┌──────┘
                          │  complex multiply
                          ▼
prod (1, 4096, 8193) complex64
            │  irfft (no /N), back to length 16384
            ▼
     (1, 4096, 16384) fp32
            │  [:, :, :8192] trim
            ▼
y    (1, 4096, 8192) fp32 ────┐
                              │  + u * D[:, None]
                              ▼
out  (1, 4096, 8192) fp32
            │  cast to bf16
            ▼
RETURN (1, 4096, 8192) bf16
```

## Per-step purpose, in one line each

| Step | Code | Purpose |
|---|---|---|
| 1 | `fft_size = 2 * seqlen` | Padding to prevent circular-conv wraparound |
| 2 | `k_f = rfft(k, n=fft_size) / fft_size` | Filter into frequency-domain + pre-normalize (moves /N off the inverse path) |
| 3 | `adjust_filter_shape_for_broadcast` | Shape plumbing so `k_f` broadcasts against `u_f` |
| 4 | `u_f = rfft(u.float(), n=fft_size)` | Input into frequency-domain (with explicit fp32 cast for stability) |
| 5a | `u_f * k_f` | Convolution theorem: time-conv → freq-multiply |
| 5b | `irfft(..., norm="forward")` | Back to time-domain (no /N since we pre-divided k_f) |
| 5c | `[..., :seqlen]` | Drop wraparound artifacts from the padded tail |
| 6 | `y + u * D.unsqueeze(-1)` | Add per-channel skip-residual (h(0) DSP term) |
| 7 | `.to(dtype=u.dtype)` | Commit back to inference dtype (bf16) |

## What changes for the wire-up

The HCM kernel (Phase 3) keeps the **3 cuFFT calls** (rfft × 2, irfft —
Triton can't beat cuFFT for these sizes) and fuses **everything else**:

| Step | Today | Fused |
|---|---|---|
| 2 `/fft_size` | separate elementwise launch | folded into a Triton kernel |
| 3 shape adjust | reshape op (~free) | implicit in the Triton load |
| 5a complex mul | elementwise launch | inside Triton kernel #1 |
| 6 skip-residual | 2× elementwise launches | inside Triton kernel #2 |
| 7 dtype cast | cast launch | folded into kernel #2's write |

The launch budget drops from **~7 ops to ~4 ops** (3 cuFFTs + 1–2
Triton epilogue kernels) per HCM layer. At 9 HCM layers × forward, that
saves ~27 launches per forward.

Two Triton kernel candidates to write (the exact split is a Phase 3
design call once we know whether CGCG@128 works — see
[`gcg_fwd_ref_corrected`](gcg_fwd_ref_corrected.md) target):

1. `mul_complex_bcast_kernel` — fuses step 2's `/fft_size` and step
   3's broadcast adjust into step 5a's complex multiply
2. `bias_residual_cast_kernel` — fuses step 6's skip-residual and
   step 7's dtype cast into the irfft's output write

## Where `fftconv_func` is used

**Defined** in exactly two places:

| Path | Line | Role |
|---|---|---|
| `vortex/model/engine.py` | 38 | Upstream definition |
| `vortex_kernels/reference/engine_ref.py` | 51 | Your vendored verbatim copy (correctness oracle) |

**Called** from exactly one place:

| Path | Line | Context |
|---|---|---|
| `vortex/model/engine.py` | **176** | Inside `parallel_fir`'s HCM branch (`elif fir_length >= 128`), under `torch.autocast("cuda")` |

That's it. Every HCM layer in evo2_7b goes through this one call site.

### ⚠️ Naming gotcha — `fftconv_func` vs `fftconv_fn`

The codebase has both. They're different things:

| Symbol | What it is | Used by | How it's invoked |
|---|---|---|---|
| **`fftconv_func`** | Free function on lines 38–83 (this walkthrough) | HCM only (`parallel_fir:176`) | Direct call |
| **`fftconv_fn`** | A **method parameter** / **attribute slot** | HCL only (`parallel_iir:325`) when `use_flashfft=True` | Gets assigned `FlashFFTConv` instance at `model.py:584` |

So if you see `fftconv_fn` references at `model.py:184, 308, 584` and
`engine.py:275, 325, 391, 536` — those are about FlashFFTConv (HCL fast
path), not the function you just read. The naming similarity is a
vortex code-quality issue, not a meaningful relationship.

## Cross-references

- [parallel_fir](parallel_fir.md) — Target 2 — the dispatch method
  that calls `fftconv_func` at line 176
- [adjust_filter_shape_for_broadcast](#) — Target intentionally not in
  the queue; the helper is short and self-explanatory once you know it's
  shape-plumbing
- [gcg_fwd_ref_corrected](gcg_fwd_ref_corrected.md) — Target 7 — the
  CGCG kernel's pure-PyTorch reference, which uses direct conv1d
  instead of FFT (different math path; relevant for the HCM Option A
  decision in Phase 3)

## Open questions / to-revisit later

- **Phase 3.0 decision**: does CGCG handle `filter_len=128` performantly
  vs `fftconv_func`? If yes, HCM is a wire-up (same as HCS); if no, the
  fusion described above is the work.
- **bidirectional path** (lines 70–77 in `engine_ref.py`): vortex
  supports `bidirectional=True` (two filters, one forward + one
  conj-flipped), but HCM never uses it. Probably worth noting in the
  PR that our kernel doesn't handle this — out-of-scope per maintainer
  email's "signature-preserving" constraint.

---

*Last updated: 2026-05-12 (Phase 1 read-through, target 1/12 — extended with inputs-explained, FFT primer, and bidirectional/k_rev sections)*
