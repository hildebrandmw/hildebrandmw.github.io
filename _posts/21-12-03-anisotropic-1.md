---
title: Implementing Anisotropic Product Quantization (Part 1) (WIP)
categories:
  - Blog
---

* TOC
{:toc}

# Deconstructing the Loss Function

## Establishing a Baseline

Skipping over many, many details, the loss function we're trying to optimize looks like
```julia
function anisotropic_loss_ref(x::StaticVector{N}, x̄::StaticVector{N}, η::Float32) where {N}
    # Find the norm of `x` and the error of the current estimate `x̄`.
    norm = norm_square(x)
    error = x - xbar
    # Parallel error is the projection of our error estimate in the direction of `x`.
    error₌ = innerproduct(error, x) * x / norm
    # Perpendicular error is the vector difference of the between the full error and the
    # parallel error.
    error₊ = error - error₌
    # Build up the over loss as a weighted sum of parallel and perpendicular loss.
    loss₌ = norm_square(error₌)
    loss₊ = norm_square(error₊)
    return η * loss₌ + loss₊
end
```
There's a lot going on here.
Lets first establish some baseline performance:
```julia
julia> using PQ, StaticArrays, BenchmarkTools

julia> x = @SVector randn(Float32, 100);

julia> x̄ = @SVector randn(Float32, 100);

julia> @benchmark PQ.anisotropic_loss_ref($x, $x̄, 1.0f0)
BenchmarkTools.Trial: 10000 samples with 983 evaluations.
 Range (min … max):  57.736 ns … 94.601 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     59.153 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):   58.686 ns ±  0.975 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

   ▅                        █▄
  ▅█▇▆▂▁▁▁▁▁▁▁▁▁▁▄▅▂▂▁▂▂▂▂▂▇██▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▁▂▂ ▃
  57.7 ns         Histogram: frequency by time          61 ns <

julia> @code_native syntax=:intel PQ.anisotropic_loss_ref(x, x̄, 1.0f0)
```
(not shown, the assembly code for this breaks down into a completely unrolled kernel with no branches or loops).
Okay, 59ns per evaluation is not too bad, but lets do some math.
With our 100 dimensional data, assume PQ with 25 partitions, 256 centroids per partition, and assume that we need on average ten passes of coordinate descent to converge, than we're looking at

$$
\begin{equation}
    25 \cdot 256 \cdot 10 = 64,000
\end{equation}
$$

evaluations per data point.
If our dataset contains 1 billion points, than this is $$6.4 \cdot 10^{13}$$ distance computations.
With a single thread, this will take

$$
\begin{equation}
    (1.28 \cdot 10^{13})(59\cdot 10^{-9}) = 3,776,000\,\text{seconds}
\end{equation}
$$

In other words, slightly over 43 days.
If we have over 100 threads working together, than we're talking on the order of 10 hours or so.
Now, this isn't the WORST thing in the world, but we can do way better.

## Take Two

The biggest problem with the reference implementation is that it does a LOT of unnecessary work.
Not only is `norm` recomputed every single time (despite being constant), but we're also performing a lot of relatively expensive SIMD computations (like the computation of `error`, all invocations of `innerproduct` or `norm_square`, etc.).
So, our goal should be to precompute as much as possible and assemble precomputed pieces together as simply as possible.
As discussed before, we can't just linearly combine partial portions of the loss together because the anisotropic loss function isn't decomposable in that way.
However, we can still do much better than the reference.

### Simplifying Parallel Loss

To understand how to decompose the anisotropic loss computation, we need to do some light linear algebra.
Lets begin with the parallel loss, which mathematically looks like

$$
\begin{align}
    N                   &= \| x \| ^ 2 \\
    e                   &= x - \overline{x}                                         && \text{Overall error}\\
    e_{\parallel}       &= \underbrace{\langle e, x \rangle}_{= \alpha} \, x /  N   && \text{Parallel Error} \\
    l_{\parallel}       &= \| e_{\parallel} \| ^ 2                                  && \text{Parallel Loss}\\
                        &= \langle{e_{\parallel}, e_{\parallel}} \rangle
\end{align}
$$

Note that we define

$$
\begin{align}
    \alpha  &= \langle e, x \rangle                                     \\
            &= \langle x - \overline{x}, x \rangle                      \\
            &= \langle x, x \rangle - \langle x, \overline{x} \rangle   \\
            &= N - \langle x, \overline{x} \rangle
\end{align}
$$

because this quantity will become useful later.
Now that we've defined $$l_{\parallel}$$ forwards, lets work back to a full expression for the parallel loss to see how it can be simplified.

$$
\begin{align}
    l_{\parallel}   &= \langle e_{\parallel}, e_{\parallel} \rangle                                    \\
                    &= \langle (\alpha / N) x, (\alpha / N) x \rangle                                  \\
                    &= \frac{\alpha^2}{N^2} \langle x, x \rangle       && \text{Pull out Constants}    \\
                    &= \frac{\alpha^2}{N}                              && \text{since } \langle x, x \rangle = N
\end{align}
$$

Now, back substituting $$\alpha$$

$$
\begin{align}
    l_{\parallel}   &= \frac{\alpha^2}{N}                                                   \\
                    &= \boxed{ \frac{(N - \langle x, \overline{x} \rangle) ^ 2}{N} }
\end{align}
$$

Awesome!
This actually gives us a pretty simple equation for the parallel loss.
Even better, the dot product quantity $$\langle x, \overline{x} \rangle$$ can be constructed from a bunch of little sums (i.e., from a residual table)!

### Simplifying Perpendicular Loss

Now we take a look at the perpendicular loss, which is slightly (but not overly so) more complicated than the parallel case.
Again, we'll work our way backwards from the definition of perpendicular loss to some (hopefully simple) expression.

$$
\begin{align}
    l_{\perp}   &= \langle e - e_{\parallel}, e - e_{\parallel} \rangle         \\
                &= \left\langle (x - \overline{x} - \frac{\alpha}{N} x, x - \overline{x} - \frac{\alpha}{N} x \right\rangle \\
                &= \left\langle \underbrace{\left( 1 - \frac{\alpha}{N} \right)}_{=v} x - \overline{x}, \left( 1 - \frac{\alpha}{N} \right) x - \overline{x} \right\rangle \\
                &= \langle vx, vx \rangle - 2 \langle vx, \overline{x} \rangle + \langle \overline{x}, \overline{x} \rangle \\
                &= \boxed{ v^2N - 2v \langle x, \overline{x} \rangle + \langle \overline{x}, \overline{x} \rangle }
\end{align}
$$

Well, this is not as pretty as the parallel loss, but still not bad at all.
Plus, we now have expressions for parallel and perpendicular loss that depend just on $$N = \| x \|$$ (which is constant), $$\langle x, \overline{x} \rangle$$, and $$\langle \overline{x}, \overline{x} \rangle$$.

We define $$\langle x, \overline{x} \rangle$$ as π and $$\langle \overline{x}, \overline{x} \rangle$$ as β.
This gives us the following function as an implementation for the anisotropic loss:
```julia
function anisotropic_loss_from_residuals(η::T, N::T, π::T, β::T) where {T}
    # Compute parallel loss in pieces in order to reuse those pieces
    # for computation of the perpendicular loss
    α = N - β
    u = α / N
    l₌ = u * α
    # Now, compute the perpendicular loss
    v = (one(T) - u)
    l₊ = (v * v * N) - (2 * v * β) + π
    # Finally, combine into the final loss
    return η * l₌ + l₊
end
```

## To the Moon

So far, we've managed to get a nearly 20x speedup over the original product quantization optimization, but we're not quite done yet.
The work we've done so far has computed the SIMD heavy loss computation into basically a scalar function.
So lets bring back SIMD!
In particular, we can evaluate `anisotropic_loss_from_residuals` for multiple values of π and β at a time by slightly tweaking the function's implementation:
```julia
const ResidualTypes{T} = Union{T, StaticVector{<:Any,T}}
@inline function anisotropic_loss_with_residuals(
    η::T,
    N::T,
    # residuals
    π::ResidualTypes{T},
    β::ResidualTypes{T},
) where {T}
    α = N .- β
    u = α / N
    l₌ = u .* α
    v = (one(Float32) .- u)
    l₊ = @. (v * v * N) - (2 * v * β) + π
    return η * l₌ + l₊
end
```
We move the arithmetic functions to broadcasting operators.
Thus, scalar computation will perform just the same (though perhaps be a little harder on the compiler, but after all that's why it is there).
But, now we can pass vectors (read `SVectors`) for π and β, evaluating many losses in parallel thanks to auto vectorization.
Indeed, we can inspect the resulting assembly code in the scalar versus broadcasted cases
```
# Scalar case
julia> code_native(PQ.anisotropic_loss_with_residuals, Tuple{Float32,Float32,Float32,Float32}; syntax=:intel, debuginfo = :none)
	.text
	vsubss	xmm4, xmm1, xmm3
	vdivss	xmm5, xmm4, xmm1
	vmulss	xmm4, xmm4, xmm5
	movabs	rax, offset .rodata.cst4
	vmovss	xmm6, dword ptr [rax]           # xmm6 = mem[0],zero,zero,zero
	vsubss	xmm5, xmm6, xmm5
	vmulss	xmm6, xmm5, xmm5
	vmulss	xmm1, xmm6, xmm1
	vaddss	xmm5, xmm5, xmm5
	vmulss	xmm3, xmm5, xmm3
	vsubss	xmm1, xmm1, xmm3
	vaddss	xmm1, xmm1, xmm2
	vmulss	xmm0, xmm4, xmm0
	vaddss	xmm0, xmm0, xmm1
	ret
	nop

# Vectorized Case
julia> code_native(PQ.anisotropic_loss_with_residuals, Tuple{Float32,Float32,SVector{16,Float32},SVector{16,Float32}}; syntax=:intel, debuginfo = :none)
	.text
	mov	rax, rdi
	vbroadcastss	ymm1, xmm1
	vmovups	ymm2, ymmword ptr [rdx]
	vmovups	ymm3, ymmword ptr [rdx + 32]
	vsubps	ymm4, ymm1, ymm2
	vsubps	ymm5, ymm1, ymm3
	vdivps	ymm6, ymm4, ymm1
	vdivps	ymm7, ymm5, ymm1
	vmulps	ymm4, ymm4, ymm6
	vmulps	ymm5, ymm5, ymm7
	movabs	rcx, offset .rodata.cst4
	vbroadcastss	ymm8, dword ptr [rcx]
	vsubps	ymm6, ymm8, ymm6
	vsubps	ymm7, ymm8, ymm7
	vaddps	ymm8, ymm6, ymm6
	vmulps	ymm2, ymm8, ymm2
	vmulps	ymm6, ymm6, ymm6
	vmulps	ymm6, ymm6, ymm1
	vsubps	ymm2, ymm6, ymm2
	vaddps	ymm2, ymm2, ymmword ptr [rsi]
	vaddps	ymm6, ymm7, ymm7
	vmulps	ymm3, ymm6, ymm3
	vmulps	ymm6, ymm7, ymm7
	vmulps	ymm1, ymm6, ymm1
	vsubps	ymm1, ymm1, ymm3
	vaddps	ymm1, ymm1, ymmword ptr [rsi + 32]
	vbroadcastss	ymm0, xmm0
	vmulps	ymm3, ymm4, ymm0
	vaddps	ymm2, ymm3, ymm2
	vmulps	ymm0, ymm5, ymm0
	vaddps	ymm0, ymm0, ymm1
	vmovups	ymmword ptr [rdi], ymm2
	vmovups	ymmword ptr [rdi + 32], ymm0
	vzeroupper
	ret
	nop	word ptr cs:[rax + rax]
```
