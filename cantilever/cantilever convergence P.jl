using ApproxOperator
import ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
import ApproxOperator.Beam: ∫εᵘNᵘdΩ, ∫εᵘNᵠdΩ, ∫εᵠNᵘdΩ, ∫εᵠNᵠdΩ, ∫dᵢgᵢdΓ, L₂d
import ApproxOperator.Elasticity: ∫vᵢtᵢdΓ
import Gmsh: gmsh
using LinearAlgebra
using Printf

# ===================== 1. 参数（与 cantilever.jl 一致）=====================
const E   = 2.1e11
const G   = 8.1e10
const A   = 1.0e-2
const Iyy = 4.0e-4
const Izz = 6.0e-4
const Iy  = 0.0
const Iz  = 0.0
const Iyz = 0.0
const k₂  = 5/6
const k₃  = 5/6

const P = -1.0e3      # 自由端整体 Y 向集中力
const L = 3.0
const α = 1.0e15

const n_elems = [5, 10, 20, 40]

# ===================== 2. 精确解（悬臂梁，自由端集中力 P）=====================
# w_b = P x²(3L−x)/(6EIyy)，w_s = P x/(k₃GA)
# φ   = w_b′ = P x(2L−x)/(2EIyy)
w_exact(x) = P*x^2*(3L - x)/(6*E*Iyy) + P*x/(k₃*G*A)
φ_exact(x) = P*x*(2L - x)/(2*E*Iyy)

u₁ᵉ(x, y, z) = 0.0
u₂ᵉ(x, y, z) = w_exact(x)
u₃ᵉ(x, y, z) = 0.0
φ₁ᵉ(x, y, z) = 0.0
φ₂ᵉ(x, y, z) = 0.0
φ₃ᵉ(x, y, z) = φ_exact(x)

# ===================== 3. 生成 n 个单元的梁网格 =====================
function build_mesh(n)
    gmsh.model.add("beam_n$n")
    p₁ = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
    p₂ = gmsh.model.geo.addPoint(L,   0.0, 0.0)
    𝑙  = gmsh.model.geo.addLine(p₁, p₂)
    gmsh.model.geo.synchronize()
    gΩ = gmsh.model.addPhysicalGroup(1, [𝑙]);  gmsh.model.setPhysicalName(1, gΩ, "Ω")
    g¹ = gmsh.model.addPhysicalGroup(0, [p₁]); gmsh.model.setPhysicalName(0, g¹, "Γ¹")
    g² = gmsh.model.addPhysicalGroup(0, [p₂]); gmsh.model.setPhysicalName(0, g², "Γ²")
    gmsh.model.mesh.setTransfiniteCurve(𝑙, n+1)
    gmsh.model.mesh.generate(1)
end

# ===================== 4. 单个网格：求解并计算相对 L₂ 误差 =====================
function solve_mesh(n)
    entities = getPhysicalGroups()
    nodes    = get𝑿ᵢ()
    nₚ       = length(nodes)
    nᵤ       = 3nₚ

    elements = getElements(nodes, entities["Ω"], 1)
    prescribe!(elements, :E=>E, :G=>G, :A=>A, :Iy=>Iy, :Iz=>Iz,
                         :Iyy=>Iyy, :Izz=>Izz, :Iyz=>Iyz, :k₂=>k₂, :k₃=>k₃)
    set∇𝝭!(elements)

    𝐊ᵘᵘ = zeros(3nₚ, 3nₚ)
    𝐊ᵘᵠ = zeros(3nₚ, 3nₚ)
    𝐊ᵠᵘ = zeros(3nₚ, 3nₚ)
    𝐊ᵠᵠ = zeros(3nₚ, 3nₚ)
    𝑎₁ = ∫εᵘNᵘdΩ => elements
    𝑎₂ = ∫εᵘNᵠdΩ => elements
    𝑎₃ = ∫εᵠNᵘdΩ => elements
    𝑎₄ = ∫εᵠNᵠdΩ => elements
    𝑎₁(𝐊ᵘᵘ)
    𝑎₂(𝐊ᵘᵠ)
    𝑎₃(𝐊ᵠᵘ)
    𝑎₄(𝐊ᵠᵠ)
    𝐤 = [𝐊ᵘᵘ 𝐊ᵘᵠ; 𝐊ᵠᵘ 𝐊ᵠᵠ]

    # 仅固定端 Γ¹ 约束；Γ² 自由
    Γ¹ = getElements(nodes, entities["Γ¹"])
    prescribe!(Γ¹, :α=>α, :g₁=>0.0, :g₂=>0.0, :g₃=>0.0,
                          :g₄=>0.0, :g₅=>0.0, :g₆=>0.0)
    set𝝭!(Γ¹)
    𝑏¹ = ∫dᵢgᵢdΓ => Γ¹

    𝐤ᵦ = copy(𝐤)
    f  = zeros(6nₚ)
    𝑏¹(𝐤ᵦ, f)

    # 自由端集中力 P（整体 Y 向）
    Γ² = getElements(nodes, entities["Γ²"])
    set𝝭!(Γ²)
    prescribe!(Γ², :t₁=>0.0, :t₂=>P, :t₃=>0.0)
    𝑏P = ∫vᵢtᵢdΓ => Γ²
    𝑏P(f)

    d = 𝐤ᵦ \ f

    push!(nodes,
          :d₁=>[d[3i-2] for i in 1:nₚ], :d₂=>[d[3i-1] for i in 1:nₚ], :d₃=>[d[3i] for i in 1:nₚ],
          :d₄=>[d[nᵤ+3i-2] for i in 1:nₚ], :d₅=>[d[nᵤ+3i-1] for i in 1:nₚ], :d₆=>[d[nᵤ+3i] for i in 1:nₚ])

    err = getElements(nodes, entities["Ω"], 10)
    prescribe!(err, :u₁=>u₁ᵉ, :u₂=>u₂ᵉ, :u₃=>u₃ᵉ,
                    :φ₁=>φ₁ᵉ, :φ₂=>φ₂ᵉ, :φ₃=>φ₃ᵉ)
    set𝝭!(err)

    Δ², norm_sq = L₂d(err)
    Δu² = sum(Δ²[1:3]);  norm_u_sq = sum(norm_sq[1:3])
    Δφ² = sum(Δ²[4:6]);  norm_φ_sq = sum(norm_sq[4:6])

    abs_u = sqrt(Δu²)
    abs_φ = sqrt(Δφ²)
    rel_u = abs_u / sqrt(norm_u_sq)
    rel_φ = abs_φ / sqrt(norm_φ_sq)
    return abs_u, abs_φ, rel_u, rel_φ
end

# ===================== 5. 循环加密 =====================
results = Tuple{Int,Float64,Float64,Float64,Float64,Float64}[]

gmsh.initialize()
for (k, n) in enumerate(n_elems)
    k > 1 && gmsh.clear()
    build_mesh(n)
    gu, gp, eu, ep = solve_mesh(n)
    push!(results, (n, L/n, gu, gp, eu, ep))
    @printf("n = %4d   h = %.6e   |e_u| = %.6e (相对 %.6e)   |e_φ| = %.6e (相对 %.6e)\n",
            n, L/n, gu, eu, gp, ep)
end
gmsh.finalize()

# ===================== 6. 收敛阶与 CSV =====================
println("\n收敛阶 (p = log(e_{k-1}/e_k) / log(h_{k-1}/h_k)):")
rates = zeros(2, length(results))

for i in 2:length(results)
    n₀, h₀, gu₀, gp₀, u₀, p₀ = results[i-1]
    n₁, h₁, gu₁, gp₁, u₁, p₁ = results[i]
    pu = log(u₀/u₁) / log(h₀/h₁)
    pp = log(p₀/p₁) / log(h₀/h₁)
    rates[1,i] = pu
    rates[2,i] = pp
    @printf("n = %4d → %4d :  p_u = %.3f   p_φ = %.3f\n", n₀, n₁, pu, pp)
end

csv = joinpath(@__DIR__, "l2_convergence_cantilever_P.csv")
open(csv, "w") do io
    println(io, "n,dof,dof_free,h,abs_u,abs_φ,rel_u,rel_φ,rate_u,rate_φ")
    for (i, (n, h, gu, gp, eu, ep)) in enumerate(results)
        @printf(io, "%d,%d,%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.6f,%.6f\n",
                n, 6*(n+1), 6*n, h, gu, gp, eu, ep,
                rates[1,i], rates[2,i])
    end
end
println("\n数据已写入: ", csv)
