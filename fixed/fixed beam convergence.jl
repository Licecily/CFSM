using ApproxOperator
import ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
import ApproxOperator.Beam: ∫εᵘNᵘdΩ, ∫εᵘNᵠdΩ, ∫εᵠNᵘdΩ, ∫εᵠNᵠdΩ, ∫dᵢgᵢdΓ, L₂d
import ApproxOperator.Elasticity: ∫vᵢbᵢdΩ
import Gmsh: gmsh
using LinearAlgebra
using Printf

# ===================== 1. 参数（与 fixed beam.jl 一致）=====================
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

const q = -1.0e3      # 整体 Y 向均布荷载（与 fixed beam.jl 相同）
const L = 3.0
const α = 1.0e15      # 罚参数；比 fixed beam.jl 的 1e13 大，压低收敛误差下限

# 单元数序列（网格逐级加密，h = L/n）
const n_elems = [5, 10, 20, 40]

# ===================== 2. 精确解（两端固定，均布荷载 q）=====================
# 见 fixed beam.jl 第 7 节：w 含弯曲 + 剪切两项，φ 为弯曲斜率
w_exact(x) = q*x^2*(L-x)^2/(24*E*Iyy) + q*x*(L-x)/(2*k₃*G*A)
φ_exact(x) = q*x*(L-x)*(L-2*x)/(12*E*Iyy)

# L₂d / prescribe! 需要的 6 个场函数（其余分量为 0）
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

    # 刚度（1 点减缩积分）
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

    # 两端固定（齐次罚函数）
    Γ¹ = getElements(nodes, entities["Γ¹"])
    prescribe!(Γ¹, :α=>α, :g₁=>0.0, :g₂=>0.0, :g₃=>0.0,
                          :g₄=>0.0, :g₅=>0.0, :g₆=>0.0)
    set𝝭!(Γ¹)
    𝑏¹ = ∫dᵢgᵢdΓ => Γ¹

    Γ² = getElements(nodes, entities["Γ²"])
    prescribe!(Γ², :α=>α, :g₁=>0.0, :g₂=>0.0, :g₃=>0.0,
                          :g₄=>0.0, :g₅=>0.0, :g₆=>0.0)
    set𝝭!(Γ²)
    𝑏² = ∫dᵢgᵢdΓ => Γ²

    𝐤ᵦ = copy(𝐤)
    f  = zeros(6nₚ)
    𝑏¹(𝐤ᵦ, f)
    𝑏²(𝐤ᵦ, f)

    # 均布荷载
    prescribe!(elements, :b₁=>0.0, :b₂=>q, :b₃=>0.0)
    𝑏q = ∫vᵢbᵢdΩ => elements
    𝑏q(f)

    d = 𝐤ᵦ \ f

    # 有限元解写回节点，供 L₂d 读取（d₁..d₃ = u₁..u₃，d₄..d₆ = φ₁..φ₃）
    push!(nodes,
          :d₁=>[d[3i-2] for i in 1:nₚ], :d₂=>[d[3i-1] for i in 1:nₚ], :d₃=>[d[3i] for i in 1:nₚ],
          :d₄=>[d[nᵤ+3i-2] for i in 1:nₚ], :d₅=>[d[nᵤ+3i-1] for i in 1:nₚ], :d₆=>[d[nᵤ+3i] for i in 1:nₚ])

    # 误差积分：独立的高阶 Gauss 单元
    err = getElements(nodes, entities["Ω"], 10)
    prescribe!(err, :u₁=>u₁ᵉ, :u₂=>u₂ᵉ, :u₃=>u₃ᵉ,
                    :φ₁=>φ₁ᵉ, :φ₂=>φ₂ᵉ, :φ₃=>φ₃ᵉ)
    set𝝭!(err)

    Δ², norm_sq = L₂d(err)
    Δu² = sum(Δ²[1:3]);  norm_u_sq = sum(norm_sq[1:3])
    Δφ² = sum(Δ²[4:6]);  norm_φ_sq = sum(norm_sq[4:6])

    abs_u = sqrt(Δu²)          # ‖u − u_ex‖_L2
    abs_φ = sqrt(Δφ²)          # ‖φ − φ_ex‖_L2
    rel_u = abs_u / sqrt(norm_u_sq)
    rel_φ = abs_φ / sqrt(norm_φ_sq)
    return abs_u, abs_φ, rel_u, rel_φ
end

# ===================== 5. 循环加密 =====================
# (n, h, abs_u, abs_φ, rel_u, rel_φ)
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

csv = joinpath(@__DIR__, "l2_convergence.csv")
open(csv, "w") do io
    println(io, "n,dof,dof_free,h,abs_u,abs_φ,rel_u,rel_φ,rate_u,rate_φ")
    for (i, (n, h, gu, gp, eu, ep)) in enumerate(results)
        @printf(io, "%d,%d,%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.6f,%.6f\n",
                n, 6*(n+1), 6*(n-1), h, gu, gp, eu, ep,
                rates[1,i], rates[2,i])
    end
end
println("\n数据已写入: ", csv)
