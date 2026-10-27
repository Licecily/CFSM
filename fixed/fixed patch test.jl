using ApproxOperator
import ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
import ApproxOperator.Beam: ∫εᵘNᵘdΩ, ∫εᵘNᵠdΩ, ∫εᵠNᵘdΩ, ∫εᵠNᵠdΩ, ∫uᵢgᵢdΓ, ∫φᵢgᵢdΓ, L₂d
import Gmsh: gmsh
using LinearAlgebra
using Printf

# ===================== 1. 材料与截面参数（与 fixed beam.jl 一致）=====================
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

# ===================== 2. 分片试验的线性场（沿 X 的梁）=====================
# 分片试验的精确解必须落在有限元插值空间内。Seg2 单元每个节点 3 个位移
# + 3 个转角，位移与转角都线性插值，因此取如下线性场：
#
#   u₁ = ε₀ x,        u₂ = c₂ + θ_z x,   u₃ = c₃ − θ_y x
#   φ₁ = κ₁₀ x,       φ₂ = θ_y,          φ₃ = θ_z
# 其中所有系数取 1.0e-3。
#
# 广义应变：ε₁₁ = ε₀，κ₁ = κ₁₀，γ₂ = u₂,₁ − φ₃ = 0，γ₃ = u₃,₁ + φ₂ = 0，
#           κ₂ = κ₃ = 0，全部为常数。
#
# 与 sample.jl（Mindlin 板）一样，精确场写成 (x,y,z) 的函数，边界罚项直接
# 在积分点调用函数取值，而不是手工代入端点坐标。
#
# 关于分布荷载：sample.jl 由精确场反算 q、m₁、m₂ 后施加。本梁格式中
#   N = EA ε₁₁，Q₂ = k₃GA γ₂，Q₃ = k₂GA γ₃，M₂ = EIyy κ₂，M₃ = EIzz κ₃，
# 常应变 ⇒ N′ = 0、M₂、M₃ 为常数；无分布力偶时平衡要求 Q₂ = Q₃ = 0，
# 因此 b₁ = b₂ = b₃ ≡ 0，分片试验不需要外荷载项（只有端部反力）。
#
# 注：常曲率状态（φ₃ 线性、挠度 u₂ 二次）不在 Seg2 线性插值空间内
#     （Seg2 的 u₂、φ₃ 都只有线性形函数），不能作为严格分片试验的精确解；
#     弯曲刚度由 fixed beam.jl 的整梁收敛对比验证。
# 精确场只随 x 变化（φ₂、φ₃ 甚至与坐标无关），y、z 形参在函数体中用不到；
# 保留 (x,y,z) 是为了与 prescribe! 的调用方式 f(ξ.x, ξ.y, ξ.z) 对应。
# 系数统一取 1.0e-3：小变形量级；分片试验是线性问题，幅值只按比例缩放结果，
# 不影响相对误差。
u₁(x, y, z) = 1.0e-3*x
u₂(x, y, z) = 1.0e-3 + 1.0e-3*x
u₃(x, y, z) = 1.0e-3 - 1.0e-3*x
φ₁(x, y, z) = 1.0e-3*x
φ₂(x, y, z) = 1.0e-3
φ₃(x, y, z) = 1.0e-3

exact_values(x, y, z) = [u₁(x,y,z), u₂(x,y,z), u₃(x,y,z),
                  φ₁(x,y,z), φ₂(x,y,z), φ₃(x,y,z)]

# ===================== 3. 读入网格（沿 X 的梁）=====================
const integrationorder = 1
const msh = "msh/beamX.msh"

gmsh.initialize()
isfile(msh) || error("找不到网格文件：$msh")
gmsh.open(msh)

entities = getPhysicalGroups()
nodes    = get𝑿ᵢ()
nₚ       = length(nodes)
nᵤ       = 3nₚ

elements = getElements(nodes, entities["Ω"], integrationorder)
println("分片：节点 ", nₚ, " 个，梁单元 ", length(elements), " 个")

# ===================== 4. 装配刚度矩阵 =====================
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

# ===================== 5. 边界施加精确场（罚函数）=====================
# 与 sample.jl 的 ∫αwwdΓ、∫αφφdΓ 相同：直接传精确场函数，由算子在各积分点取值。
# α 取 1e15：分片试验只看复现精度，端部约束残差 ~ EA/(αL) ~ 1e-6。
const α = 1.0e15

elements_Γ¹ = getElements(nodes, entities["Γ¹"])
prescribe!(elements_Γ¹, :α=>α,
           :g₁=>u₁, :g₂=>u₂, :g₃=>u₃, :g₄=>φ₁, :g₅=>φ₂, :g₆=>φ₃)
set𝝭!(elements_Γ¹)
𝑏¹ᵘ = ∫uᵢgᵢdΓ => elements_Γ¹
𝑏¹ᵠ = ∫φᵢgᵢdΓ => elements_Γ¹

elements_Γ² = getElements(nodes, entities["Γ²"])
prescribe!(elements_Γ², :α=>α,
           :g₁=>u₁, :g₂=>u₂, :g₃=>u₃, :g₄=>φ₁, :g₅=>φ₂, :g₆=>φ₃)
set𝝭!(elements_Γ²)
𝑏²ᵘ = ∫uᵢgᵢdΓ => elements_Γ²
𝑏²ᵠ = ∫φᵢgᵢdΓ => elements_Γ²

# 无分布荷载，右端项只有罚项贡献 α N_I g_I
𝐊ᵘᵘα = zeros(3nₚ, 3nₚ)
𝐊ᵠᵠα = zeros(3nₚ, 3nₚ)
fᵘα = zeros(3nₚ)
fᵠα = zeros(3nₚ)
𝑏¹ᵘ(𝐊ᵘᵘα, fᵘα); 𝑏¹ᵠ(𝐊ᵠᵠα, fᵠα)
𝑏²ᵘ(𝐊ᵘᵘα, fᵘα); 𝑏²ᵠ(𝐊ᵠᵠα, fᵠα)
𝐊 = [𝐊ᵘᵘ + 𝐊ᵘᵘα  𝐊ᵘᵠ;
      𝐊ᵠᵘ          𝐊ᵠᵠ + 𝐊ᵠᵠα]
f = [fᵘα; fᵠα]
d = 𝐊 \ f

# 把有限元解写回节点，供 L₂d 读取（d₁..d₃ = u₁..u₃，d₄..d₆ = φ₁..φ₃）
push!(nodes,
      :d₁=>[d[3i-2] for i in 1:nₚ], :d₂=>[d[3i-1] for i in 1:nₚ], :d₃=>[d[3i] for i in 1:nₚ],
      :d₄=>[d[nᵤ+3i-2] for i in 1:nₚ], :d₅=>[d[nᵤ+3i-1] for i in 1:nₚ], :d₆=>[d[nᵤ+3i] for i in 1:nₚ])

# ===================== 6. 节点值检查 =====================
# 节点 i 的 6 个自由度：位移块 [3i-2, 3i]，转角块 [nᵤ+3i-2, nᵤ+3i]
𝐝(i)  = vcat(d[3i-2:3i], d[nᵤ+3i-2:nᵤ+3i])
𝐝ᵉ(i) = exact_values(nodes[i].x, nodes[i].y, nodes[i].z)

𝑥     = [nodes[i].x for i in 1:nₚ]
sorted_idx  = sortperm(𝑥)
end_nodes  = [sorted_idx[1], sorted_idx[end]]
interior_nodes  = setdiff(sorted_idx, end_nodes)

Δ = [𝐝(i) .- 𝐝ᵉ(i) for i in 1:nₚ]
field_amp = maximum(abs.(vcat([𝐝ᵉ(i) for i in 1:nₚ]...)))

println("\n=== 分片试验：沿 X 方向的梁 ===")
println("      x        max_c|Δd_c|   (c = u₁,u₂,u₃,φ₁,φ₂,φ₃)")
for k in sorted_idx
    marker = k ∈ end_nodes ? "  (端部，罚约束)" : ""
    @printf("%9.4f   %12.3e%s\n", 𝑥[k], maximum(abs.(Δ[k])), marker)
end

comp_names = ("u₁", "u₂", "u₃", "φ₁", "φ₂", "φ₃")
println("\n各分量最大绝对误差（内部节点）:")
for c in 1:6
    𝑒 = maximum([abs(Δ[i][c]) for i in interior_nodes])
    @printf("  %s : %12.3e\n", comp_names[c], 𝑒)
end

max_err = maximum([maximum(abs.(Δ[i])) for i in interior_nodes])
rel_err = max_err / field_amp
println("\n内部节点最大绝对误差 ", max_err)
println("相对误差（以场幅值 ", field_amp, " 为基准）", rel_err)

# ===================== 7. 域内 L₂ 误差（算子 L₂d 见 beam.jl）=====================
# 刚度装配用 1 点减缩积分；误差积分另取高阶 Gauss，避免积分误差掩盖真实误差
# （请求 10 阶，gmsh 对 Seg2 实际给出 6 个积分点）。
# L₂d 内部按 𝝭 重建有限元场 Σ N_I d_I，并读取 prescribe! 到积分点的精确场。
elements_L₂ = getElements(nodes, entities["Ω"], 10)
prescribe!(elements_L₂, :u₁=>u₁, :u₂=>u₂, :u₃=>u₃,
                         :φ₁=>φ₁, :φ₂=>φ₂, :φ₃=>φ₃)
set𝝭!(elements_L₂)

# Δ²[c] = ∫(dᶜ_FE − dᶜ_exact)²dΩ，范数²[c] = ∫(dᶜ_exact)²dΩ
err_sq, norm_sq = L₂d(elements_L₂)

L2_err = sqrt.(err_sq)
L2_norm = sqrt.(norm_sq)

println("\n域内 L₂ 误差（高阶 Gauss 积分）:")
for c in 1:6
    @printf("  %s : |e|_L₂ = %12.3e   相对 = %12.3e\n",
            comp_names[c], L2_err[c], L2_err[c]/L2_norm[c])
end

# 位移与转角分开统计（单位不同，不混合）
rel_L2_u = sqrt(sum(err_sq[1:3])) / sqrt(sum(norm_sq[1:3]))
rel_L2_φ = sqrt(sum(err_sq[4:6])) / sqrt(sum(norm_sq[4:6]))
@printf("\n相对 L₂ 误差：位移 %.3e，转角 %.3e\n", rel_L2_u, rel_L2_φ)

passed = rel_err < 1.0e-5 && rel_L2_u < 1.0e-5 && rel_L2_φ < 1.0e-5
println(passed ? "分片试验：通过" : "分片试验：未通过")

gmsh.finalize()
