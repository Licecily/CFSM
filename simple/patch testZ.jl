using ApproxOperator
import ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
import ApproxOperator.Beam: ∫εᵘNᵘdΩ, ∫εᵘNᵠdΩ, ∫εᵠNᵘdΩ, ∫εᵠNᵠdΩ, ∫dᵢgᵢdΓ, L₂d
import Gmsh: gmsh
using LinearAlgebra
using Printf

# ===================== 1. 材料与截面参数（与 simple beam.jl 一致）=====================
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

# ===================== 2. 分片试验的线性场（沿 Z 的梁）=====================
# 本文件是简支梁算例的单元分片试验。注意：分片试验要求在整个边界上施加
# 精确场；一维梁的边界就是两个端点，所以 Γ¹、Γ² 都要施加精确值，这与
# 简支梁实际的铰支/滚动支座条件无关。简支梁边值问题本身
# （均布荷载、跨中集中力）由 simple beam.jl 的整梁对比验证。
#
# 沿 Z 的梁在 beam.jl 的 𝐑 中走特殊分支（eₓ ∥ 整体 Z，s = 0）：
# e_y = X，e_z = eₓ × e_y = Z × X = Y。
# 局部分量与整体分量的对应关系：
#   u₁ˡ = u₃,  u₂ˡ = u₁,  u₃ˡ = u₂
#   φ₁ˡ = φ₃,  φ₂ˡ = φ₁,  φ₃ˡ = φ₂
#
# 局部线性场（s 为沿梁轴的局部坐标，这里 s = z）：
#   u₁ˡ = ε₀ z,       u₂ˡ = c₂ + θ_z z,  u₃ˡ = c₃ − θ_y z
#   φ₁ˡ = κ₁₀ z,      φ₂ˡ = θ_y,         φ₃ˡ = θ_z
#
# 由 dᵍ = Rᵀ dˡ 转换到整体分量，得到本文件使用的线性场：
#   u₁ = c₂ + θ_z z,  u₂ = c₃ − θ_y z,   u₃ = ε₀ z
#   φ₁ = θ_y,         φ₂ = θ_z,          φ₃ = κ₁₀ z
#
# 广义应变全为常数：ε₁₁ = ε₀，κ₁ = κ₁₀，γ₂ = u₁,₁ − φ₂ = 0，
# γ₃ = u₂,₁ + φ₁ = 0，κ₂ = κ₃ = 0；因此 b₁ = b₂ = b₃ ≡ 0，
# 分片试验不需要外荷载项（只有端部反力）。
#
# 注：常曲率状态（φ 线性、挠度二次）不在 Seg2 线性插值空间内，
#     不能作为严格分片试验的精确解；弯曲刚度由 simple beam.jl 验证。
# 系数统一取 1.0e-3（小变形量级；线性问题，幅值只按比例缩放结果）。
u₁(x, y, z) = 1.0e-3 + 1.0e-3*z
u₂(x, y, z) = 1.0e-3 - 1.0e-3*z
u₃(x, y, z) = 1.0e-3*z
φ₁(x, y, z) = 1.0e-3
φ₂(x, y, z) = 1.0e-3
φ₃(x, y, z) = 1.0e-3*z

精确值(x, y, z) = [u₁(x,y,z), u₂(x,y,z), u₃(x,y,z),
                  φ₁(x,y,z), φ₂(x,y,z), φ₃(x,y,z)]

# ===================== 3. 读入网格（沿 Z 的梁）=====================
const integrationorder = 1
const msh = "msh/beamZ.msh"

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
𝐤 = [𝐊ᵘᵘ 𝐊ᵘᵠ; 𝐊ᵠᵘ 𝐊ᵠᵠ]

# ===================== 5. 边界施加精确场（罚函数）=====================
# 与 sample.jl 的 ∫αwwdΓ、∫αφφdΓ 相同：直接传精确场函数，由算子在各积分点取值。
# α 取 1e15：分片试验只看复现精度，端部约束残差 ~ EA/(αL) ~ 1e-6。
const α = 1.0e15

elements_Γ¹ = getElements(nodes, entities["Γ¹"])
prescribe!(elements_Γ¹, :α=>α,
           :g₁=>u₁, :g₂=>u₂, :g₃=>u₃, :g₄=>φ₁, :g₅=>φ₂, :g₆=>φ₃)
set𝝭!(elements_Γ¹)
𝑏¹ = ∫dᵢgᵢdΓ => elements_Γ¹

elements_Γ² = getElements(nodes, entities["Γ²"])
prescribe!(elements_Γ², :α=>α,
           :g₁=>u₁, :g₂=>u₂, :g₃=>u₃, :g₄=>φ₁, :g₅=>φ₂, :g₆=>φ₃)
set𝝭!(elements_Γ²)
𝑏² = ∫dᵢgᵢdΓ => elements_Γ²

# 无分布荷载，右端项只有罚项贡献 α N_I g_I
𝐤ᵦ = copy(𝐤)
f  = zeros(6nₚ)
𝑏¹(𝐤ᵦ, f)
𝑏²(𝐤ᵦ, f)
d = 𝐤ᵦ \ f

# 把有限元解写回节点，供 L₂d 读取（d₁..d₃ = u₁..u₃，d₄..d₆ = φ₁..φ₃）
push!(nodes,
      :d₁=>[d[3i-2] for i in 1:nₚ], :d₂=>[d[3i-1] for i in 1:nₚ], :d₃=>[d[3i] for i in 1:nₚ],
      :d₄=>[d[nᵤ+3i-2] for i in 1:nₚ], :d₅=>[d[nᵤ+3i-1] for i in 1:nₚ], :d₆=>[d[nᵤ+3i] for i in 1:nₚ])

# ===================== 6. 节点值检查 =====================
# 节点 i 的 6 个自由度：位移块 [3i-2, 3i]，转角块 [nᵤ+3i-2, nᵤ+3i]
𝐝(i)  = vcat(d[3i-2:3i], d[nᵤ+3i-2:nᵤ+3i])
𝐝ᵉ(i) = 精确值(nodes[i].x, nodes[i].y, nodes[i].z)

𝑧     = [nodes[i].z for i in 1:nₚ]
次序  = sortperm(𝑧)
端点  = [次序[1], 次序[end]]
内部  = setdiff(次序, 端点)

Δ = [𝐝(i) .- 𝐝ᵉ(i) for i in 1:nₚ]
场幅值 = maximum(abs.(vcat([𝐝ᵉ(i) for i in 1:nₚ]...)))

println("\n=== 分片试验：简支梁算例（沿 Z 方向）===")
println("      z        max_c|Δd_c|   (c = u₁,u₂,u₃,φ₁,φ₂,φ₃)")
for k in 次序
    标记 = k ∈ 端点 ? "  (端部，罚约束)" : ""
    @printf("%9.4f   %12.3e%s\n", 𝑧[k], maximum(abs.(Δ[k])), 标记)
end

分量名 = ("u₁", "u₂", "u₃", "φ₁", "φ₂", "φ₃")
println("\n各分量最大绝对误差（内部节点）:")
for c in 1:6
    𝑒 = maximum([abs(Δ[i][c]) for i in 内部])
    @printf("  %s : %12.3e\n", 分量名[c], 𝑒)
end

最大误差 = maximum([maximum(abs.(Δ[i])) for i in 内部])
相对误差 = 最大误差 / 场幅值
println("\n内部节点最大绝对误差 ", 最大误差)
println("相对误差（以场幅值 ", 场幅值, " 为基准）", 相对误差)

# ===================== 7. 域内 L₂ 误差（算子 L₂d 见 beam.jl）=====================
# 刚度装配用 1 点减缩积分；误差积分另取高阶 Gauss，避免积分误差掩盖真实误差
# （请求 10 阶，gmsh 对 Seg2 实际给出 6 个积分点）。
# L₂d 内部按 𝝭 重建有限元场 Σ N_I d_I，并读取 prescribe! 到积分点的精确场。
elements_L₂ = getElements(nodes, entities["Ω"], 10)
prescribe!(elements_L₂, :u₁=>u₁, :u₂=>u₂, :u₃=>u₃,
                         :φ₁=>φ₁, :φ₂=>φ₂, :φ₃=>φ₃)
set𝝭!(elements_L₂)

# Δ²[c] = ∫(dᶜ_FE − dᶜ_exact)²dΩ，范数²[c] = ∫(dᶜ_exact)²dΩ
误差², 范数² = L₂d(elements_L₂)

L₂误差 = sqrt.(误差²)
L₂范数 = sqrt.(范数²)

println("\n域内 L₂ 误差（高阶 Gauss 积分）:")
for c in 1:6
    @printf("  %s : |e|_L₂ = %12.3e   相对 = %12.3e\n",
            分量名[c], L₂误差[c], L₂误差[c]/L₂范数[c])
end

# 位移与转角分开统计（单位不同，不混合）
相对L₂位移 = sqrt(sum(误差²[1:3])) / sqrt(sum(范数²[1:3]))
相对L₂转角 = sqrt(sum(误差²[4:6])) / sqrt(sum(范数²[4:6]))
@printf("\n相对 L₂ 误差：位移 %.3e，转角 %.3e\n", 相对L₂位移, 相对L₂转角)

通过 = 相对误差 < 1.0e-5 && 相对L₂位移 < 1.0e-5 && 相对L₂转角 < 1.0e-5
println(通过 ? "分片试验：通过" : "分片试验：未通过")

gmsh.finalize()
