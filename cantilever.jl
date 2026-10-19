
using ApproxOperator
import ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
import ApproxOperator.Beam: ∫δdᵘKᵍᵘᵘdᵘds, ∫δdᵘKᵍᵘᵠdᵠds, ∫δdᵠKᵍᵠᵠdᵠds, ∫δdᵢgᵢdΓ
import ApproxOperator.Elasticity: ∫vᵢbᵢdΩ, ∫vᵢtᵢdΓ
import Gmsh: gmsh
using LinearAlgebra

# ===================== 1. 材料与截面参数 =====================
const E   = 2.1e11        # 弹性模量
const G   = 8.1e10        # 剪切模量（各向同性时取 G = E/(2(1+ν))）
const A   = 1.0e-2        # 截面面积
const Iyy = 4.0e-4        # 绕局部 y 轴的惯性矩
const Izz = 6.0e-4        # 绕局部 z 轴的惯性矩
const Iy  = 0.0           # 一阶矩（形心主轴 ⇒ 0）
const Iz  = 0.0
const Iyz = 0.0           # 惯性积（主轴 ⇒ 0）
const k₂  = 5/6           # 剪切修正系数（局部 y 方向）
const k₃  = 5/6           # 剪切修正系数（局部 z 方向）

# 荷载（整体 Y 向，负号朝下）
const q = -1.0e3          # 均布线荷载 N/m
const P = -1.0e3          # 自由端集中力 N

# 3D.md 7.2 的局部坐标系约定（参考向量取整体 Z 轴）下，沿整体 X 的梁为
#   局部 y = 整体 Z、局部 z = −整体 Y
# 所以整体 Y 向荷载由 EIyy 与 k₃ 控制（整体 Z 向荷载才是 EIzz、k₂）。

# ===================== 2. 读入 / 生成网格 =====================
const integrationorder = 1        # Seg2 取 1 点高斯（减缩积分，避免剪切自锁）
const geo = joinpath(@__DIR__, "beam.geo")
const msh = joinpath(@__DIR__, "beam.msh")

gmsh.initialize()
if isfile(msh)
    gmsh.open(msh)
else
    gmsh.open(geo)        # beam.geo 内含 Mesh 1; Save "beam.msh";
end

entities = getPhysicalGroups()
nodes    = get𝑿ᵢ()
nₚ       = length(nodes)
nᵤ       = 3nₚ                     # 位移块自由度总数（与 Beam 算子里的 size(k,1)÷2 一致）
L        = maximum(nodes.x)

elements = getElements(nodes, entities["Ω"], integrationorder)   # 梁单元 Seg2
println("节点 ", nₚ, " 个，梁单元 ", length(elements), " 个（", typeof(elements[1]), "）")

# ===================== 3. 截面特性与形函数 =====================
prescribe!(elements, :E=>E, :G=>G, :A=>A, :Iy=>Iy, :Iz=>Iz,
                     :Iyy=>Iyy, :Izz=>Izz, :Iyz=>Iyz, :k₂=>k₂, :k₃=>k₃)
set∇𝝭!(elements)          # 同时备好 𝝭 与 ∂𝝭∂x

# ===================== 4. 装配整体刚度矩阵 K（3D.md 第 5、7、8 节）=====================
𝐤 = zeros(6nₚ, 6nₚ)
𝑎₁ = ∫δdᵘKᵍᵘᵘdᵘds => elements        # Kᵘᵘ
𝑎₂ = ∫δdᵘKᵍᵘᵠdᵠds => elements        # Kᵘᵠ（同时装配左下角转置块 Kᵠᵘ）
𝑎₃ = ∫δdᵠKᵍᵠᵠdᵠds => elements        # Kᵠᵠ
𝑎₁(𝐤)
𝑎₂(𝐤)
𝑎₃(𝐤)

# ===================== 5. 边界条件：仅 Γ¹ 固定（罚函数）=====================
# 罚项：∫δdᵢ α (gᵢ − dᵢ) dΓ → 对角刚度 += α NᵢNⱼ，右端项 += α Nᵢ gᵢ（Poi1 单元 N=1、𝑤=1）
#   Γ² 自由，所以这里只处理 Γ¹ 一组；固定端 6 个自由度全部约束。
const α = 1.0e13

elements_Γ¹ = getElements(nodes, entities["Γ¹"])
prescribe!(elements_Γ¹, :α=>α,
                         :g₁=>0.0, :g₂=>0.0, :g₃=>0.0,   # 位移固定
                         :g₄=>0.0, :g₅=>0.0, :g₆=>0.0)   # 转角也固定
set𝝭!(elements_Γ¹)
𝑏₂¹ = ∫δdᵢgᵢdΓ => elements_Γ¹

𝐤ᵦ = copy(𝐤)                       # 加罚项后的刚度矩阵（保留原 𝐤 备用）
fα = zeros(6nₚ)                    # 罚项对右端项的贡献（齐次固定端时为 0）
𝑏₂¹(𝐤ᵦ, fα)

# ===================== 6. 两个算例：d = 𝐤 \ f =====================
𝐼ᵗ = argmax(nodes.x)               # 自由端节点（x = L）

# --- 算例 1：整体 Y 向均布荷载 q ---
f = zeros(6nₚ)
prescribe!(elements, :b₁=>0.0, :b₂=>q, :b₃=>0.0)   # b 是整体坐标分量上的线荷载 (N/m)
𝑏₁ = ∫vᵢbᵢdΩ => elements                            # f[3I-1] += ∫ N_I b₂ ds
𝑏₁(f)
f .+= fα
d₁ = 𝐤ᵦ \ f

# --- 算例 2：自由端集中力 P ---
# Γ² 是自由端 ⇒ 正好用端部集中力算子；Poi1 单元 𝑤 = 1，tᵢ 是整体分量
elements_Γ² = getElements(nodes, entities["Γ²"])
set𝝭!(elements_Γ²)
prescribe!(elements_Γ², :t₁=>0.0, :t₂=>P, :t₃=>0.0)
𝑏₃ = ∫vᵢtᵢdΓ => elements_Γ²
f = zeros(6nₚ)
𝑏₃(f)                              # f[3I-2..3I] += tᵢ
f .+= fα
d₂ = 𝐤ᵦ \ f

# ===================== 7. 结果与解析解对照 =====================
# 悬臂梁（Timoshenko）：弯曲项 + 剪切项
v均布解析 = q*L^4/(8*E*Iyy) + q*L^2/(2*k₃*G*A)
v集中解析 = P*L^3/(3*E*Iyy) + P*L/(k₃*G*A)

println("\n自由端挠度（节点 𝐼 = ", 𝐼ᵗ, "，x = ", nodes[𝐼ᵗ].x, "）")
println("均布荷载 q   : FE = ", d₁[3𝐼ᵗ-1], "  解析 = ", v均布解析,
        "  相对误差 = ", abs(d₁[3𝐼ᵗ-1] - v均布解析)/abs(v均布解析))
println("端部集中力 P : FE = ", d₂[3𝐼ᵗ-1], "  解析 = ", v集中解析,
        "  相对误差 = ", abs(d₂[3𝐼ᵗ-1] - v集中解析)/abs(v集中解析))

# 解写回节点（分块排列 ⇒ 每个节点 6 个分量），便于后续 VTK / 误差统计
push!(nodes,
      :u₁=>[d₂[3i-2] for i in 1:nₚ], :u₂=>[d₂[3i-1] for i in 1:nₚ], :u₃=>[d₂[3i] for i in 1:nₚ],
      :φ₁=>[d₂[nᵤ+3i-2] for i in 1:nₚ], :φ₂=>[d₂[nᵤ+3i-1] for i in 1:nₚ], :φ₃=>[d₂[nᵤ+3i] for i in 1:nₚ])

gmsh.finalize()
