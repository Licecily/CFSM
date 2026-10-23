using ApproxOperator
import ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
import ApproxOperator.Beam: ∫εᵘNᵘdΩ, ∫εᵘNᵠdΩ, ∫εᵠNᵘdΩ, ∫εᵠNᵠdΩ, L₂d
import ApproxOperator.Elasticity: ∫vᵢbᵢdΩ
import Gmsh: gmsh
using LinearAlgebra
using Printf

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
const P = -1.0e3          # 跨中集中力 N

# 沿整体 X 的梁：局部 y = 整体 Z、局部 z = −整体 Y（3D.md 7.2），
# 整体 Y 向弯曲由 EIyy 与 k₃GA 控制，对应截面转角 φ₃。

# ===================== 2. 读入网格（沿 X 的梁）=====================
const integrationorder = 1        # Seg2 取 1 点高斯（减缩积分，避免剪切自锁）
const msh = "msh/beamX.msh"

gmsh.initialize()
isfile(msh) || error("找不到网格文件：$msh")
gmsh.open(msh)

entities = getPhysicalGroups()
nodes    = get𝑿ᵢ()
nₚ       = length(nodes)
nᵤ       = 3nₚ
L        = maximum(nodes.x)

elements = getElements(nodes, entities["Ω"], integrationorder)
println("节点 ", nₚ, " 个，梁单元 ", length(elements), " 个（", typeof(elements[1]), "）")

# ===================== 3. 截面特性与形函数 =====================
prescribe!(elements, :E=>E, :G=>G, :A=>A, :Iy=>Iy, :Iz=>Iz,
                     :Iyy=>Iyy, :Izz=>Izz, :Iyz=>Iyz, :k₂=>k₂, :k₃=>k₃)
set∇𝝭!(elements)

# ===================== 4. 装配整体刚度矩阵 K =====================
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

# ===================== 5. 简支边界（罚函数）=====================
# 简支梁：Γ¹ 铰支座，Γ² 滚动支座。
#   Γ¹（x = 0）：u₁ = u₂ = u₃ = 0，φ₁ = 0（约束绕梁轴的刚体扭转）
#   Γ²（x = L）：u₂ = u₃ = 0（u₁、φ₁ 自由 ⇒ 滚动支座）
# 两个支座都约束 u₂（竖向）与 u₃（侧向），恰好去掉全部 6 个刚体模态；
# 弯曲转角 φ₂、φ₃ 不约束 ⇒ 端部弯矩为零，这正是"简支"。
# 本算例荷载只有整体 Y 向分量，u₃、φ₂ 的响应恒为零。
#
# 只约束部分自由度，不能直接用 ∫dᵢgᵢdΓ（该算子一次约束全部 6 个分量）；
# 对单节点的 Poi1 罚项，N = 1、𝑤 = 1，齐次约束 g = 0 时等价于在对应
# 自由度对角线上加 α。这里直接按自由度编号加罚。
const α = 1.0e15

𝑥  = [nodes[i].x for i in 1:nₚ]
i¹ = argmin(abs.(𝑥 .- 0.0))        # 左端节点
i² = argmin(abs.(𝑥 .- L))          # 右端节点

𝐤ᵦ = copy(𝐤)

# Γ¹ 铰支座：u₁、u₂、u₃、φ₁
for p in (3i¹-2, 3i¹-1, 3i¹, nᵤ+3i¹-2)
    𝐤ᵦ[p,p] += α
end
# Γ² 滚动支座：u₂、u₃
for p in (3i²-1, 3i²)
    𝐤ᵦ[p,p] += α
end

# 齐次约束 ⇒ 右端项无罚项贡献
fα = zeros(6nₚ)

# ===================== 6. 两个算例 =====================
# --- 算例 1：整体 Y 向均布荷载 q ---
f = zeros(6nₚ)
prescribe!(elements, :b₁=>0.0, :b₂=>q, :b₃=>0.0)
𝑏₁ = ∫vᵢbᵢdΩ => elements
𝑏₁(f)
f .+= fα
d₁ = 𝐤ᵦ \ f

# --- 算例 2：跨中集中力 P ---
𝐼ₘ = argmin(abs.(𝑥 .- L/2))        # 跨中节点（网格 n 为偶数）
f = zeros(6nₚ)
f[3𝐼ₘ-1] = P
f .+= fα
d₂ = 𝐤ᵦ \ f

# ===================== 7. 整梁精确解 =====================
# 简支梁（0 ≤ x ≤ L，两端 w = 0、弯矩为零），Timoshenko：弯曲 + 剪切。
# 与 fixed beam.jl 采用同一套符号约定（q、P 为整体 Y 向荷载，向下为负）。
#
# 均布荷载 q：
#   弯矩 M = −q x(L−x)/2 ⇒ φ = w_b′ = q(L³ − 6Lx² + 4x³)/(24EIyy)
#   w_b = q x(L³ − 2Lx² + x³)/(24EIyy)，w_s = q x(L−x)/(2k₃GA)
#   w  = w_b + w_s
w均布(x) = q*x*(L^3 - 2*L*x^2 + x^3)/(24*E*Iyy) + q*x*(L-x)/(2*k₃*G*A)
φ均布(x) = q*(L^3 - 6*L*x^2 + 4*x^3)/(24*E*Iyy)

# 跨中集中力 P（利用对称性，x̄ = L − x 表示镜像点）：
#   0 ≤ x ≤ L/2：w_b = P x(3L² − 4x²)/(48EIyy)，w_s = P x/(2k₃GA)
#                φ   = P(L² − 4x²)/(16EIyy)
#   L/2 ≤ x ≤ L：w(x) = w(L−x)，φ(x) = −φ(L−x)
w集中(x) = x <= L/2 ? P*x*(3*L^2 - 4*x^2)/(48*E*Iyy) + P*x/(2*k₃*G*A) :
                       P*(L-x)*(3*L^2 - 4*(L-x)^2)/(48*E*Iyy) + P*(L-x)/(2*k₃*G*A)
φ集中(x) = x <= L/2 ? P*(L^2 - 4*x^2)/(16*E*Iyy) :
                      -P*(L^2 - 4*(L-x)^2)/(16*E*Iyy)

# ===================== 8. 整梁分析（逐节点 + L₂ 误差）=====================
function 整梁分析(名称, d, wᵉ, φᵉ, nodes, entities, nₚ, nᵤ)
    # 有限元解写回节点，供 L₂d 读取（d₁..d₃ = u₁..u₃，d₄..d₆ = φ₁..φ₃）
    push!(nodes,
          :d₁=>[d[3i-2] for i in 1:nₚ], :d₂=>[d[3i-1] for i in 1:nₚ], :d₃=>[d[3i] for i in 1:nₚ],
          :d₄=>[d[nᵤ+3i-2] for i in 1:nₚ], :d₅=>[d[nᵤ+3i-1] for i in 1:nₚ], :d₆=>[d[nᵤ+3i] for i in 1:nₚ])

    𝑥    = [nodes[i].x for i in 1:nₚ]
    次序 = sortperm(𝑥)

    # 精确解只有 u₂、φ₃ 非零
    wᶠ = [d[3i-1]  for i in 1:nₚ]
    φᶠ = [d[nᵤ+3i] for i in 1:nₚ]
    wᵉˡ = wᵉ.(𝑥)
    φᵉˡ = φᵉ.(𝑥)

    println("\n", 名称)
    println("      x          w_FE          w_exact       |Δw|          φ_FE          φ_exact       |Δφ|")
    for k in 次序
        @printf("%9.4f  %13.6e  %13.6e  %10.3e  %13.6e  %13.6e  %10.3e\n",
                𝑥[k], wᶠ[k], wᵉˡ[k], abs(wᶠ[k]-wᵉˡ[k]),
                φᶠ[k], φᵉˡ[k], abs(φᶠ[k]-φᵉˡ[k]))
    end
    # 端点精确值不全为零时，相对误差仍以精确解的最大幅值为基准
    𝑒ʷ = maximum(abs.(wᶠ .- wᵉˡ))/maximum(abs.(wᵉˡ))
    𝑒ᵠ = maximum(abs.(φᶠ .- φᵉˡ))/maximum(abs.(φᵉˡ))
    @printf("全梁最大相对误差：w %.3e，φ %.3e\n", 𝑒ʷ, 𝑒ᵠ)

    # 域内 L₂ 误差（高阶 Gauss 积分，避免积分误差掩盖离散误差）
    err = getElements(nodes, entities["Ω"], 10)
    prescribe!(err, :u₁=>(x,y,z)->0.0, :u₂=>(x,y,z)->wᵉ(x),
                    :u₃=>(x,y,z)->0.0, :φ₁=>(x,y,z)->0.0,
                    :φ₂=>(x,y,z)->0.0, :φ₃=>(x,y,z)->φᵉ(x))
    set𝝭!(err)

    Δ², 范² = L₂d(err)
    相对u = sqrt(sum(Δ²[1:3])) / sqrt(sum(范²[1:3]))
    相对φ = sqrt(sum(Δ²[4:6])) / sqrt(sum(范²[4:6]))
    @printf("相对 L₂ 误差：位移 %.3e，转角 %.3e\n", 相对u, 相对φ)

    return 相对u, 相对φ
end

# ===================== 9. 执行分析 =====================
println("=== 简支梁：均布荷载 q = $q N/m ===")
结果q = 整梁分析("算例 1：均布荷载 q", d₁, w均布, φ均布, nodes, entities, nₚ, nᵤ)

println("\n=== 简支梁：跨中集中力 P = $P N ===")
结果P = 整梁分析("算例 2：跨中集中力 P", d₂, w集中, φ集中, nodes, entities, nₚ, nᵤ)

gmsh.finalize()
