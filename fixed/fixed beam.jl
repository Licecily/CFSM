
using ApproxOperator
import ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
import ApproxOperator.Beam: ∫εᵘNᵘdΩ, ∫εᵘNᵠdΩ, ∫εᵠNᵘdΩ, ∫εᵠNᵠdΩ, ∫uᵢgᵢdΓ, ∫φᵢgᵢdΓ
import ApproxOperator.Elasticity: ∫vᵢbᵢdΩ, ∫vᵢtᵢdΓ
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

# 3D.md 7.2 的局部坐标系约定（参考向量取整体 Z 轴）下，沿整体 X 的梁为
#   局部 y = 整体 Z、局部 z = −整体 Y
# 所以整体 Y 向荷载由 EIyy 与 k₃ 控制（整体 Z 向荷载才是 EIzz、k₂）。

# ===================== 2. 读入 / 生成网格 =====================
const integrationorder = 1        # Seg2 取 1 点高斯（减缩积分，避免剪切自锁，见 3D.md 注）
const geo = "msh/beam.geo"
const msh = "msh/beam.msh"

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
prescribe!(elements, :E=>E, :G=>G, :A=>A, :Iy=>Iy, :Iz=>Iz,:Iyy=>Iyy, :Izz=>Izz, :Iyz=>Iyz, :k₂=>k₂, :k₃=>k₃)
set∇𝝭!(elements)          # 同时备好 𝝭 与 ∂𝝭∂x

# ===================== 4. 装配整体刚度矩阵 K（3D.md 第 5、7、8 节）=====================
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

# ===================== 5. 边界条件：Γ¹、Γ² 两端固定（罚函数）=====================
# 罚项：∫δdᵢ α (gᵢ − dᵢ) dΓ → 对角刚度 += α NᵢNⱼ，右端项 += α Nᵢ gᵢ（Poi1 单元 N=1、𝑤=1）
#   约束残差 ~ 1/α；α 越大越准，但条件数 ~ α/λ_min 变差。此处 α/最大对角元 ~ 1e3。
#   每组各调用一次 getElements（组内共享 data）；两组不能 vcat 后再 set𝝭!。
const α = 1.0e13

elements_Γ¹ = getElements(nodes, entities["Γ¹"])        # entities[name] 本身就是 dim => tags
prescribe!(elements_Γ¹, :α=>α,:g₁=>0.0, :g₂=>0.0, :g₃=>0.0,:g₄=>0.0, :g₅=>0.0, :g₆=>0.0)  
set𝝭!(elements_Γ¹)
𝑏₂¹ᵘ = ∫uᵢgᵢdΓ => elements_Γ¹
𝑏₂¹ᵠ = ∫φᵢgᵢdΓ => elements_Γ¹

elements_Γ² = getElements(nodes, entities["Γ²"])
prescribe!(elements_Γ², :α=>α,:g₁=>0.0, :g₂=>0.0, :g₃=>0.0,:g₄=>0.0, :g₅=>0.0, :g₆=>0.0)
set𝝭!(elements_Γ²)
𝑏₂²ᵘ = ∫uᵢgᵢdΓ => elements_Γ²
𝑏₂²ᵠ = ∫φᵢgᵢdΓ => elements_Γ²

𝐊ᵘᵘα = zeros(3nₚ, 3nₚ)
𝐊ᵠᵠα = zeros(3nₚ, 3nₚ)
fᵘα = zeros(3nₚ)
fᵠα = zeros(3nₚ)
𝑏₂¹ᵘ(𝐊ᵘᵘα, fᵘα); 𝑏₂¹ᵠ(𝐊ᵠᵠα, fᵠα)
𝑏₂²ᵘ(𝐊ᵘᵘα, fᵘα); 𝑏₂²ᵠ(𝐊ᵠᵠα, fᵠα)
𝐊 = [𝐊ᵘᵘ + 𝐊ᵘᵘα  𝐊ᵘᵠ;
      𝐊ᵠᵘ          𝐊ᵠᵠ + 𝐊ᵠᵠα]

# ===================== 6. 两个算例：d = 𝐊 \ f =====================
𝐼ₘ = argmin(abs.(nodes.x .- L/2))        # 跨中节点（beam.geo 的 n 取偶数则有节点落在跨中）

# --- 算例 1：整体 Y 向均布荷载 q ---
fᵘ = zeros(3nₚ)
prescribe!(elements, :b₁=>0.0, :b₂=>q, :b₃=>0.0)   # b 是整体坐标分量上的线荷载 (N/m)
𝑏₁ = ∫vᵢbᵢdΩ => elements                            # f[3I-1] += ∫ N_I b₂ ds
𝑏₁(fᵘ)
f = [fᵘ .+ fᵘα; fᵠα]
d₁ = 𝐊 \ f

# --- 算例 2：跨中集中力 P ---
# 这里直接写自由度；若要用算子，见第 8 节"端部集中力"的写法。
fᵘ = zeros(3nₚ)
fᵘ[3𝐼ₘ-1] = P
f = [fᵘ .+ fᵘα; fᵠα]
d₂ = 𝐊 \ f

# ===================== 7. 全梁精确解与逐节点对照 =====================
# 梁沿整体 X，局部 y = 整体 Z、局部 z = −整体 Y（3D.md 7.2），所以整体 Y 向
# 弯曲由 EIyy 与 k₃GA 控制，对应的截面转角是整体 φ₃。取 x ∈ [0, L]，
# q、P 为整体 Y 向荷载（向下为负）。其余分量恒为零：u₁ = u₃ = φ₁ = φ₂ = 0。

# --- 算例 1：两端固定，均布荷载 q（Timoshenko 梁，弯曲 + 剪切）---
#   弯曲挠度  w_b(x) = q x²(L−x)²/(24EIyy)
#   剪切挠度  w_s(x) = q x(L−x)/(2k₃GA)
#   截面转角  φ(x)   = w_b′(x) = q x(L−x)(L−2x)/(12EIyy)
w_uniform(x) = q*x^2*(L-x)^2/(24*E*Iyy) + q*x*(L-x)/(2*k₃*G*A)
φ_uniform(x) = q*x*(L-x)*(L-2*x)/(12*E*Iyy)

# --- 算例 2：两端固定，跨中集中力 P（Timoshenko 梁，弯曲 + 剪切）---
#   0 ≤ x ≤ L/2：w_b = P x²(3L−4x)/(48EIyy)，w_s = P x/(2k₃GA)
#                φ   = w_b′ = P x(L−2x)/(8EIyy)
#   L/2 ≤ x ≤ L：由对称性 w(x) = w(L−x)，φ(x) = P(L−x)(L−2x)/(8EIyy)，
#                w_b = P(L−x)²(4x−L)/(48EIyy)，w_s = P(L−x)/(2k₃GA)
w_point(x) = x ≤ L/2 ? P*x^2*(3L-4*x)/(48*E*Iyy) + P*x/(2*k₃*G*A) :
                     P*(L-x)^2*(4*x-L)/(48*E*Iyy) + P*(L-x)/(2*k₃*G*A)
φ_point(x) = x ≤ L/2 ? P*x*(L-2*x)/(8*E*Iyy) : P*(L-x)*(L-2*x)/(8*E*Iyy)

# --- 有限元解（u₂、φ₃）与精确解在全梁所有节点上的对比 ---
𝑥    = [nodes[i].x for i in 1:nₚ]
sorted_idx = sortperm(𝑥)

wᶠ¹ = [d₁[3i-1]      for i in 1:nₚ]      # 均布荷载：FE 位移
φᶠ¹ = [d₁[nᵤ+3i]     for i in 1:nₚ]      # 均布荷载：FE 转角
wᵉ¹ = w_uniform.(𝑥)
φᵉ¹ = φ_uniform.(𝑥)

wᶠ² = [d₂[3i-1]      for i in 1:nₚ]      # 集中力：FE 位移
φᶠ² = [d₂[nᵤ+3i]     for i in 1:nₚ]      # 集中力：FE 转角
wᵉ² = w_point.(𝑥)
φᵉ² = φ_point.(𝑥)

function print_comparison(title, 𝑥, sorted_idx, wᶠ, wᵉ, φᶠ, φᵉ)
    println("\n", title)
    println("      x          w_FE          w_exact       |Δw|          φ_FE          φ_exact       |Δφ|")
    for k in sorted_idx
        @printf("%9.4f  %13.6e  %13.6e  %10.3e  %13.6e  %13.6e  %10.3e\n",
                𝑥[k], wᶠ[k], wᵉ[k], abs(wᶠ[k]-wᵉ[k]),
                φᶠ[k], φᵉ[k], abs(φᶠ[k]-φᵉ[k]))
    end
    # 端点处精确解为 0，故相对误差以精确解的最大幅值为基准
    𝑒ʷ = maximum(abs.(wᶠ .- wᵉ))/maximum(abs.(wᵉ))
    𝑒ᵠ = maximum(abs.(φᶠ .- φᵉ))/maximum(abs.(φᵉ))
    @printf("全梁最大相对误差：w %.3e，φ %.3e\n", 𝑒ʷ, 𝑒ᵠ)
end

print_comparison("=== 算例 1：均布荷载 q = $q N/m ===", 𝑥, sorted_idx, wᶠ¹, wᵉ¹, φᶠ¹, φᵉ¹)
print_comparison("=== 算例 2：跨中集中力 P = $P N ===", 𝑥, sorted_idx, wᶠ², wᵉ², φᶠ², φᵉ²)

# 解写回节点（分块排列 ⇒ 每个节点 6 个分量），便于后续 VTK / 误差统计
push!(nodes, :u₁=>[d₂[3i-2] for i in 1:nₚ], :u₂=>[d₂[3i-1] for i in 1:nₚ], :u₃=>[d₂[3i] for i in 1:nₚ],:φ₁=>[d₂[nᵤ+3i-2] for i in 1:nₚ], :φ₂=>[d₂[nᵤ+3i-1] for i in 1:nₚ], :φ₃=>[d₂[nᵤ+3i] for i in 1:nₚ])

gmsh.finalize()
