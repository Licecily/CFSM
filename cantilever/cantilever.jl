using ApproxOperator
import ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
import ApproxOperator.Beam: ∫δdᵘKᵍᵘᵘdᵘds, ∫δdᵘKᵍᵘᵠdᵠds, ∫δdᵠKᵍᵠᵠdᵠds, ∫δdᵢgᵢdΓ, L₂d
import ApproxOperator.Elasticity: ∫vᵢbᵢdΩ, ∫vᵢtᵢdΓ
import Gmsh: gmsh
using LinearAlgebra
using Printf

# ===================== 1. 材料与截面参数 =====================
const E   = 2.1e11        # 弹性模量
const G   = 8.1e10        # 剪切模量
const A   = 1.0e-2        # 截面面积
const Iyy = 4.0e-4        # 绕局部 y 轴的惯性矩
const Izz = 6.0e-4        # 绕局部 z 轴的惯性矩
const Iy  = 0.0
const Iz  = 0.0
const Iyz = 0.0
const k₂  = 5/6
const k₃  = 5/6

# 荷载（整体 Y 向，负号朝下）
const q = -1.0e3          # 均布线荷载 N/m
const P = -1.0e3          # 自由端集中力 N

# ===================== 2. 读入网格（沿 X 的梁，与 fixed 算例共用）=====================
const integrationorder = 1
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
𝐤 = zeros(6nₚ, 6nₚ)
𝑎₁ = ∫δdᵘKᵍᵘᵘdᵘds => elements
𝑎₂ = ∫δdᵘKᵍᵘᵠdᵠds => elements
𝑎₃ = ∫δdᵠKᵍᵠᵠdᵠds => elements
𝑎₁(𝐤)
𝑎₂(𝐤)
𝑎₃(𝐤)

# ===================== 5. 边界条件：仅 Γ¹ 固定（罚函数）=====================
# Γ² 自由端不约束；固定端 6 个自由度全部约束。
# α 取 1e15，使端部约束残差远低于网格离散误差。
const α = 1.0e15

elements_Γ¹ = getElements(nodes, entities["Γ¹"])
prescribe!(elements_Γ¹, :α=>α,
                         :g₁=>0.0, :g₂=>0.0, :g₃=>0.0,
                         :g₄=>0.0, :g₅=>0.0, :g₆=>0.0)
set𝝭!(elements_Γ¹)
𝑏₂¹ = ∫δdᵢgᵢdΓ => elements_Γ¹

𝐤ᵦ = copy(𝐤)
fα = zeros(6nₚ)
𝑏₂¹(𝐤ᵦ, fα)

# ===================== 6. 两个算例 =====================
# --- 算例 1：整体 Y 向均布荷载 q ---
f = zeros(6nₚ)
prescribe!(elements, :b₁=>0.0, :b₂=>q, :b₃=>0.0)
𝑏₁ = ∫vᵢbᵢdΩ => elements
𝑏₁(f)
f .+= fα
d₁ = 𝐤ᵦ \ f

# --- 算例 2：自由端集中力 P ---
elements_Γ² = getElements(nodes, entities["Γ²"])
set𝝭!(elements_Γ²)
prescribe!(elements_Γ², :t₁=>0.0, :t₂=>P, :t₃=>0.0)
𝑏₃ = ∫vᵢtᵢdΓ => elements_Γ²
f = zeros(6nₚ)
𝑏₃(f)
f .+= fα
d₂ = 𝐤ᵦ \ f

# ===================== 7. 整梁精确解 =====================
# 悬臂梁（x=0 固定，x=L 自由），Timoshenko：弯曲 + 剪切
#
# 均布荷载 q：
#   w_b = q x²(6L² − 4Lx + x²)/(24EIyy)
#   w_s = q x(2L − x)/(2k₃GA)
#   φ   = w_b′ = q x(3L² − 3Lx + x²)/(6EIyy)
wᵍ(x) = q*x^2*(6L^2 - 4L*x + x^2)/(24*E*Iyy) + q*x*(2L - x)/(2*k₃*G*A)
φᵍ(x) = q*x*(3L^2 - 3L*x + x^2)/(6*E*Iyy)

# 自由端集中力 P：
#   w_b = P x²(3L − x)/(6EIyy)，w_s = P x/(k₃GA)
#   φ   = w_b′ = P x(2L − x)/(2EIyy)
wᴾ(x) = P*x^2*(3L - x)/(6*E*Iyy) + P*x/(k₃*G*A)
φᴾ(x) = P*x*(2L - x)/(2*E*Iyy)

# ===================== 8. 整梁分析（逐节点 + L₂ 误差）=====================
function 整梁分析(名称, d, wᵉ, φᵉ, nodes, entities, nₚ, nᵤ)
    # 有限元解写回节点，供 L₂d 读取（d₁..d₃ = u₁..u₃，d₄..d₆ = φ₁..φ₃）
    push!(nodes,
          :d₁=>[d[3i-2] for i in 1:nₚ], :d₂=>[d[3i-1] for i in 1:nₚ], :d₃=>[d[3i] for i in 1:nₚ],
          :d₄=>[d[nᵤ+3i-2] for i in 1:nₚ], :d₅=>[d[nᵤ+3i-1] for i in 1:nₚ], :d₆=>[d[nᵤ+3i] for i in 1:nₚ])

    # 节点 i 的 6 个自由度；精确解只有 u₂、φ₃ 非零
    𝐝(i)  = vcat(d[3i-2:3i], d[nᵤ+3i-2:nᵤ+3i])
    𝐝ᵉ(i) = [0.0, wᵉ(nodes[i].x), 0.0, 0.0, 0.0, φᵉ(nodes[i].x)]

    𝑥     = [nodes[i].x for i in 1:nₚ]
    次序  = sortperm(𝑥)
    Δ     = [𝐝(i) .- 𝐝ᵉ(i) for i in 1:nₚ]
    场幅值 = maximum(abs.(vcat([𝐝ᵉ(i) for i in 1:nₚ]...)))
    最大误差 = maximum([maximum(abs.(Δ[i])) for i in 1:nₚ])
    相对误差 = 最大误差 / 场幅值

    println("\n==================== ", 名称, " ====================")
    println("      x        max_c|Δd_c|   (c = u₁,u₂,u₃,φ₁,φ₂,φ₃)")
    for k in 次序
        @printf("%9.4f   %12.3e\n", 𝑥[k], maximum(abs.(Δ[k])))
    end

    分量名 = ("u₁", "u₂", "u₃", "φ₁", "φ₂", "φ₃")
    println("\n各分量最大绝对误差:")
    for c in 1:6
        @printf("  %s : %12.3e\n", 分量名[c], maximum([abs(Δ[i][c]) for i in 1:nₚ]))
    end
    @printf("\n全梁最大绝对误差 %12.3e，相对误差 %.3e\n", 最大误差, 相对误差)

    # 域内 L₂ 误差
    err = getElements(nodes, entities["Ω"], 10)
    prescribe!(err, :u₁=>(x,y,z)->0.0, :u₂=>(x,y,z)->wᵉ(x),
                    :u₃=>(x,y,z)->0.0, :φ₁=>(x,y,z)->0.0,
                    :φ₂=>(x,y,z)->0.0, :φ₃=>(x,y,z)->φᵉ(x))
    set𝝭!(err)

    Δ², 范² = L₂d(err)
    相对u = sqrt(sum(Δ²[1:3])) / sqrt(sum(范²[1:3]))
    相对φ = sqrt(sum(Δ²[4:6])) / sqrt(sum(范²[4:6]))

    println("\n域内 L₂ 误差:")
    for c in 1:6
        if 范²[c] > 0
            @printf("  %s : |e|_L₂ = %12.3e   相对 = %12.3e\n",
                    分量名[c], sqrt(Δ²[c]), sqrt(Δ²[c])/sqrt(范²[c]))
        else
            @printf("  %s : |e|_L₂ = %12.3e   相对 = —\n", 分量名[c], sqrt(Δ²[c]))
        end
    end
    @printf("\n相对 L₂ 误差：位移 %.3e，转角 %.3e\n", 相对u, 相对φ)

    return 相对误差, 相对u, 相对φ
end

# ===================== 9. 执行分析 =====================
结果q = 整梁分析("算例 1：均布荷载 q", d₁, wᵍ, φᵍ, nodes, entities, nₚ, nᵤ)
结果P = 整梁分析("算例 2：自由端集中力 P", d₂, wᴾ, φᴾ, nodes, entities, nₚ, nᵤ)

gmsh.finalize()
