
using ApproxOperator
import ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
import ApproxOperator.Beam: ∫δdᵘKᵍᵘᵘdᵘds, ∫δdᵘKᵍᵘᵠdᵠds, ∫δdᵠKᵍᵠᵠdᵠds, ∫δdᵢgᵢdΓ
import ApproxOperator.Elasticity: ∫vᵢbᵢdΩ, ∫vᵢtᵢdΓ
import Gmsh: gmsh
using LinearAlgebra
using WriteVTK

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
prescribe!(elements, :E=>E, :G=>G, :A=>A, :Iy=>Iy, :Iz=>Iz,:Iyy=>Iyy, :Izz=>Izz, :Iyz=>Iyz, :k₂=>k₂, :k₃=>k₃)
set∇𝝭!(elements)          # 同时备好 𝝭 与 ∂𝝭∂x

# ===================== 4. 装配整体刚度矩阵 K（3D.md 第 5、7、8 节）=====================
𝐤 = zeros(6nₚ, 6nₚ)
𝑎₁ = ∫δdᵘKᵍᵘᵘdᵘds => elements        # Kᵘᵘ
𝑎₂ = ∫δdᵘKᵍᵘᵠdᵠds => elements        # Kᵘᵠ（同时装配左下角转置块 Kᵠᵘ）
𝑎₃ = ∫δdᵠKᵍᵠᵠdᵠds => elements        # Kᵠᵠ
𝑎₁(𝐤)
𝑎₂(𝐤)
𝑎₃(𝐤)

# ===================== 5. 边界条件：Γ¹、Γ² 两端固定（罚函数）=====================
# 罚项：∫δdᵢ α (gᵢ − dᵢ) dΓ → 对角刚度 += α NᵢNⱼ，右端项 += α Nᵢ gᵢ（Poi1 单元 N=1、𝑤=1）
#   约束残差 ~ 1/α；α 越大越准，但条件数 ~ α/λ_min 变差。此处 α/最大对角元 ~ 1e3。
#   每组各调用一次 getElements（组内共享 data）；两组不能 vcat 后再 set𝝭!。
const α = 1.0e13

elements_Γ¹ = getElements(nodes, entities["Γ¹"])        # entities[name] 本身就是 dim => tags
prescribe!(elements_Γ¹, :α=>α,:g₁=>0.0, :g₂=>0.0, :g₃=>0.0,:g₄=>0.0, :g₅=>0.0, :g₆=>0.0)  
set𝝭!(elements_Γ¹)
𝑏₂¹ = ∫δdᵢgᵢdΓ => elements_Γ¹

elements_Γ² = getElements(nodes, entities["Γ²"])
prescribe!(elements_Γ², :α=>α,:g₁=>0.0, :g₂=>0.0, :g₃=>0.0,:g₄=>0.0, :g₅=>0.0, :g₆=>0.0)
set𝝭!(elements_Γ²)
𝑏₂² = ∫δdᵢgᵢdΓ => elements_Γ²

𝐤ᵦ = copy(𝐤)                       # 加罚项后的刚度矩阵（保留原 𝐤 备用）
fα = zeros(6nₚ)                    # 罚项对右端项的贡献（齐次固定端时为 0）
𝑏₂¹(𝐤ᵦ, fα)
𝑏₂²(𝐤ᵦ, fα)

# ===================== 6. 两个算例：d = 𝐤 \ f =====================
𝐼ₘ = argmin(abs.(nodes.x .- L/2))        # 跨中节点（beam.geo 的 n 取偶数则有节点落在跨中）

# --- 算例 1：整体 Y 向均布荷载 q ---
f = zeros(6nₚ)
prescribe!(elements, :b₁=>0.0, :b₂=>q, :b₃=>0.0)   # b 是整体坐标分量上的线荷载 (N/m)
𝑏₁ = ∫vᵢbᵢdΩ => elements                            # f[3I-1] += ∫ N_I b₂ ds
𝑏₁(f)
f .+= fα
d₁ = 𝐤ᵦ \ f

# --- 算例 2：跨中集中力 P ---
# 这里直接写自由度；若要用算子，见第 8 节"端部集中力"的写法。
f = zeros(6nₚ)
f[3𝐼ₘ-1] = P
f .+= fα
d₂ = 𝐤ᵦ \ f

# ===================== 7. 结果与解析解对照 =====================
v均布解析 = q*L^4/(384*E*Iyy) + q*L^2/(8*k₃*G*A)
v集中解析 = P*L^3/(192*E*Iyy) + P*L/(4*k₃*G*A)

println("\n跨中挠度（节点 𝐼 = ", 𝐼ₘ, "，x = ", nodes[𝐼ₘ].x, "）")
println("均布荷载 q : FE = ", d₁[3𝐼ₘ-1], "  解析 = ", v均布解析,
        "  相对误差 = ", abs(d₁[3𝐼ₘ-1] - v均布解析)/abs(v均布解析))
println("集中力 P   : FE = ", d₂[3𝐼ₘ-1], "  解析 = ", v集中解析,
        "  相对误差 = ", abs(d₂[3𝐼ₘ-1] - v集中解析)/abs(v集中解析))

# 解写回节点（分块排列 ⇒ 每个节点 6 个分量），便于后续 VTK / 误差统计
push!(nodes,
      :u₁=>[d₂[3i-2] for i in 1:nₚ], :u₂=>[d₂[3i-1] for i in 1:nₚ], :u₃=>[d₂[3i] for i in 1:nₚ],
      :φ₁=>[d₂[nᵤ+3i-2] for i in 1:nₚ], :φ₂=>[d₂[nᵤ+3i-1] for i in 1:nₚ], :φ₃=>[d₂[nᵤ+3i] for i in 1:nₚ])

# ===================== 8. VTK 输出（CFSM/vtk/，WriteVTK）=====================
# 梁单元 → VTK_LINE；节点数据含 displacement / rotation / u₂
#   变形形状用 ParaView 的 Warp By Vector（Scale Factor 取 ~1e5）
function writevtkbeam(file, nodes, elements, d, nᵤ)
    mkpath(dirname(file))
    nₚ = length(nodes)
    x = [p.x for p in nodes]
    y = [p.y for p in nodes]
    z = [p.z for p in nodes]
    # WriteVTK 的节点编号从 1 开始，与 gmsh 的 𝐼 一致
    cells = [MeshCell(VTKCellTypes.VTK_LINE, (el.𝓒[1].𝐼, el.𝓒[2].𝐼)) for el in elements]
    u = zeros(3, nₚ)
    φ = zeros(3, nₚ)
    for i in 1:nₚ
        u[:, i] = d[3i-2:3i]              # 整体 u₁, u₂, u₃
        φ[:, i] = d[nᵤ+3i-2:nᵤ+3i]        # 整体 φ₁, φ₂, φ₃
    end
    files = vtk_grid(file, x, y, z, cells) do vtk
        vtk["displacement"] = u           # 3×nₚ 矩阵 ⇒ 节点向量场
        vtk["rotation"]     = φ
        vtk["u2"]           = u[2, :]     # 标量场，便于直接着色
    end
    println("VTK 已写出: ", join(files, ", "))
end

vtkdir = joinpath(@__DIR__, "vtk")
writevtkbeam(joinpath(vtkdir, "fixed_beam_uniform"), nodes, elements, d₁, nᵤ)
writevtkbeam(joinpath(vtkdir, "fixed_beam_point"),   nodes, elements, d₂, nᵤ)

# ===================== 9. 备用：端部集中力 / 集中力矩 =====================
# 端部集中力（例如把 Γ² 改成自由端后在该端加力）：Poi1 单元的 𝑤 = 1，可直接用 Elasticity 的算子
#   els_Γ² = getElements(nodes, entities["Γ²"])
#   set𝝭!(els_Γ²)
#   prescribe!(els_Γ², :t₁=>0.0, :t₂=>P, :t₃=>0.0)
#   (∫vᵢtᵢdΓ=>els_Γ²)(f)         # f[3I-2..3I] += tᵢ（局部量先乘 Rᵀ 转成整体）
# 端部集中力矩没有现成算子（∫vᵢtᵢdΓ 只写 3I-2..3I 三个平动自由度），直接写自由度即可：
#   f[nᵤ+3I-2] = M₁   f[nᵤ+3I-1] = M₂   f[nᵤ+3I] = M₃

gmsh.finalize()
