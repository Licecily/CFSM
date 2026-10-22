using CairoMakie
using Printf

# 跨中集中力 P 工况的相对 L2 误差数据（fixed beam convergence P.jl）
n  = [10.0, 20.0, 40.0, 80.0]
eu = [3.101360e-02, 7.798033e-03, 1.952074e-03, 4.879652e-04]
ep = [3.999986e-02, 9.999860e-03, 2.499860e-03, 6.248600e-04]

# 双对数坐标下最小二乘拟合的斜率
function 拟合斜率(n, err)
    x = log10.(n)
    y = log10.(err)
    x̄ = sum(x) / length(x)
    ȳ = sum(y) / length(y)
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄).^2)
end

斜率u = 拟合斜率(n, eu)
斜率p = 拟合斜率(n, ep)

CairoMakie.set_theme!(
    fonts = (; regular = "Microsoft YaHei", bold = "Microsoft YaHei",
               italic = "Microsoft YaHei", bolditalic = "Microsoft YaHei"),
    fontsize = 16)

fig = Figure(size = (820, 560))
ax  = Axis(fig[1, 1])

# 先画原始数据，再设对数轴
scatterlines!(ax, n, eu, markersize = 11, linewidth = 2.2,
              color = :dodgerblue, label = "位移")
scatterlines!(ax, n, ep, markersize = 11, linewidth = 2.2,
              color = :orange, marker = :rect, label = "转角")

ax.xscale = log10
ax.yscale = log10
ax.xticks = n
ax.xlabel = "单元数   log₁₀(n)"
ax.ylabel = "误差     log₁₀(L₂)"
ax.title  = "固支梁（集中力）"

# 在两条曲线附近标注拟合斜率（x 取 n=40 处，上下错开）
text!(ax, 40, 0.8 * eu[3], text = @sprintf("kᵤ= %.2f", 斜率u),
      color = :dodgerblue, align = (:right, :top), fontsize = 14)
text!(ax, 40, 1.8 * ep[3], text = @sprintf("kᵩ= %.2f", 斜率p),
      color = :orange, align = (:right, :bottom), fontsize = 14)

axislegend(ax, position = :rt, framevisible = false)

save(joinpath(@__DIR__, "l2_error_P.png"), fig, px_per_unit = 3)
save(joinpath(@__DIR__, "l2_error_P.pdf"), fig)

println("已输出 l2_error_P.png/pdf（集中力，位移+转角同坐标系）")
